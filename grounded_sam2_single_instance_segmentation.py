import os, sys
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import matplotlib.pyplot as plt

#!sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /home/yangmi/.conda/envs/SAM2/lib/python3.11/site-packages/basicsr/data/degradations.py

import gc
import argparse
import copy
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import AutoProcessor, AutoModelForCausalLM
from utils.supervision_utils import CUSTOM_COLOR_MAP
from grounding_dino.groundingdino.models import build_model
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from grounding_dino.groundingdino.util.slconfig import SLConfig
from grounding_dino.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from huggingface_hub import hf_hub_download
from custom_dataset import GroundedSAM2TestDataset, custom_transform

## annotation
from annotator.hed import HEDdetector, nms
from annotator.util import HWC3, resize_image

import supervision as sv
from scipy import ndimage
from scipy.spatial import distance
import random

import cv2
from typing import Union
import PIL
from PIL import Image, ImageDraw, ImageFont
#from shapely.geometry import Polygon
import requests
import torch
from io import BytesIO
from torchvision import transforms
from torchvision.ops import box_convert
from torch.utils.data import DataLoader

import locale
locale.getpreferredencoding = lambda: "UTF-8"
import traceback
import shutil

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of product position extraction based on Grouned SAM.")

    parser.add_argument("--input_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to data instance.")
    
    parser.add_argument("--data_hed_dir", 
                        default=None, 
                        type=str, 
                        required=False, 
                        help="Path to data hed.")

    parser.add_argument("--output_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to data hed background.")
    
    parser.add_argument("--img_format", 
                        default='png', 
                        type=str, 
                        help="Path to the image.")
    
    parser.add_argument("--gpu_id", 
                        default=0, 
                        type=int, 
                        required=False,
                        help="gpu id")

    parser.add_argument("--product_images",
                    nargs='+', 
                    default=None,
                    required=False,
                    help="The background image with the product")

    parser.add_argument("--similarity_threshold", 
                        default=2.5, #0.916, 
                        type=float, 
                        required=False,
                        help="The threshold to remove hed images")

    parser.add_argument("--hed_value", 
                        default=220, 
                        type=int, 
                        required=False,
                        help="The hed value for product")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args


def touch_border(img_bw, border_type='bottom', thres=500):
    '''
    returns true if contour drawn touches border
    '''
    #img_bw = img_bw.astype(np.uint8) # skip for this use case 
    if isinstance(border_type, str):

        match border_type:
            case 'top':
                return True if np.sum((img_bw[0:5, :])) > thres else False
            case 'bottom':
                return True if np.sum(img_bw[-6:-1, :]) > thres else False
            case 'left':
                return True if np.sum(img_bw[:, 0:5]) > thres else False
            case 'right':
                return True if np.sum(img_bw[:, -6:-1]) > thres else False

    if isinstance(border_type, list):
        cleared_borders = []
        for border in border_type:
            cleared_borders.append(touch_border(img_bw, border))
        return cleared_borders


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def contour_intersect(cnt_ref, cnt_query):
    ## src:
    ## Contour is a list of points
    ## Connect each point to the following point to get a line
    ## If any of the lines intersect, then break

    for ref_idx in range(len(cnt_ref)-1):
    ## Create reference line_ref with point AB
        A = cnt_ref[ref_idx][0]
        B = cnt_ref[ref_idx+1][0]

        for query_idx in range(len(cnt_query)-1):
            ## Create query line_query with point CD
            C = cnt_query[query_idx][0]
            D = cnt_query[query_idx+1][0]

            ## Check if line intersect
            if ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D):
                ## If true, break loop earlier
                return True

    return False


def get_valid_masks(masks):
    valid_masks = []
    for msk in masks:
        # TBC
        mask_region = msk["segmentation"]
        # expand mask region
        mask_region = cv2.dilate(mask_region.astype(np.uint8), kernel=(3, 3), iterations=10)
        #print(np.count_nonzero(mask_region==True, keepdims=False), msk['area'])
        touch_borders = touch_border(mask_region, border_type=['top', 'bottom', 'left', 'right']).count(True)
        corner_cases = [touch_border(mask_region, border_type=['top', 'left']) == [True, True],
                        touch_border(mask_region, border_type=['top', 'right']) == [True, True],
                        touch_border(mask_region, border_type=['bottom', 'left']) == [True, True],
                        touch_border(mask_region, border_type=['bottom', 'right']) == [True, True]
                        ]
        if corner_cases.count(True) > 0:
            continue
        #print(touch_borders)
        if touch_borders > 2:
            print(f"mask touches borders, skipped.")
            continue
        elif touch_borders > 0 and msk['area'] < 1e5:
            print(f"mask with insufficient infomation, skipped.")
            continue
        valid_masks.append(msk)
    
    return valid_masks


def find_cut_off_point(arr, threshold):
    rear = np.count_nonzero(arr==2048, keepdims=False)
    
    low, high = 0, len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2
        
        if arr[mid] >= threshold:
            high = mid - 1  # Move left
        else:
            low = mid + 1  # Move right
    
    return low  if low > 0 else len(arr)-rear # This will be the index of the first element >= threshold


def find_adjacent(current_contour, contours_set, allowed_distance=5):
    # if shared bounds exit (e.g., 50 pixels)
    adjacent_contours_set = {}
    for query_idx, query_cnt in contours_set.items():
        count_shared_points = 0
        # iterate through each point in query contour
        for point in query_cnt:
            diff = current_contour[:, 0] - point
            distance = np.einsum('ij,ij->i', diff, diff)
            point_ind = np.where(distance < allowed_distance)
            if len(point_ind[0]) > 0:
                count_shared_points += 1
        if count_shared_points > 50 or contour_intersect(current_contour, query_cnt):
            adjacent_contours_set[query_idx] = query_cnt
    
    return adjacent_contours_set


def dup_control(new_set, sets_exist):
    """
    if a incoming new set is a subset/duplicate of any existed set, skip
    elif an existed set is a subset of the incoming new set, replace
    else a incoming new set is not seen in existed sets, append
    """ 
    if len(sets_exist) == 0:
        sets_exist.append(new_set)
        return sets_exist
    
    sets_exist = sorted(sets_exist, key=(lambda x: len(x)), reverse=True)
    #print(f"sorted by set size: {sets_exist}")
    for i, exist_set in enumerate(sets_exist):
        if exist_set & new_set:
            if exist_set < new_set:
                sets_exist[i] = new_set
                return sets_exist
            else: # new_set.issubset(exist_set) eq set_a == set_b: returns True
                return sets_exist
        else:
            continue
    sets_exist.append(new_set)
    return sets_exist
        

def merge_masks(masks, im_w, im_h):
    # find contour center
    dist_list = []
    for idx in range(len(masks)):
        cnts = masks[idx]['contour']
        if cnts is None:
            dist_list.append(im_w+im_h) # outside the frame
        else:
            M = cv2.moments(cnts)
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            dist = distance.euclidean((int(im_w/2), int(im_h/2)), (cX, cY))
            dist_list.append(dist)
    
    # 1) sort contours by dist2cntr
    sort_idx = np.argsort(dist_list)#[::-1]
    init_range = find_cut_off_point(np.array(dist_list)[sort_idx], 0.33*min(im_w, im_h))
    #print(dist_list, "cut-off", np.array(dist_list)[sort_idx[: init_range]])
    set_comb = []
    for init_idx in sort_idx[: init_range]:
        # 2) find adjacent contours of the current contour
        curr_cnt = masks[init_idx]['contour']
        merge_area = masks[init_idx]['area']
        merge_set = set()
        merge_set.add(init_idx)
        merge_seg_mask = np.full((im_h, im_w), True, dtype=bool)

        # 2.1 find adjacent contours
        cnts_set = {}
        for res in sort_idx:
            if res != init_idx and dist_list[res] < min(im_w, im_h):
                cnts_set[res] = masks[res]['contour']
        adj_cnts_set = find_adjacent(curr_cnt, cnts_set)
        
        available_ids = adj_cnts_set.keys()
        available_ids = list(set(available_ids) - merge_set)
        if len(available_ids) == 0 and merge_area < 1e-2 * im_w * im_h:
            continue
        
        while merge_area < 0.1 * im_w * im_h and len(available_ids) > 0: # area \in (0.1, 0.4)
            #adj_cnts_set = find_adjacent(curr_cnt, cnts_set)
            #available_ids = adj_cnts_set.keys()
            #available_ids = list(set(available_ids) - merge_set)
            
            # 2.2 mask comb
            random_adj_idx = random.choice(available_ids)
            seg_mask = masks[random_adj_idx]['segmentation']
            merge_seg_mask = merge_seg_mask &~ seg_mask
            merge_area += masks[random_adj_idx]['area']
            count_boundaries = np.count_nonzero(touch_border(merge_seg_mask, border_type=['top', 'bottom', 'left', 'right'])==True, keepdims=False)
            if count_boundaries >= 2 or len(merge_set) > 5 or merge_area > 0.4 * im_w * im_h: # TBC
                break
            else:
                merge_set.add(random_adj_idx)
                available_ids = list(set(available_ids) - merge_set)
        
        # reject duplicate
        set_comb = dup_control(merge_set, set_comb)

    mask_comb = []
    for exist_set in set_comb:
        mask_comb.append(list(map(lambda i: masks[i], exist_set)))
    #print(f"{set_comb}, mask set to return {len(mask_comb)}.")

    return mask_comb


np.random.seed(3)
def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))  


def show_mask_all(image, mask, box_coords=None, input_labels=None, borders=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca(), random_color=True, borders=borders) # TBC
    if box_coords is not None:
        # boxes
        show_box(box_coords, plt.gca())
    plt.axis('off')
    if input_labels is not None:
        plt.savefig(input_labels, bbox_inches='tight')
    else:
        plt.show()


def load_model_hf(repo_id, filename, ckpt_config_filename):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    #checkpoint = torch.load(cache_file, map_location=device)
    checkpoint = torch.load(cache_file)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    #model.to(device)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def instance_outline_extraction_by_mask(grounding_model, sam2_predictor, input_dir, output_dir, img_format='png', image_resolution=1024, expand_pixel=3, device='cuda'):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_filename_list = [i for i in os.listdir(input_dir)]
    images_path = [os.path.join(input_dir, file_path)
                        for file_path in image_filename_list]

    hedDetector = HEDdetector()
    kernel = np.ones((3, 3), np.uint8)
    image_dim = 1024

    instance_types = {'product': ['beauty product', 'cosmetic product', 'skincare product', 'makeup product', 'personal care product', 'gift box'],
                      'human': ['human face', 'face', 'human hand', 'hand', 'palm'],
                      'botanic': ['flower', 'blossom', 'plant', 'branch', 'rattan'],
                      'landmark': ['rock', 'table', 'stair', 'window', 'mirror'],
                      'effect': ['shadow', 'water splash', 'liquid', 'cream', 'bubble', 'lighting']
    }
    
    for img_path, img_name in zip(images_path, image_filename_list):
        #####################################
        img_id = '.'.join(img_name.split('.')[:-1])
        #extract mask
        image_source, image = load_image(img_path, image_dim)
        input_ind = 0
        mask_saved = False
        product_contours = []
        other_contours = []

        for type_name in instance_types.keys():
            for obj_name in instance_types[type_name]:
                det_boxes, _, phrases = predict(
                        model=grounding_model,
                        image=image,
                        caption=obj_name,
                        box_threshold=0.35,
                        text_threshold=0.25
                    )
                #print(det_boxes) # \in (0, 1)
                # process the box prompt for SAM 2
                h, w, _ = image_source.shape
                det_boxes = det_boxes * torch.Tensor([w, h, w, h])
                input_boxes = box_convert(boxes=det_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                sam2_predictor.set_image(image_source)
                mask_all = np.full((image_source.shape[1], image_source.shape[0]), True, dtype=bool)
                

                if det_boxes.size(0) != 0:
                    masks, scores, logits = sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )
                    """
                    Post-process the output of the model to get the masks, scores, and logits for visualization
                    """
                    assert masks.shape[0] == input_boxes.shape[0]
                    
                    # convert the shape to (n, H, W)
                    if masks.ndim == 4:
                        masks = masks.squeeze(1)

                    for mask in masks:
                        im = np.stack((mask,)*3, axis=-1)
                        im = im.astype(np.uint8)*255
                        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        _, thresh = cv2.threshold(imgray, 127, 255, 0)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        if len(contours) >= 50:
                            continue
                        product_contours.append(contours) if type_name == 'product' else other_contours.append(contours)
                        mask_all = mask_all & ~mask.astype(bool)
                    input_ind += 1
                    if False in mask_all:
                        input_labels = output_dir + '/' + img_id + '_' + obj_name + '_' + str(input_ind) + '.' + img_format
                        show_mask_all(image_source, mask_all, box_coords=None, input_labels=input_labels, borders=False)
                        mask_saved = True
                    else:
                        print(f"the outline of {obj_name} in {img_name} cannot be extracted.")
                else:
                    print(f"dino of {obj_name} in {img_name} is None.")
            
        if mask_saved == False:
            print(f"No mask for this instance >~< {img_id}.")
            continue
        
        # Preserving product-related masks | other masks that are suitable for sketch gen
        # conditions: server overlap | slightly overlap | no overlap  
        filter_mask(product_contours, other_contours) # not excutable!


def instance_outline_extraction_by_automask(mask_generator, input_dir, output_dir, img_format='png', image_resolution=1024, expand_pixel=3, device='cuda'):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # create dataset
    test_dataset = GroundedSAM2TestDataset(image_dir=input_dir, transform=custom_transform)
    # create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    hedDetector = HEDdetector()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_pixel, expand_pixel))
    image_dim = 1024
    for batch_idx, (image, img_name) in enumerate(test_loader):
        #####################################
        img_name = os.path.basename(img_name[0])
        #if '005496129018467' not in img_name:
        #if batch_idx < 513:
        #    continue

        img_id = '.'.join(img_name.split('.')[:-1])
        print(f" batch no. {batch_idx}  {img_name}")

        #extract mask
        image_source = np.array(transforms.functional.to_pil_image(image.squeeze(0)))
        #image_source = image.squeeze(0).numpy()
        
        # predict with sam2 automask
        masks = mask_generator.generate(image_source)
        # filter usable mask
        masks = get_valid_masks(masks)
        if len(masks) == 0:
            print(f"{img_id} has no segmentations.")
            continue
        #print(masks[0].keys(), masks[0]['area'], masks[0]['segmentation'])
        if output_dir is not None:
            sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
            plt.figure(figsize=(10, 10))
            plt.imshow(image_source)
            ax = plt.gca()
            ax.set_autoscale_on(False)

            img = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4)) # RGBA
            img[:, :, 3] = 0

            for ann in sorted_masks:
                m = ann['segmentation']
                #print(f"org seg {np.count_nonzero(m==True)}")
                color_mask = np.concatenate([np.random.random(3), [0.5]])
                img[m] = color_mask
                # expand segmentation region

                m = cv2.dilate(m.astype(np.uint8), kernel, iterations=5)
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
                #contours = sorted(contours, key=(lambda x: cv2.contourArea(x)), reverse=False)
                if len(contours) > 1: # typical in occluded objs
                    cnt_area = np.nonzero(np.array([cv2.contourArea(cnt) for cnt in contours]) > 0.1 * ann['area'])
                    #contours = [cv2.convexHull(np.vstack(contours))]
                    ann['contour'] = None if cnt_area[0].shape[0] > 1 else contours[cnt_area[0][0]]
                else:
                    ann['contour'] = contours[0]
                #print(f"dilated seg {np.count_nonzero(m==True)}")
                ann['segmentation_edge'] = m.astype(bool) 
            ax.imshow(img)
            img_save_path = output_dir + '/' + img_id + '.' + img_format
            plt.savefig(img_save_path, bbox_inches='tight')
            plt.close()

            try:
                filtered_mask_comb = merge_masks(sorted_masks, image_dim, image_dim) # for now
                #sorted_masks = sorted(filtered_masks, key=(lambda x: x['area']), reverse=True)
                print(f"mask comb returned {len(filtered_mask_comb)}")
                
                for fid in range(len(filtered_mask_comb)):
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image_source)
                    ax = plt.gca()
                    ax.set_autoscale_on(False)
                    img = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4))
                    img[:, :, 3] = 0
                    mask_all = np.full((image_source.shape[0], image_source.shape[1]), True, dtype=bool)
                    mask_inn = np.full((image_source.shape[0], image_source.shape[1]), True, dtype=bool)
                    if len(filtered_mask_comb[fid]) == len(sorted_masks[0].keys()):
                        _masks = [filtered_mask_comb[fid]]
                    else:
                        _masks = filtered_mask_comb[fid]

                    for ann in _masks:
                        m = ann['segmentation_edge']
                        color_mask = np.concatenate([np.random.random(3), [0.5]])
                        img[m] = color_mask
                        mask_all = mask_all & ~m # blend image & mask
                        mask_inn = mask_inn & ~ann['segmentation']
                        #print(f"count seg pixels {np.count_nonzero(mask_all==False)} & {np.count_nonzero(mask_inn==False)}")
                        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        # Try to smooth contours
                        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                        cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
                    ax.imshow(img)
                    img_save_path = os.path.dirname(output_dir) + '/filtered_' + os.path.basename(output_dir) + '/' + img_id + f'_{fid}' + '.' + img_format
                    plt.savefig(img_save_path, bbox_inches='tight')
                    plt.close()

                    ###### fill holes inside product #######
                    mask_all = ~mask_all
                    mask_all = mask_all.astype(int)
                    mask_all = ndimage.binary_fill_holes(mask_all).astype(int)
                    mask_all = mask_all.astype(bool)
                    mask_all = ~mask_all
                    ###### fill holes inside product #######

                    ##### fill small holes outside product #######
                    ite = 8
                    mask_all = mask_all.astype(int)
                    mask_all = ndimage.binary_closing(mask_all,iterations=ite).astype(int)
                    mask_all = mask_all.astype(bool)
                    ##### fill small holes outside product #######

                    ##### flip surrounding pixels due to previous fill small holes outside product #######
                    mask_all[0:ite+2, :] = True
                    mask_all[:, 0:ite+2] = True
                    mask_all[image_dim-ite-1:, :] = True
                    mask_all[:, image_dim-ite-1:] = True
                    ##### flip surrounding pixels due to previous fill small holes outside product #######
                    cv2.imwrite(os.path.join(output_dir.replace('sam2auto_seg', 'contour_sam2auto_seg'), img_id+f"{fid}_dilated.png"), mask_all.astype(int) * 255)

                    mask_inn = ~mask_inn
                    mask_inn = mask_inn.astype(np.uint8)
                    mask_inn = cv2.dilate(mask_inn, kernel, iterations=3) # contour preserving
                    mask_inn = ndimage.binary_fill_holes(mask_inn).astype(int)
                    mask_inn = np.array(mask_inn, dtype=bool)
                    mask_inn = ~mask_inn
                    cv2.imwrite(os.path.join(output_dir.replace('sam2auto_seg', 'contour_sam2auto_seg'), img_id+f"{fid}_org.png"), mask_inn.astype(int) * 255)

                    mask_contour = ~mask_all & mask_inn
                    #print(f"count seg pixels {np.count_nonzero(mask_all==True)} & {np.count_nonzero(mask_inn==True)} & {np.count_nonzero(mask_contour==True)}")
                    cv2.imwrite(os.path.join(output_dir.replace('sam2auto_seg', 'contour_sam2auto_seg'), img_id+f"{fid}_sub.png"), mask_contour.astype(int) * args.hed_value)

                    # HED extraction
                    hed = HWC3(image_source) # np.array.shape (1024, 1024, 3 or 4)
                    hed = hedDetector(hed) * mask_all
                    #hed = hed * mask_inn
                    #hed = HWC3(hed)
                    hed = np.where(mask_contour==True, args.hed_value, hed)
                    #hed[hed > 60] = args.hed_value
                    #hed[hed <= 60] = 0

                    hed = cv2.resize(hed, (image_resolution, image_resolution), interpolation=cv2.INTER_LINEAR)
                    img_masked = Image.fromarray(hed)   #Image.fromarray(hed)
                    img_save_path = os.path.dirname(output_dir) + '/hed_' + os.path.basename(output_dir) + '/' + img_id + f'_{fid}' + '.' + img_format
                    img_masked.save(img_save_path, img_format)

            except:
                #print(f"{img_id} has no usable segmentation.")
                raise ValueError


if __name__ == "__main__":
    # /s3bucket/beauty-lvm/v1/controlnola-controlnet-trainingdata/image 
    args = parse_args(['--input_dir', '/home/yangmi/Grounded-SAM-2/input_testset', '--output_dir', '/home/yangmi/Grounded-SAM-2/sam2auto_seg'])
    device = torch.device(args.gpu_id)
    
    # use float16 for the entire notebook
    torch.autocast(device_type="cuda:"+str(args.gpu_id), dtype=torch.float16).__enter__()
    torch.autocast(device_type="cuda:0", dtype=torch.float16).__enter__()

    # build dino
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename).to(device)

    # build sam-2
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    mask_generator: SAM2AutomaticMaskGenerator = SAM2AutomaticMaskGenerator(sam2_model, pred_iou_thresh=0.5)
    # copy path to make files writable
    #shutil.copytree(args.input_dir, 'demo_images/'+args.input_dir.split('/')[-1])
    # way toooooooo slow

    # run inference
    #instance_outline_extraction_by_mask(groundingdino_model, sam2_predictor, 
    #                                input_dir=args.input_dir, output_dir=args.output_dir,
    #                                device=device,
    #                               )
    os.makedirs(args.output_dir.replace('sam2auto_seg', 'filtered_sam2auto_seg'), exist_ok=True)
    os.makedirs(args.output_dir.replace('sam2auto_seg', 'hed_sam2auto_seg'), exist_ok=True)
    os.makedirs(args.output_dir.replace('sam2auto_seg', 'contour_sam2auto_seg'), exist_ok=True)
    instance_outline_extraction_by_automask(mask_generator, input_dir=args.input_dir, output_dir=args.output_dir, device=device)