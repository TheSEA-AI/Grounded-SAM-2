import os, sys
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import torch
import matplotlib.pyplot as plt

#!sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /home/yangmi/.conda/envs/SAM2/lib/python3.11/site-packages/basicsr/data/degradations.py

import gc
import argparse
import copy
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from utils.supervision_utils import CUSTOM_COLOR_MAP
from grounding_dino.groundingdino.models import build_model
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from grounding_dino.groundingdino.util.slconfig import SLConfig
from grounding_dino.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from huggingface_hub import hf_hub_download

## annotation
from annotator.hed import HEDdetector, nms
from annotator.util import HWC3, resize_image

import supervision as sv
from scipy import ndimage

import cv2
from typing import Union
import PIL
from PIL import Image, ImageDraw, ImageFont
import requests
import torch
from io import BytesIO
from torchvision import transforms
from torchvision.ops import box_convert

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
                        default=2.5,#0.916, 
                        type=float, 
                        required=False,
                        help="The threshold to remove hed images")

    parser.add_argument("--hed_value", 
                        default=190, 
                        type=int, 
                        required=False,
                        help="The hed value for product")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args


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


def instance_outline_extraction_by_mask(grounding_model, sam2_predictor, input_dir, output_dir, img_format = 'png', image_resolution = 1024, expand_pixel=3, device='cuda'):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_filename_list = [i for i in os.listdir(input_dir)]
    images_path = [os.path.join(input_dir, file_path)
                        for file_path in image_filename_list]

    hedDetector = HEDdetector()
    kernel = np.ones((3, 3), np.uint8)
    image_dim = 1024

    instance_types = {'product': ['beauty product', 'cosmetic product', 'skincare product', 'makeup product', 'personal care product', 'gift boxes'],
                      'human': ['human faces', 'faces', 'human hands', 'hands'],
                      'botanic': ['flowers', 'blossom', 'plants', 'bush'],
                      'landmark': ['rocks', 'tables', 'stairs', 'windows', 'mirror'],
                      'effect': ['shadows', 'water splash', 'bubbles', 'lighting']
    }

    for img_path, img_name in zip(images_path, image_filename_list):
        #####################################
        img_id = '.'.join(img_name.split('.')[:-1])
        #extract mask
        image_source, image = load_image(img_path, image_dim)
        input_ind = 0
        mask_saved = False
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
                mask_all = np.full((image_source.shape[1],image_source.shape[1]), True, dtype=bool)
                #print(obj_name, det_boxes)
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
                        mask_all = mask_all & ~mask.astype(bool)
                    input_ind += 1
                    #print(output_dir)
                    input_labels = output_dir + '/tmp_seg/' + img_id + '_' + obj_name + '_' + str(input_ind) + '.' + img_format
                    show_mask_all(image_source, mask_all, box_coords=None, input_labels=input_labels, borders=False)
                    mask_saved = True
                else:
                    print(f"the outline of {obj_name} in {img_name} cannot be extracted.")
                    #raise ValueError(f"the outline of {obj_name} in {img_name} cannot be extracted.")
            
            if mask_saved == False:
                print(f"No mask for this instance >~< {img_id}.")
                continue


if __name__ == "__main__":
    args = parse_args(['--input_dir', '/mys3bucket/beauty-lvm/v1/controlnola-controlnet-trainingdata/batch1', '--output_dir', '/home/yangmi/Grounded-SAM-2'])
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

    # copy path to make files writable
    #shutil.copytree(args.input_dir, 'demo_images/'+args.input_dir.split('/')[-1])
    # way toooooooo slow

    # run inference
    instance_outline_extraction_by_mask(groundingdino_model, sam2_predictor, 
                                    input_dir=args.input_dir, output_dir=args.output_dir,
                                    device=device,
                                   )