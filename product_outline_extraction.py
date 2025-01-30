##### for extracting hed images where the inner lines of produts are removed

import os, sys
import gc

sys.path.append('/home/ec2-user/webui-server/ControlNOLA')
#sys.path.append(os.path.join(os.getcwd(), "ControlNOLA"))

import argparse
import copy
from pathlib import Path

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO and SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
#from transformers import pipeline

import supervision as sv
from scipy import ndimage

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

## annotation
from annotator.hed import HEDdetector, nms
from annotator.util import HWC3, resize_image

# diffusers
import PIL
import requests
import torch
from io import BytesIO
from torchvision import transforms

import locale
locale.getpreferredencoding = lambda: "UTF-8"
import traceback

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
                        required=True, 
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
                        default=3.0,#0.916, 
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

##the latest version with multiple product types and filling holes etc.
##the holes are becasue of SAM noise
def product_outline_extraction_by_mask_multiple_product_types(args, grounding_model, sam2_predictor, input_dir, output_dir, img_format = 'png', image_resolution = 1024, device='cuda'):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_filename_list = [i for i in os.listdir(input_dir)]
    images_path = [os.path.join(input_dir, file_path)
                        for file_path in image_filename_list]

    hedDetector = HEDdetector()
    kernel = np.ones((3, 3), np.uint8)
    image_dim = 1024
    for img_path, img_name in zip(images_path, image_filename_list):
        #####################################
        #extract mask

        image_source, image = load_image(img_path, image_dim)
        sam2_predictor.set_image(image_source)
        product_types = ["beauty product", "cosmetic product", "skincare product", "makeup product", "personal care product"]
        mask_all = np.full((image_source.shape[1],image_source.shape[1]), True, dtype=bool)
        for product_type in product_types:
            boxes, _, _ = predict(
                model=grounding_model,
                image=image,
                caption=product_type,
                box_threshold=0.35,
                text_threshold=0.25
            )

            # process the box prompt for SAM 2
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            if boxes.size(0) != 0:
                masks, _, _ = sam2_predictor.predict(
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
            else:
                raise ValueError(f"the product outline in {img_name} cannot be extracted.")


        ##### fill holes inside product #######
        mask_all = ~mask_all
        mask_all = mask_all.astype(int)
        mask_all = ndimage.binary_fill_holes(mask_all).astype(int)
        mask_all = mask_all.astype(bool)
        mask_all = ~mask_all
        ##### fill holes inside product #######

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

        mask_all = np.stack((mask_all,)*3, axis=-1)
        ################

        mask = ~mask_all
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = np.array(mask, dtype=bool)

        image_raw = Image.open(img_path)#.convert("RGB")
        if image_raw.mode in ('RGBA', 'LA') or (image_raw.mode == 'P' and 'transparency' in image_raw.info):
            # Create a white background image of the same size
            img = Image.new('RGBA', image_raw.size, (255, 255, 255, 255))  # White background
            # Paste the image on the white background using the alpha channel as a mask
            image_raw = image_raw.convert('RGBA')
            img.paste(image_raw, mask=image_raw.split()[3])
            # Convert the image to RGB mode (to remove the alpha channel)
            img = img.convert('RGB')
        else:
            # If the image doesn't have transparency, no change is needed
            img = image_raw.convert('RGB')

        #img = img.resize((image_dim, image_dim), Image.LANCZOS)
        image_array = np.asarray(img)

        #white_array = np.ones_like(image_array) * args.hed_value
        white_array = np.ones((image_dim, image_dim, 3)) * args.hed_value
        white_array = white_array * mask_all
        white_array = white_array * mask

        hed = HWC3(image_array)
        hed = hedDetector(hed) 
        hed = cv2.resize(hed, (image_resolution, image_resolution),interpolation=cv2.INTER_LINEAR)
        hed = hed * mask_all[:,:,0]
        hed = hed*mask[:,:,0]
        hed = HWC3(hed)
        hed = np.where(white_array>0, white_array, hed)
        hed[hed > 60] = args.hed_value
        hed[hed <= 60] = 0

        hed = cv2.resize(hed, (image_resolution, image_resolution),interpolation=cv2.INTER_LINEAR)
        img_masked = Image.fromarray(hed)
        img_save_path = output_dir + '/' + img_name
        img_masked.save(img_save_path, img_format)

## function for data hed background filtering
def filter_hed(args, data_hed_background_dir, data_similarity_dict, similarity_threshold, product_images, img_format = 'png'):

    large_value = 100
    kernel = np.ones((3, 3), np.uint8)

    image_filename_list = [i for i in os.listdir(data_hed_background_dir)]
    images_path = [os.path.join(data_hed_background_dir, file_path)
                        for file_path in image_filename_list]

    ## make a copy of origial hed images
    image_dirs = data_hed_background_dir.split('/')
    new_image_dir = '/'+image_dirs[0]
    for i in range(1, len(image_dirs)-1):
        new_image_dir += image_dirs[i] + '/'
    new_image_dir += 'data_hed_background_original'
    Path(new_image_dir).mkdir(parents=True, exist_ok=True)
    #print(f'data_hed_background_dir={data_hed_background_dir}')
    #print(f'new_data_hed_background_dir={new_image_dir}')

    for img_name, img_path in zip(image_filename_list, images_path):
        img = Image.open(img_path).convert("RGB")
        img.save(new_image_dir+'/'+img_name, 'png')

    ## calculate similarities
    img_similarity_dict_all = {}
    for product_image in product_images:
        img1 = cv2.imread(os.path.join(data_hed_background_dir, product_image), cv2.IMREAD_GRAYSCALE)
        img1[img1 > 60] = args.hed_value
        img1[img1 <= 60] = 0
        ret1, thresh1 = cv2.threshold(img1, 127, 255,0)
        contours1,hierarchy1 = cv2.findContours(thresh1,2,1)
        cnt1 = contours1[0]
        area_cnt1 = cv2.contourArea(cnt1)

        img_similarity_dic = {}
        for img_name, img_path in zip(image_filename_list, images_path):
            if img_name not in product_images:
                img2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img2[img2 > 60] = args.hed_value
                img2[img2 <= 60] = 0
                ret2, thresh2 = cv2.threshold(img2, 127, 255,0)
                contours2,hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours2) <=2 and len(contours2) > 0:
                    cnt2 = contours2[0]
                    ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
                    if img_name not in img_similarity_dic:
                        img_similarity_dic[img_name] = ret
                    else:
                        if img_similarity_dic[img_name] > ret:
                            img_similarity_dic[img_name] = ret
                elif len(contours2) > 2:
                    for cnt2 in contours2:
                        area_cnt2 = cv2.contourArea(cnt2)
                        if area_cnt2 >= 0.6*area_cnt1:
                          ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
                          
                          if img_name not in img_similarity_dic:
                              img_similarity_dic[img_name] = ret
                          else:
                              if img_similarity_dic[img_name] > ret:
                                  img_similarity_dic[img_name] = ret
                else:
                    img_similarity_dic[img_name] = large_value

        img_similarity_dict_all[product_image] = img_similarity_dic


    ##measure similarity
    candidates = {}
    for img_name, img_path in zip(image_filename_list, images_path):
        if img_name not in product_images:
            remove = []

            global_similarity = 1000
            for k, v in img_similarity_dict_all.items():
                data_simi_list = list(data_similarity_dict[k].values())
                for k_product, v_product in v.items():
                    if img_name == k_product:
                        if min(data_simi_list) <= 0.1:
                            v_product = v_product*10.0

                        if global_similarity > v_product:
                            global_similarity = v_product

                        if v_product >= similarity_threshold:
                            remove.append(True)
                        else:
                            remove.append(False)

            if False in remove:
                candidates[img_name] = global_similarity
                #print(f'img_name={img_name}, minimum similarity={global_similarity}')
                #os.remove(img_path)

    ##do filtering
    for img_name, img_path in zip(image_filename_list, images_path):
        if img_name not in product_images:
            remove = []

            target_similarity = 1000
            for k, v in img_similarity_dict_all.items():
                data_simi_list = list(data_similarity_dict[k].values())
                for k_product, v_product in v.items():
                    if img_name == k_product:
                        if target_similarity > v_product:
                            target_similarity = v_product
                        if min(data_simi_list) <= 0.1:
                            v_product = v_product*10.0
                        if v_product >= similarity_threshold:
                            remove.append(True)
                        else:
                            remove.append(False)

            if False not in remove:
                os.remove(img_path)
            else:##mask unwanted objects in images with more than two contours、
                #masks = []
                img2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img2[img2 > 60] = args.hed_value
                img2[img2 <= 60] = 0
                ret2, thresh2 = cv2.threshold(img2, 127, 255,0)
                contours2,hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours2) > 2:
                    img_shape = (1024, 1024)
                    #tmp_image = np.asarray(Image.open(img_path).convert("RGB"))
                    image_raw = Image.open(img_path)#.convert("RGB")
                    if image_raw.mode in ('RGBA', 'LA') or (image_raw.mode == 'P' and 'transparency' in image_raw.info):
                        # Create a white background image of the same size
                        tmp_image = Image.new('RGBA', image_raw.size, (255, 255, 255, 255))  # White background
                        # Paste the image on the white background using the alpha channel as a mask
                        image_raw = image_raw.convert('RGBA')
                        tmp_image.paste(image_raw, mask=image_raw.split()[3])
                        # Convert the image to RGB mode (to remove the alpha channel)
                        tmp_image = tmp_image.convert('RGB')
                    else:
                        # If the image doesn't have transparency, no change is needed
                        tmp_image = image_raw.convert('RGB')

                    tmp_image = np.asarray(tmp_image)

                    for cnt2 in contours2:
                        tmp_similarity = 10000
                        for product_image in product_images:
                            img1 = cv2.imread(os.path.join(data_hed_background_dir, product_image), cv2.IMREAD_GRAYSCALE)
                            img1[img1 > 60] = args.hed_value
                            img1[img1 <= 60] = 0
                            ret1, thresh1 = cv2.threshold(img1, 127, 255,0)
                            contours1,hierarchy1 = cv2.findContours(thresh1,2,1)
                            cnt1 = contours1[0]

                            ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
                            if tmp_similarity > ret:
                                tmp_similarity = ret

                        if tmp_similarity == target_similarity:
                          mask  = make_mask_contour(img_shape, cnt2.reshape(-1,2)).astype(bool)
                          mask = np.stack((~mask,)*3, axis=-1)

                          tmp_mask = ~mask
                          tmp_mask = tmp_mask.astype(np.uint8)
                          tmp_mask = cv2.dilate(tmp_mask, kernel, iterations=3)
                          tmp_mask = np.array(tmp_mask, dtype=bool)

                          tmp_white_array = np.ones_like(tmp_image) * args.hed_value
                          tmp_white_array = tmp_white_array * mask
                          tmp_white_array = tmp_white_array * tmp_mask
                          tmp_image = Image.fromarray(tmp_white_array)
                          tmp_image.save(img_path, img_format)

    ## remove more than 2 images
    if len(candidates.keys()) > 2:
        similarity_list = list(candidates.values())
        similarity_list.sort()

        for k, v in candidates.items():
            #print(f'img name={k}, similarity={v}')
            for img_name, img_path in zip(image_filename_list, images_path):
                if img_name not in product_images:
                    if k == img_name:
                        if v > similarity_list[1]:
                            os.remove(img_path)

    return new_image_dir

## the filtering for data is not enabled
## this is mainly for calculating data similarities to be used in hed filtering
def filter_data(args, hed_background_dir, hed_dir, product_images):

    #print(f'product_images={product_images}')
    image_filename_list = [i for i in os.listdir(hed_dir)]
    images_path = [os.path.join(hed_dir, file_path)
                        for file_path in image_filename_list]

    ## make a copy of origial hed images
    image_dirs = hed_dir.split('/')
    new_image_dir = '/'+image_dirs[0]
    for i in range(1, len(image_dirs)-1):
        new_image_dir += image_dirs[i] + '/'
    new_image_dir += 'data_hed_original'
    Path(new_image_dir).mkdir(parents=True, exist_ok=True)

    for img_name, img_path in zip(image_filename_list, images_path):
        img = Image.open(img_path).convert("RGB")
        img.save(new_image_dir+'/'+img_name, 'png')

    ### calculate similarities
    img_similarity_dict_all = {}
    for product_image in product_images:
        img1 = cv2.imread(os.path.join(hed_background_dir, product_image), cv2.IMREAD_GRAYSCALE)
        img1[img1 > 60] = args.hed_value
        img1[img1 <= 60] = 0
        ret1, thresh1 = cv2.threshold(img1, 127, 255,0)
        contours1,hierarchy1 = cv2.findContours(thresh1,2,1)
        cnt1 = contours1[0]

        img_similarity_dic = {}
        for img_name, img_path in zip(image_filename_list, images_path):
            if img_name not in product_images:
                img2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img2[img2 > 60] = args.hed_value
                img2[img2 <= 60] = 0
                ret2, thresh2 = cv2.threshold(img2, 127, 255,0)
                contours2, hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #print(f'img_name={img_name}, contours2 len={len(contours2)}')
                for cnt in contours2:
                    ret = cv2.matchShapes(cnt1,cnt,1,0.0)
                    if img_name not in img_similarity_dic:
                        img_similarity_dic[img_name] = ret
                    else:
                        if img_similarity_dic[img_name] > ret:
                            img_similarity_dic[img_name] = ret
        
        img_similarity_dict_all[product_image] = img_similarity_dic

    return img_similarity_dict_all

def make_mask_contour(img_shape: tuple, contour: Union[list, np.ndarray]) -> np.ndarray:
    contour = np.array(contour, dtype=np.int32)
    shapeC = np.shape(contour)

    if len(shapeC) != 2:
        raise ValueError("the shape is not valid")

    if shapeC[1] != 2:
        raise ValueError("the shape is not valid")

    contour = contour.reshape((1, shapeC[0], 2))

    mask = np.zeros((img_shape[1], img_shape[0]), dtype=np.uint8)
    cv2.fillPoly(mask, contour, 1)
    return mask

## function for saving product hed as transparent png
def product_hed_transparent_bg(args, product_images, data_hed_background_dir):

    ## create data_hed_transparent_dir
    image_dirs = data_hed_background_dir.split('/')
    data_hed_transparent_dir = '/'+image_dirs[0]
    for i in range(1, len(image_dirs)-1):
        data_hed_transparent_dir += image_dirs[i] + '/'
    data_hed_transparent_dir += 'data_hed_transparent'
    Path(data_hed_transparent_dir).mkdir(parents=True, exist_ok=True)

    image_filename_list = [i for i in os.listdir(data_hed_background_dir) if i.endswith('.png')]
    images_path = [os.path.join(data_hed_background_dir, file_path)
                        for file_path in image_filename_list]

    img1 = cv2.imread(os.path.join(data_hed_background_dir, product_images[0]), cv2.IMREAD_GRAYSCALE)
    img1[img1 > 60] = args.hed_value
    img1[img1 <= 60] = 0
    ret1, thresh1 = cv2.threshold(img1, 127, 255,0)
    contours1,hierarchy1 = cv2.findContours(thresh1,2,1)
    cnt1 = contours1[0]
    area_cnt1 = cv2.contourArea(cnt1)

    img_shape = (1024, 1024)
    for img_name, img_path in zip(image_filename_list, images_path):
        #print(f'img_name={img_name}')
        img2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img2[img2 > 60] = args.hed_value
        img2[img2 <= 60] = 0
        ret2, thresh2 = cv2.threshold(img2, 127, 255,0)
        contours2, hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours2) !=0:
          if len(contours2) == 2:
            area_cnt2 = cv2.contourArea(contours2[0])
            if area_cnt2 <= 2.0*area_cnt1:
              mask  = make_mask_contour(img_shape, contours2[0].reshape(-1,2)).astype(np.uint8)*255
              mask = np.stack((mask,)*3, axis=-1)
              #tmp_image = Image.open(img_path).convert("RGB")
              image_raw = Image.open(img_path)#.convert("RGB")
              if image_raw.mode in ('RGBA', 'LA') or (image_raw.mode == 'P' and 'transparency' in image_raw.info):
                # Create a white background image of the same size
                tmp_image = Image.new('RGBA', image_raw.size, (255, 255, 255, 255))  # White background
                # Paste the image on the white background using the alpha channel as a mask
                image_raw = image_raw.convert('RGBA')
                tmp_image.paste(image_raw, mask=image_raw.split()[3])
                # Convert the image to RGB mode (to remove the alpha channel)
                tmp_image = tmp_image.convert('RGB')
              else:
                # If the image doesn't have transparency, no change is needed
                tmp_image = image_raw.convert('RGB')
                
              mask_img = Image.fromarray(mask).convert('L')
              tmp_image.putalpha(mask_img)
              tmp_image.save(data_hed_transparent_dir+'/'+img_name, 'png')
          else:
            index = 0
            rec_center = []
            for cnt2 in contours2:
              area_cnt2 = cv2.contourArea(cnt2)
              rec = cv2.minAreaRect(cnt2)

              if area_cnt2 >= 0.5*area_cnt1 and area_cnt2 <= 2.0*area_cnt1:
                if len(rec_center) == 0:
                  rec_center.append(rec[0])

                  mask  = make_mask_contour(img_shape, cnt2.reshape(-1,2)).astype(np.uint8)*255
                  mask = np.stack((mask,)*3, axis=-1)
                  #tmp_image = Image.open(img_path).convert("RGB")
                  image_raw = Image.open(img_path)#.convert("RGB")
                  if image_raw.mode in ('RGBA', 'LA') or (image_raw.mode == 'P' and 'transparency' in image_raw.info):
                    # Create a white background image of the same size
                    tmp_image = Image.new('RGBA', image_raw.size, (255, 255, 255, 255))  # White background
                    # Paste the image on the white background using the alpha channel as a mask
                    image_raw = image_raw.convert('RGBA')
                    tmp_image.paste(image_raw, mask=image_raw.split()[3])
                    # Convert the image to RGB mode (to remove the alpha channel)
                    tmp_image = tmp_image.convert('RGB')
                  else:
                    # If the image doesn't have transparency, no change is needed
                    tmp_image = image_raw.convert('RGB')

                  mask_img = Image.fromarray(mask).convert('L')
                  tmp_image.putalpha(mask_img)
                  #tmp_image.save(data_hed_transparent_dir+'/'+str(index)+img_name, 'png')
                  if index == 0:
                      tmp_image.save(data_hed_transparent_dir+'/'+img_name, 'png')
                  else:
                      tmp_image.save(data_hed_transparent_dir+'/'+str(index)+img_name, 'png')
                  index += 1
                elif rec[0] not in rec_center:
                  dist = 1000
                  for rec_c in rec_center:
                    tmp_dist = np.linalg.norm(np.array(rec[0])-np.array(rec_c))
                    if tmp_dist < dist:
                      dist = tmp_dist
                  if dist > 1.0:
                    rec_center.append(rec[0])
                    mask  = make_mask_contour(img_shape, cnt2.reshape(-1,2)).astype(np.uint8)*255
                    mask = np.stack((mask,)*3, axis=-1)
                    #tmp_image = Image.open(img_path).convert("RGB")
                    image_raw = Image.open(img_path)#.convert("RGB")
                    if image_raw.mode in ('RGBA', 'LA') or (image_raw.mode == 'P' and 'transparency' in image_raw.info):
                        # Create a white background image of the same size
                        tmp_image = Image.new('RGBA', image_raw.size, (255, 255, 255, 255))  # White background
                        # Paste the image on the white background using the alpha channel as a mask
                        image_raw = image_raw.convert('RGBA')
                        tmp_image.paste(image_raw, mask=image_raw.split()[3])
                        # Convert the image to RGB mode (to remove the alpha channel)
                        tmp_image = tmp_image.convert('RGB')
                    else:
                        # If the image doesn't have transparency, no change is needed
                        tmp_image = image_raw.convert('RGB')

                    mask_img = Image.fromarray(mask).convert('L')
                    tmp_image.putalpha(mask_img)
                    #tmp_image.save(data_hed_transparent_dir+'/'+str(index)+img_name, 'png')
                    if index == 0:
                        tmp_image.save(data_hed_transparent_dir+'/'+img_name, 'png')
                    else:
                        tmp_image.save(data_hed_transparent_dir+'/'+str(index)+img_name, 'png')
                    index += 1

    return data_hed_transparent_dir

## check whether hed is over-extracted
def examine_image_hed(args, grounding_model, sam2_predictor, product_images, data_dir, data_hed_dir, data_similarity_dict, similarity_threshold = 0.916, device='cuda'):
  large_value = 100

  image_filename_list = [i for i in os.listdir(data_hed_dir)]
  images_path = [os.path.join(data_hed_dir, file_path)
                      for file_path in image_filename_list]

  ## calculate similarities
  img_similarity_dict_all = {}
  for product_image in product_images:
      img1 = cv2.imread(os.path.join(data_hed_dir, product_image), cv2.IMREAD_GRAYSCALE)
      img1[img1 > 60] = args.hed_value
      img1[img1 <= 60] = 0
      ret1, thresh1 = cv2.threshold(img1, 127, 255,0)
      contours1,hierarchy1 = cv2.findContours(thresh1,2,1)
      cnt1 = contours1[0]
      area_cnt1 = cv2.contourArea(cnt1)

      img_similarity_dic = {}
      for img_name, img_path in zip(image_filename_list, images_path):
          if img_name not in product_images:
              img2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
              img2[img2 > 60] = args.hed_value
              img2[img2 <= 60] = 0
              ret2, thresh2 = cv2.threshold(img2, 127, 255,0)
              contours2,hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
              if len(contours2) > 0:
                  for cnt2 in contours2:
                    area_cnt2 = cv2.contourArea(cnt2)
                    if area_cnt2 >= 0.6*area_cnt1:
                      ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
                      if img_name not in img_similarity_dic:
                          img_similarity_dic[img_name] = ret
                      else:
                          if img_similarity_dic[img_name] > ret:
                              img_similarity_dic[img_name] = ret
                    else:
                      img_similarity_dic[img_name] = large_value
              else:
                  img_similarity_dic[img_name] = large_value

      img_similarity_dict_all[product_image] = img_similarity_dic


  ##do re-extraction
  for img_name, img_path in zip(image_filename_list, images_path):
      if img_name not in product_images:
          remove = []

          for k, v in img_similarity_dict_all.items():
              data_simi_list = list(data_similarity_dict[k].values())
              for k_product, v_product in v.items():
                  if img_name == k_product:
                      if min(data_simi_list) <= 0.1:
                          v_product = v_product*10.0
                      #print(f'img_name={img_name}, similarity={v_product}, similarity_threshold={similarity_threshold}')
                      if v_product >= similarity_threshold:
                          remove.append(True)
                      else:
                          remove.append(False)
          
          #print(f'remove={remove}')
          if False not in remove:
            #os.remove(img_path)
            image_outline_re_extraction_by_mask_multiple_product_types(grounding_model, sam2_predictor, data_dir, img_path, img_name, device=device)


##re-extract an image hed when hed is over-extracted.
def image_outline_re_extraction_by_mask_multiple_product_types(grounding_model, sam2_predictor, data_dir, output_path, img_name, img_format = 'png', image_resolution = 1024, device='cuda'):

    img_path = data_dir + '/' + img_name

    hedDetector = HEDdetector()
    kernel = np.ones((3, 3), np.uint8)
    image_dim = 1024
    
    #####################################
    #extract mask
    image_source, image = load_image(img_path, image_dim)
    sam2_predictor.set_image(image_source)
    product_types = ["beauty product", "cosmetic product", "skincare product", "makeup product", "personal care product"]
    mask_all = np.full((image_source.shape[1],image_source.shape[1]), True, dtype=bool)
    individual_masks = []
    for product_type in product_types:
        boxes, _, _ = predict(
                model=grounding_model,
                image=image,
                caption=product_type,
                box_threshold=0.35,
                text_threshold=0.25
            )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        if boxes.size(0) != 0:
            masks, _, _ = sam2_predictor.predict(
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
                individual_masks.append(~mask.astype(bool))
        else:
            raise ValueError("the product cannot be extracted.")

    ##### fill holes inside product #######
    mask_all = ~mask_all
    mask_all = mask_all.astype(int)
    mask_all = ndimage.binary_fill_holes(mask_all).astype(int)
    mask_all = mask_all.astype(bool)
    mask_all = ~mask_all
    ##### fill holes inside product #######

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

    mask_all = np.stack((mask_all,)*3, axis=-1)
    ################
    mask = ~mask_all
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = np.array(mask, dtype=bool)

    image_raw = Image.open(img_path)#.convert("RGB")
    if image_raw.mode in ('RGBA', 'LA') or (image_raw.mode == 'P' and 'transparency' in image_raw.info):
        # Create a white background image of the same size
        img = Image.new('RGBA', image_raw.size, (255, 255, 255, 255))  # White background
        # Paste the image on the white background using the alpha channel as a mask
        image_raw = image_raw.convert('RGBA')
        img.paste(image_raw, mask=image_raw.split()[3])
        # Convert the image to RGB mode (to remove the alpha channel)
        img = img.convert('RGB')
    else:
        # If the image doesn't have transparency, no change is needed
        img = image_raw.convert('RGB')
            
    #img = img.resize((image_dim, image_dim), Image.LANCZOS)
    image_array = np.asarray(img)

    #white_array = np.ones_like(image_array) * args.hed_value
    white_array = np.ones((image_dim, image_dim, 3)) * args.hed_value
    white_array = white_array * mask_all
    white_array = white_array * mask

    hed = HWC3(image_array)
    hed = hedDetector(hed) 
    hed = cv2.resize(hed, (image_resolution, image_resolution),interpolation=cv2.INTER_LINEAR)
    hed = hed * mask_all[:,:,0]
    hed = HWC3(hed)
    hed = np.where(white_array>0, white_array, hed)

    for individual_mask in individual_masks:
      ##### fill holes inside product #######
      individual_mask = ~individual_mask
      individual_mask = individual_mask.astype(int)
      individual_mask = ndimage.binary_fill_holes(individual_mask).astype(int)
      individual_mask = individual_mask.astype(bool)
      individual_mask = ~individual_mask
      ##### fill holes inside product #######

      ##### fill small holes outside product #######
      ite = 8
      individual_mask = individual_mask.astype(int)
      individual_mask = ndimage.binary_closing(individual_mask,iterations=ite).astype(int)
      individual_mask = individual_mask.astype(bool)
      ##### fill small holes outside product #######

      ##### flip surrounding pixels due to previous fill small holes outside product #######
      individual_mask[0:ite+2, :] = True
      individual_mask[:, 0:ite+2] = True
      individual_mask[image_dim-ite-1:, :] = True
      individual_mask[:, image_dim-ite-1:] = True
      ##### flip surrounding pixels due to previous fill small holes outside product #######
      
      individual_mask = np.stack((individual_mask,)*3, axis=-1)
      ################
      tmp_mask = ~individual_mask
      tmp_mask = tmp_mask.astype(np.uint8)
      tmp_mask = cv2.dilate(tmp_mask, kernel, iterations=3)
      tmp_mask = np.array(tmp_mask, dtype=bool)

      tmp_white_array = np.ones_like(image_array) * args.hed_value
      tmp_white_array = tmp_white_array * individual_mask
      tmp_white_array = tmp_white_array * tmp_mask
      hed = np.where(tmp_white_array>0, tmp_white_array, hed)

    hed = cv2.resize(hed, (image_resolution, image_resolution),interpolation=cv2.INTER_LINEAR)
    img_masked = Image.fromarray(hed)
    img_masked.save(output_path, img_format)


# extract product with transparent background
def product_transparent_bg(args, data_hed_transparent_dir):
    
    ## create data_hed_transparent_dir
    image_dirs = data_hed_transparent_dir.split('/')
    data_product_transparent_dir = '/'+image_dirs[0]
    for i in range(1, len(image_dirs)-1):
        data_product_transparent_dir += image_dirs[i] + '/'
    data_product_transparent_dir += 'data_product_transparent'
    Path(data_product_transparent_dir).mkdir(parents=True, exist_ok=True)

    image_filename_list = [i for i in os.listdir(args.input_dir)]
    images_path = [os.path.join(args.input_dir, file_path)
                        for file_path in image_filename_list]
    
    image_hed_filename_list = [i for i in os.listdir(data_hed_transparent_dir)]
    images_hed_path = [os.path.join(data_hed_transparent_dir, file_path)
                        for file_path in image_hed_filename_list]
    
    for img_name, img_path in zip(image_filename_list, images_path):
        product_image = Image.open(img_path)
        product_image = product_image.convert("RGBA")
        for img_hed_name, img_hed_path in zip(image_hed_filename_list, images_hed_path):
            if img_name in img_hed_name:
                hed_image = Image.open(img_hed_path)
                product_image = product_image.resize(hed_image.size, Image.LANCZOS)
                if hed_image.mode == 'RGBA':
                    _, _, _, alpha = hed_image.split()
                    product_image.putalpha(alpha)
                    product_image.save(data_product_transparent_dir+'/'+img_hed_name, 'png')


##### for extracting hed images where the inner lines of produts are removed
if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.gpu_id)
    
    # use float16 for the entire notebook
    torch.autocast(device_type="cuda:"+str(args.gpu_id), dtype=torch.float16).__enter__()
    torch.autocast(device_type="cuda:0", dtype=torch.float16).__enter__()
    #torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
    

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    try:
        # build SAM2 image predictor
        sam2_checkpoint = "/home/ec2-user/webui-server/Grounded_Segment_Anything_2/checkpoints/sam2_hiera_base_plus.pt"#sam2_hiera_base_plus.pt, sam2_hiera_large.pt
        model_cfg = "sam2_hiera_b+.yaml"#sam2_hiera_b+.yaml, sam2_hiera_l.yaml
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)

        # build grounding dino model
        model_id = "IDEA-Research/grounding-dino-base"
        grounding_model = load_model(
            model_config_path="/home/ec2-user/webui-server/Grounded_Segment_Anything_2/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
            model_checkpoint_path="/home/ec2-user/webui-server/Grounded_Segment_Anything_2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
            device=device
        )
        # FIXME: figure how does this influence the G-DINO model
        #torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        product_outline_extraction_by_mask_multiple_product_types(args, grounding_model, sam2_predictor, args.input_dir, args.output_dir, args.img_format, device=device)
        #print(f'similarity={args.similarity_threshold}')
        #print(f'args.product_images={args.product_images}, len(args.product_images)={len(args.product_images)}')
        if args.product_images is not None:
            data_similarity_dict_all = filter_data(args, args.output_dir, args.data_hed_dir, args.product_images)
            data_hed_bg_original = filter_hed(args, args.output_dir, data_similarity_dict_all, args.similarity_threshold, args.product_images)
            examine_image_hed(args, grounding_model, sam2_predictor, args.product_images, args.input_dir, args.data_hed_dir, data_similarity_dict_all, args.similarity_threshold, device=device)
            data_hed_transparent_dir = product_hed_transparent_bg(args, args.product_images, data_hed_bg_original)
            product_transparent_bg(args, data_hed_transparent_dir)
        print(f'product outline extraction process finished.')
    except:
        traceback.print_exc()
    finally:
        del sam2_model
        del sam2_predictor
        del grounding_model
        gc.collect()
        torch.cuda.empty_cache()
