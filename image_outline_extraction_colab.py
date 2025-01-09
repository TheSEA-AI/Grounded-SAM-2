import os, sys
import gc

#sys.path.append('/home/ec2-user/webui-server/ControlNOLA')
sys.path.append(os.path.join(os.getcwd(), "ControlNOLA"))

import argparse
import copy
from pathlib import Path

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
#from transformers import pipeline

import supervision as sv
from scipy import ndimage

import cv2
import numpy as np
import matplotlib.pyplot as plt


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

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of product position extraction based on Grouned SAM.")

    parser.add_argument("--input_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to the image.")

    parser.add_argument("--output_dir", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="Path to the image.")
    
    parser.add_argument("--img_format", 
                        default='png', 
                        type=str, 
                        help="Path to the image.")
    
    parser.add_argument("--product_type", 
                        default=None, 
                        type=str, 
                        required=False,
                        help="The type of the product.")
    
    parser.add_argument("--gpu_id", 
                        default=0, 
                        type=int, 
                        required=False,
                        help="gpu id")

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
def image_outline_extraction_by_mask_multiple_product_types(args, grounding_model, sam2_predictor, intput_dir, output_dir, img_format = 'png', image_resolution = 1024, device='cuda'):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_filename_list = [i for i in os.listdir(intput_dir)]
    images_path = [os.path.join(intput_dir, file_path)
                        for file_path in image_filename_list]

    hedDetector = HEDdetector()
    kernel = np.ones((3, 3), np.uint8)
    image_dim = 1024
    for img_path, img_name in zip(images_path, image_filename_list):
        #mask = product_mask_extraction(img_path, product_type)
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
                raise ValueError(f"the image outline in {img_name} cannot be extracted.")

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
            
        img = img.resize((image_dim, image_dim), Image.LANCZOS)
        image_array = np.asarray(img)

        white_array = np.ones_like(image_array) * args.hed_value
        white_array = white_array * mask_all
        white_array = white_array * mask

        hed = HWC3(image_array)
        hed = hedDetector(hed) * mask_all[:,:,0]
        hed = HWC3(hed)
        hed = np.where(white_array>0, white_array, hed)

        hed = cv2.resize(hed, (image_resolution, image_resolution),interpolation=cv2.INTER_LINEAR)
        img_masked = Image.fromarray(hed)
        img_save_path = output_dir + '/' + img_name
        img_masked.save(img_save_path, img_format)

##### for extracting hed images where the inner lines of produts are removed
if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.gpu_id)

    # use float16 for the entire notebook
    #torch.autocast(device_type="cuda:"+str(args.gpu_id), dtype=torch.float16).__enter__()
    #torch.autocast(device_type="cuda:0", dtype=torch.float16).__enter__()
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try: 
        # build SAM2 image predictor
        sam2_checkpoint = "./checkpoints/sam2_hiera_base_plus.pt"#sam2_hiera_base_plus.pt, sam2_hiera_large.pt
        model_cfg = "sam2_hiera_b+.yaml"#sam2_hiera_b+.yaml, sam2_hiera_l.yaml
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)

        # build grounding dino model
        model_id = "IDEA-Research/grounding-dino-base"
        grounding_model = load_model(
            model_config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
            model_checkpoint_path="gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
            device=device
        )
        # FIXME: figure how does this influence the G-DINO model
        #torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        #args.gpu_id=os.environ["CUDA_VISIBLE_DEVICES"]
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'device={device}')
        image_outline_extraction_by_mask_multiple_product_types(args, grounding_model, sam2_predictor, args.input_dir, args.output_dir, args.img_format, device=device)
        print(f'image outline extraction process finished.')
        #row_position, col_position = row_col_position(args.img_path, args.product_type)
        #print(f'row_position={row_position},col_position={col_position}')
    except:
        traceback.print_exc()
    finally:
        del sam2_model
        del sam2_predictor
        del grounding_model
        gc.collect()
        torch.cuda.empty_cache()
