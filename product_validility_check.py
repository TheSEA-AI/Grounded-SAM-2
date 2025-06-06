import os, sys
import gc

sys.path.append('/home/ec2-user/webui-server/ControlNOLA')

from pathlib import Path

# Grounding DINO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

import torch
import locale
locale.getpreferredencoding = lambda: "UTF-8"

def image_outline_extraction_by_mask_multiple_product_types(grounding_model, sam2_predictor, input_dir, output_dir, image_resolution = 1024, device='cuda'):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_filename_list = [i for i in os.listdir(input_dir)]
    images_path = [os.path.join(input_dir, file_path)
                        for file_path in image_filename_list]

    failure_cases = []
    for img_path, img_name in zip(images_path, image_filename_list):
        image_source, image = load_image(img_path, image_resolution)
        sam2_predictor.set_image(image_source)
        product_types = ["beauty product", "cosmetic product", "skincare product", "makeup product", "personal care product"]
        for product_type in product_types:
            boxes, _, _ = predict(
                model=grounding_model,
                image=image,
                caption=product_type,
                box_threshold=0.35,
                text_threshold=0.25,
                device = device
            )
            # process the box prompt for SAM 2
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            
            if boxes.size(0) == 0:
                failure_cases.append(img_name)

        return failure_cases

def do_checking(gpu_id, input_dir, output_dir):

    device = torch.device('cuda')
    torch.cuda.set_device(gpu_id)

    if device != 'cpu':
        # use float16 for the entire notebook
        torch.autocast(device_type="cuda:"+str(gpu_id), dtype=torch.float16).__enter__()
        torch.autocast(device_type="cuda:0", dtype=torch.float16).__enter__()
        #torch.autocast(device_type="cpu", dtype=torch.float16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # build SAM2 image predictor
    sam2_checkpoint = "/home/ec2-user/webui-server/Grounded_Segment_Anything_2/checkpoints/sam2_hiera_base_plus.pt"#sam2_hiera_base_plus.pt, sam2_hiera_large.pt
    model_cfg = "sam2_hiera_b+.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path="/home/ec2-user/webui-server/Grounded_Segment_Anything_2/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
        model_checkpoint_path="/home/ec2-user/webui-server/Grounded_Segment_Anything_2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
        device=device
    )
    
    failure_cases = image_outline_extraction_by_mask_multiple_product_types(grounding_model, sam2_predictor, input_dir, output_dir, device=device)

    del sam2_model
    del sam2_predictor
    del grounding_model
    gc.collect()
    torch.cuda.empty_cache()

    return failure_cases
