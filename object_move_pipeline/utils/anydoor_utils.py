import os
import sys
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional

# Add AnyDoor and dinov2 directories to path
anydoor_path = os.path.join(os.path.dirname(__file__), '../../AnyDoor')
sys.path.insert(0, anydoor_path)
sys.path.insert(0, os.path.join(anydoor_path, 'dinov2'))

import torch
import einops
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity
from datasets.data_utils import *
from omegaconf import OmegaConf


class ObjectPlacer:
    """Place objects on new backgrounds using AnyDoor"""
    
    def __init__(
        self,
        config_path: str = None,
        device: Optional[str] = None
    ):
        """
        Initialize AnyDoor model
        
        Args:
            config_path: Path to inference config file
            device: Device to run model on
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '../../AnyDoor/configs/inference.yaml'
            )
        
        config = OmegaConf.load(config_path)
        model_ckpt = config.pretrained_model
        model_config = config.config_file
        
        print(f"[AnyDoor] Loading AnyDoor model on {self.device}...")
        disable_verbosity()
        
        self.model = create_model(model_config).cpu()
        self.model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
        self.model = self.model.to(self.device)
        self.ddim_sampler = DDIMSampler(self.model)
        
        print("[AnyDoor] Model loaded successfully")
    
    def process_pairs(
        self, 
        ref_image: np.ndarray, 
        ref_mask: np.ndarray, 
        tar_image: np.ndarray, 
        tar_mask: np.ndarray
    ):
        """
        Process reference and target image pairs
        
        Args:
            ref_image: Reference object image (RGB, np.ndarray)
            ref_mask: Reference object mask (binary, np.ndarray)
            tar_image: Target background image (RGB, np.ndarray)
            tar_mask: Target position mask (binary, np.ndarray)
            
        Returns:
            Dictionary with processed data for model inference
        """
        # ========= Reference (Object) ===========
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        
        # Filter mask
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)
        
        y1, y2, x1, x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
        ref_mask = ref_mask[y1:y2, x1:x2]
        
        # Expand
        ratio = 1.2
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
        
        # To square and resize
        masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
        masked_ref_image = cv2.resize(masked_ref_image, (224, 224)).astype(np.uint8)
        
        ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value=0, random=False)
        ref_mask_3 = cv2.resize(ref_mask_3, (224, 224)).astype(np.uint8)
        ref_mask = ref_mask_3[:, :, 0]
        
        masked_ref_image_compose = masked_ref_image
        ref_mask_compose = ref_mask
        ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose / 255)
        
        # ========= Target (Background) ===========
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1, 1.2])
        
        # Crop
        tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
        y1, y2, x1, x2 = tar_box_yyxx_crop
        
        cropped_target_image = tar_image[y1:y2, x1:x2, :]
        tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
        y1, y2, x1, x2 = tar_box_yyxx
        
        # Collage
        ref_image_collage = cv2.resize(ref_image_collage, (x2 - x1, y2 - y1))
        ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2 - x1, y2 - y1))
        ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)
        
        collage = cropped_target_image.copy()
        collage[y1:y2, x1:x2, :] = ref_image_collage
        
        collage_mask = cropped_target_image.copy() * 0.0
        collage_mask[y1:y2, x1:x2, :] = 1.0
        
        # Size before and after padding
        H1, W1 = collage.shape[0], collage.shape[1]
        cropped_target_image = pad_to_square(cropped_target_image, pad_value=0, random=False).astype(np.uint8)
        collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value=-1, random=False).astype(np.uint8)
        
        H2, W2 = collage.shape[0], collage.shape[1]
        cropped_target_image = cv2.resize(cropped_target_image, (512, 512)).astype(np.float32)
        collage = cv2.resize(collage, (512, 512)).astype(np.float32)
        collage_mask = (cv2.resize(collage_mask, (512, 512)).astype(np.float32) > 0.5).astype(np.float32)
        
        masked_ref_image_aug = masked_ref_image / 255
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0
        collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)
        
        item = dict(
            ref=masked_ref_image_aug.copy(),
            jpg=cropped_target_image.copy(),
            hint=collage.copy(),
            extra_sizes=np.array([H1, W1, H2, W2]),
            tar_box_yyxx_crop=np.array(tar_box_yyxx_crop)
        )
        return item
    
    def crop_back(self, pred: np.ndarray, tar_image: np.ndarray, extra_sizes, tar_box_yyxx_crop):
        """Crop prediction back to original image size"""
        H1, W1, H2, W2 = extra_sizes
        y1, y2, x1, x2 = tar_box_yyxx_crop
        pred = cv2.resize(pred, (W2, H2))
        m = 5  # margin pixels
        
        if W1 == H1:
            tar_image[y1 + m:y2 - m, x1 + m:x2 - m, :] = pred[m:-m, m:-m]
            return tar_image
        
        if W1 < W2:
            pad1 = int((W2 - W1) / 2)
            pad2 = W2 - W1 - pad1
            pred = pred[:, pad1:-pad2, :]
        else:
            pad1 = int((H2 - H1) / 2)
            pad2 = H2 - H1 - pad1
            pred = pred[pad1:-pad2, :, :]
        
        gen_image = tar_image.copy()
        gen_image[y1 + m:y2 - m, x1 + m:x2 - m, :] = pred[m:-m, m:-m]
        return gen_image
    
    def place(
        self,
        ref_image: np.ndarray,
        ref_mask: np.ndarray,
        tar_image: np.ndarray,
        tar_mask: np.ndarray,
        guidance_scale: float = 5.0,
        ddim_steps: int = 50,
        save_intermediate: bool = False,
        output_dir: str = "./outputs"
    ) -> np.ndarray:
        """
        Place object on background at specified location
        
        Args:
            ref_image: Reference object image (RGB, np.ndarray)
            ref_mask: Reference object mask (binary, np.ndarray)
            tar_image: Target background image (RGB, np.ndarray)
            tar_mask: Target position mask (binary, np.ndarray)
            guidance_scale: Classifier-free guidance scale
            ddim_steps: Number of DDIM sampling steps
            save_intermediate: Whether to save intermediate results
            output_dir: Directory to save outputs
            
        Returns:
            Generated image with object placed on background (RGB, np.ndarray)
        """
        # Process image pairs
        item = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        
        ref = item['ref']
        tar = item['jpg']
        hint = item['hint']
        num_samples = 1
        
        # Prepare control and clip inputs
        control = torch.from_numpy(hint.copy()).float().to(self.device)
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        clip_input = torch.from_numpy(ref.copy()).float().to(self.device)
        clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
        clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()
        
        # Prepare conditioning
        guess_mode = False
        H, W = 512, 512
        
        cond = {
            "c_concat": [control],
            "c_crossattn": [self.model.get_learned_conditioning(clip_input)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [self.model.get_learned_conditioning([torch.zeros((1, 3, 224, 224)).to(self.device)] * num_samples)]
        }
        shape = (4, H // 8, W // 8)
        
        # Sampling
        strength = 1.0
        self.model.control_scales = [strength] * 13
        
        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps, num_samples,
            shape, cond, verbose=False, eta=0.0,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=un_cond
        )
        
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()
        
        pred = x_samples[0]
        pred = np.clip(pred, 0, 255)
        
        # Crop back to original size
        sizes = item['extra_sizes']
        tar_box_yyxx_crop = item['tar_box_yyxx_crop']
        gen_image = self.crop_back(pred, tar_image, sizes, tar_box_yyxx_crop)
        
        # Save intermediate results if requested
        if save_intermediate:
            os.makedirs(output_dir, exist_ok=True)
            Image.fromarray(gen_image.astype(np.uint8)).save(
                os.path.join(output_dir, "step3_final_result.png")
            )
            print(f"[AnyDoor] Final result saved to {output_dir}")
        
        return gen_image


def place_object_on_background(
    ref_image: np.ndarray,
    ref_mask: np.ndarray,
    background_path: str,
    target_mask: np.ndarray,
    guidance_scale: float = 5.0,
    ddim_steps: int = 50,
    save_intermediate: bool = False,
    output_dir: str = "./outputs"
) -> np.ndarray:
    """
    Convenience function to place object on background
    
    Args:
        ref_image: Reference object image
        ref_mask: Reference object mask
        background_path: Path to background image
        target_mask: Where to place the object
        guidance_scale: Guidance scale
        ddim_steps: Number of steps
        save_intermediate: Save intermediate results
        output_dir: Output directory
        
    Returns:
        Generated image with object placed
    """
    tar_image = cv2.imread(background_path)
    tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
    
    placer = ObjectPlacer()
    return placer.place(
        ref_image, ref_mask, tar_image, target_mask,
        guidance_scale, ddim_steps, save_intermediate, output_dir
    )

