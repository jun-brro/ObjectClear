import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple

# Add parent directory to path to import objectclear
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from objectclear.pipelines import ObjectClearPipeline
from objectclear.utils import resize_by_short_side


class ObjectRemover:
    """Remove objects from images and fill with natural background"""
    
    def __init__(
        self, 
        device: Optional[str] = None,
        use_fp16: bool = True,
        seed: int = 42,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize ObjectClear pipeline
        
        Args:
            device: Device to run model on
            use_fp16: Use float16 for faster inference
            seed: Random seed for reproducibility
            cache_dir: Cache directory for model weights
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        torch_dtype = torch.float16 if use_fp16 else torch.float32
        variant = "fp16" if use_fp16 else None
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print(f"[ObjectClear] Loading ObjectClear pipeline on {self.device}...")
        self.pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
            "jixin0101/ObjectClear",
            torch_dtype=torch_dtype,
            apply_attention_guided_fusion=True,
            cache_dir=cache_dir,
            variant=variant,
        )
        self.pipe.to(self.device)
        print("[ObjectClear] Pipeline loaded successfully")
    
    def remove(
        self,
        image: Image.Image,
        mask: np.ndarray,
        steps: int = 20,
        guidance_scale: float = 2.5,
        save_intermediate: bool = False,
        output_dir: str = "./outputs"
    ) -> Image.Image:
        """
        Remove object from image and fill with natural background
        
        Args:
            image: Input PIL Image (RGB)
            mask: Binary mask (H, W) - True/255 for object to remove
            steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            save_intermediate: Whether to save intermediate results
            output_dir: Directory to save outputs
            
        Returns:
            PIL Image with object removed and background filled
        """
        # Convert mask to PIL Image
        if isinstance(mask, np.ndarray):
            if mask.dtype == bool:
                mask = (mask.astype(np.uint8) * 255)
            mask_pil = Image.fromarray(mask, mode="L")
        else:
            mask_pil = mask
        
        # Store original size
        original_size = image.size
        
        # Resize for optimal performance (shorter side = 512)
        image_resized = resize_by_short_side(image, 512, resample=Image.BICUBIC)
        mask_resized = resize_by_short_side(mask_pil, 512, resample=Image.NEAREST)
        
        w, h = image_resized.size
        
        # Run ObjectClear
        result = self.pipe(
            prompt="remove the instance of object",
            image=image_resized,
            mask_image=mask_resized,
            generator=self.generator,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=h,
            width=w,
            return_attn_map=False,
        )
        
        inpainted_image = result.images[0]
        
        # Resize back to original size
        inpainted_image = inpainted_image.resize(original_size, Image.BICUBIC)
        
        # Save intermediate results if requested
        if save_intermediate:
            os.makedirs(output_dir, exist_ok=True)
            inpainted_image.save(os.path.join(output_dir, "step2_background_filled.png"))
            print(f"[ObjectClear] Intermediate results saved to {output_dir}")
        
        return inpainted_image


def remove_object_from_background(
    image_path: str,
    mask: np.ndarray,
    device: Optional[str] = None,
    steps: int = 20,
    guidance_scale: float = 2.5,
    save_intermediate: bool = False,
    output_dir: str = "./outputs"
) -> Image.Image:
    """
    Convenience function to remove object from image file
    
    Args:
        image_path: Path to input image
        mask: Binary mask of object to remove
        device: Device to run model on
        steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        save_intermediate: Whether to save intermediate results
        output_dir: Directory to save outputs
        
    Returns:
        PIL Image with object removed
    """
    image = Image.open(image_path).convert("RGB")
    remover = ObjectRemover(device=device)
    return remover.remove(image, mask, steps, guidance_scale, save_intermediate, output_dir)

