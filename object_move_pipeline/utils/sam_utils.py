import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
from typing import List, Tuple, Dict, Optional


class ObjectExtractor:
    """Extract objects from images using SAM"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize SAM model
        
        Args:
            device: Device to run model on. If None, auto-detects CUDA availability
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SAM] Loading SAM model on {self.device}...")
        self.model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        print("[SAM] Model loaded successfully")
    
    def extract(
        self, 
        image: Image.Image, 
        point: Tuple[int, int],
        save_intermediate: bool = False,
        output_dir: str = "./outputs"
    ) -> Dict[str, np.ndarray]:
        """
        Extract object from image using a point prompt
        
        Args:
            image: Input PIL Image (RGB)
            point: (x, y) coordinates of a point on the object
            save_intermediate: Whether to save intermediate results
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing:
                - 'mask': Binary mask (H, W) - True for object, False for background
                - 'object_rgba': Object with transparent background (H, W, 4)
                - 'object_cropped': Tightly cropped object (H', W', 3)
                - 'bbox': Bounding box (y_min, y_max, x_min, x_max)
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Prepare input
        input_points = [[[point[0], point[1]]]]
        inputs = self.processor(image, input_points=input_points, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process masks
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores
        
        # Select best mask
        masks_tensor = masks[0]
        scores_tensor = scores[0, 0].detach().cpu()
        best_mask_idx = torch.argmax(scores_tensor).item()
        best_mask = masks_tensor[0, best_mask_idx]
        best_mask_bool = (best_mask > 0.5).numpy()
        
        # Convert image to numpy
        orig_np = np.array(image)
        
        # Create RGBA image with transparent background
        alpha = (best_mask_bool * 255).astype(np.uint8)
        object_rgba = np.dstack([orig_np, alpha])
        
        # Create tightly cropped object
        object_rgb = orig_np.copy()
        object_rgb[~best_mask_bool] = 0
        ys, xs = np.where(best_mask_bool)
        
        if ys.size > 0 and xs.size > 0:
            y_min, y_max = int(ys.min()), int(ys.max()) + 1
            x_min, x_max = int(xs.min()), int(xs.max()) + 1
            cropped_object = object_rgb[y_min:y_max, x_min:x_max]
            bbox = (y_min, y_max, x_min, x_max)
        else:
            cropped_object = object_rgb
            bbox = (0, object_rgb.shape[0], 0, object_rgb.shape[1])
        
        result = {
            'mask': best_mask_bool,
            'object_rgba': object_rgba,
            'object_cropped': cropped_object,
            'bbox': bbox
        }
        
        # Save intermediate results if requested
        if save_intermediate:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Save binary mask
            mask_uint8 = (best_mask_bool.astype(np.uint8) * 255)
            Image.fromarray(mask_uint8, mode="L").save(
                os.path.join(output_dir, "step1_object_mask.png")
            )
            
            # Save RGBA object
            Image.fromarray(object_rgba, mode="RGBA").save(
                os.path.join(output_dir, "step1_object_rgba.png")
            )
            
            # Save cropped object
            Image.fromarray(cropped_object).save(
                os.path.join(output_dir, "step1_object_cropped.png")
            )
            
            print(f"[SAM] Intermediate results saved to {output_dir}")
        
        return result


def extract_object_with_sam(
    image_path: str,
    point: Tuple[int, int],
    device: Optional[str] = None,
    save_intermediate: bool = False,
    output_dir: str = "./outputs"
) -> Dict[str, np.ndarray]:
    """
    Convenience function to extract object from image file
    
    Args:
        image_path: Path to input image
        point: (x, y) coordinates of a point on the object
        device: Device to run model on
        save_intermediate: Whether to save intermediate results
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with extraction results (see ObjectExtractor.extract)
    """
    image = Image.open(image_path).convert("RGB")
    extractor = ObjectExtractor(device=device)
    return extractor.extract(image, point, save_intermediate, output_dir)

