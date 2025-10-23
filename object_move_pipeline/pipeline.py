import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.sam_utils import ObjectExtractor
from utils.objectclear_utils import ObjectRemover
from utils.anydoor_utils import ObjectPlacer


class ObjectMovePipeline:
    """Complete pipeline for moving objects in images"""
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_fp16: bool = True,
        sam_model: Optional[ObjectExtractor] = None,
        objectclear_model: Optional[ObjectRemover] = None,
        anydoor_model: Optional[ObjectPlacer] = None
    ):
        """
        Initialize the pipeline
        
        Args:
            device: Device to run models on (default: auto-detect)
            use_fp16: Use float16 for faster inference
            sam_model: Pre-initialized SAM model (optional)
            objectclear_model: Pre-initialized ObjectClear model (optional)
            anydoor_model: Pre-initialized AnyDoor model (optional)
        """
        self.device = device
        self.use_fp16 = use_fp16
        
        # Initialize models (lazy loading)
        self._sam = sam_model
        self._objectclear = objectclear_model
        self._anydoor = anydoor_model
    
    @property
    def sam(self):
        """Lazy load SAM model"""
        if self._sam is None:
            self._sam = ObjectExtractor(device=self.device)
        return self._sam
    
    @property
    def objectclear(self):
        """Lazy load ObjectClear model"""
        if self._objectclear is None:
            self._objectclear = ObjectRemover(
                device=self.device,
                use_fp16=self.use_fp16
            )
        return self._objectclear
    
    @property
    def anydoor(self):
        """Lazy load AnyDoor model"""
        if self._anydoor is None:
            self._anydoor = ObjectPlacer(device=self.device)
        return self._anydoor
    
    def create_target_mask(
        self,
        image_shape: Tuple[int, int],
        center_point: Tuple[int, int],
        object_shape: Tuple[int, int],
        dilation_factor: float = 1.0
    ) -> np.ndarray:
        """
        Create a target mask for object placement
        
        Args:
            image_shape: (height, width) of target image
            center_point: (x, y) center point for object placement
            object_shape: (height, width) of object to place
            dilation_factor: Factor to expand mask (default: 1.0)
            
        Returns:
            Binary mask (H, W) with True at placement location
        """
        h, w = image_shape
        obj_h, obj_w = object_shape
        
        # Adjust object size by dilation factor
        obj_h = int(obj_h * dilation_factor)
        obj_w = int(obj_w * dilation_factor)
        
        # Calculate bounding box
        cx, cy = center_point
        x1 = max(0, cx - obj_w // 2)
        x2 = min(w, cx + obj_w // 2)
        y1 = max(0, cy - obj_h // 2)
        y2 = min(h, cy + obj_h // 2)
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        
        return mask
    
    def run(
        self,
        input_image_path: str,
        object_point: Tuple[int, int],
        target_point: Tuple[int, int],
        output_path: str,
        save_intermediate: bool = True,
        output_dir: Optional[str] = None,
        objectclear_steps: int = 20,
        objectclear_guidance: float = 2.5,
        anydoor_steps: int = 50,
        anydoor_guidance: float = 5.0,
        target_mask_dilation: float = 1.0
    ) -> Image.Image:
        """
        Run the complete object move pipeline
        
        Args:
            input_image_path: Path to input image
            object_point: (x, y) point on the object to move
            target_point: (x, y) center point for new object location
            output_path: Path to save final result
            save_intermediate: Save intermediate results
            output_dir: Directory for intermediate results (default: same as output)
            objectclear_steps: Number of diffusion steps for ObjectClear
            objectclear_guidance: Guidance scale for ObjectClear
            anydoor_steps: Number of DDIM steps for AnyDoor
            anydoor_guidance: Guidance scale for AnyDoor
            target_mask_dilation: Factor to expand target mask (> 1.0 = larger)
            
        Returns:
            Final PIL Image with object moved
        """
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.dirname(output_path)
            if not output_dir:
                output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("OBJECT MOVE PIPELINE")
        print("=" * 60)
        print(f"Input: {input_image_path}")
        print(f"Object point: {object_point}")
        print(f"Target point: {target_point}")
        print(f"Output: {output_path}")
        print("=" * 60)
        
        # Load input image
        input_image = Image.open(input_image_path).convert("RGB")
        h, w = input_image.size[1], input_image.size[0]  # PIL: (W, H)
        
        # ===== STEP 1: Extract Object with SAM =====
        print("\n[STEP 1/3] Extracting object with SAM...")
        extraction_result = self.sam.extract(
            input_image,
            object_point,
            save_intermediate=save_intermediate,
            output_dir=output_dir
        )
        
        object_mask = extraction_result['mask']
        object_rgba = extraction_result['object_rgba']
        object_cropped = extraction_result['object_cropped']
        object_bbox = extraction_result['bbox']
        
        print(f"  ✓ Object extracted (bbox: {object_bbox})")
        
        # ===== STEP 2: Remove Object and Fill Background =====
        print("\n[STEP 2/3] Removing object and filling background with ObjectClear...")
        background_filled = self.objectclear.remove(
            input_image,
            object_mask,
            steps=objectclear_steps,
            guidance_scale=objectclear_guidance,
            save_intermediate=save_intermediate,
            output_dir=output_dir
        )
        
        print(f"  ✓ Background filled successfully")
        
        # ===== STEP 3: Place Object at New Location with AnyDoor =====
        print("\n[STEP 3/3] Placing object at new location with AnyDoor...")
        
        # Convert background to numpy
        background_np = np.array(background_filled)
        
        # Convert object to numpy (RGB)
        object_rgb = object_rgba[:, :, :3]
        object_mask_uint8 = (object_mask > 0).astype(np.uint8)
        
        # Create target mask for new location
        obj_h, obj_w = object_cropped.shape[0], object_cropped.shape[1]
        target_mask = self.create_target_mask(
            (h, w),
            target_point,
            (obj_h, obj_w),
            dilation_factor=target_mask_dilation
        )
        
        # Save target mask if intermediate results are enabled
        if save_intermediate:
            Image.fromarray(target_mask * 255, mode='L').save(
                os.path.join(output_dir, "step3_target_mask.png")
            )
        
        # Place object on background
        final_result_np = self.anydoor.place(
            object_rgb,
            object_mask_uint8,
            background_np,
            target_mask,
            guidance_scale=anydoor_guidance,
            ddim_steps=anydoor_steps,
            save_intermediate=save_intermediate,
            output_dir=output_dir
        )
        
        print(f"  ✓ Object placed at new location")
        
        # Convert to PIL Image
        final_result = Image.fromarray(final_result_np.astype(np.uint8))
        
        # Save final result
        final_result.save(output_path)
        print(f"\n{'=' * 60}")
        print(f"✓ COMPLETE! Final result saved to: {output_path}")
        print(f"{'=' * 60}\n")
        
        return final_result


def parse_point(point_str: str) -> Tuple[int, int]:
    """Parse point string 'x,y' to tuple (x, y)"""
    try:
        x, y = map(int, point_str.split(','))
        return (x, y)
    except:
        raise ValueError(f"Invalid point format: {point_str}. Expected format: 'x,y'")


def main():
    parser = argparse.ArgumentParser(
        description="Move objects in images using SAM + ObjectClear + AnyDoor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python pipeline.py --input image.jpg --object_point 100,200 --target_point 500,300
  
  # With custom output path
  python pipeline.py -i image.jpg -op 100,200 -tp 500,300 -o result.png
  
  # Adjust quality parameters
  python pipeline.py -i image.jpg -op 100,200 -tp 500,300 \\
      --objectclear_steps 30 --anydoor_steps 100
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input image path'
    )
    parser.add_argument(
        '-op', '--object_point',
        type=str,
        required=True,
        help='Point on object to move (format: x,y)'
    )
    parser.add_argument(
        '-tp', '--target_point',
        type=str,
        required=True,
        help='Target center point for object placement (format: x,y)'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output image path (default: ./outputs/result.png)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory for intermediate results (default: same as output)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )
    parser.add_argument(
        '--no_fp16',
        action='store_true',
        help='Disable float16 for ObjectClear (slower but more precise)'
    )
    parser.add_argument(
        '--no_intermediate',
        action='store_true',
        help='Do not save intermediate results'
    )
    
    # ObjectClear parameters
    parser.add_argument(
        '--objectclear_steps',
        type=int,
        default=20,
        help='Number of diffusion steps for ObjectClear (default: 20)'
    )
    parser.add_argument(
        '--objectclear_guidance',
        type=float,
        default=2.5,
        help='Guidance scale for ObjectClear (default: 2.5)'
    )
    
    # AnyDoor parameters
    parser.add_argument(
        '--anydoor_steps',
        type=int,
        default=50,
        help='Number of DDIM steps for AnyDoor (default: 50)'
    )
    parser.add_argument(
        '--anydoor_guidance',
        type=float,
        default=5.0,
        help='Guidance scale for AnyDoor (default: 5.0)'
    )
    parser.add_argument(
        '--target_mask_dilation',
        type=float,
        default=1.0,
        help='Target mask dilation factor (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Parse points
    try:
        object_point = parse_point(args.object_point)
        target_point = parse_point(args.target_point)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Set default output path
    if args.output is None:
        args.output = "./outputs/result.png"
    
    # Create pipeline
    pipeline = ObjectMovePipeline(
        device=args.device,
        use_fp16=not args.no_fp16
    )
    
    # Run pipeline
    try:
        pipeline.run(
            input_image_path=args.input,
            object_point=object_point,
            target_point=target_point,
            output_path=args.output,
            save_intermediate=not args.no_intermediate,
            output_dir=args.output_dir,
            objectclear_steps=args.objectclear_steps,
            objectclear_guidance=args.objectclear_guidance,
            anydoor_steps=args.anydoor_steps,
            anydoor_guidance=args.anydoor_guidance,
            target_mask_dilation=args.target_mask_dilation
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

