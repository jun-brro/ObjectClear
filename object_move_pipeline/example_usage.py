import os
import sys
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import ObjectMovePipeline


def example_basic():
    """Basic example: Move an object in an image"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Object Move")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ObjectMovePipeline(use_fp16=True)
    
    # Run pipeline
    result = pipeline.run(
        input_image_path='../raw_image.png',
        object_point=(1323, 832),  # Click on object
        target_point=(500, 500),    # Where to move it
        output_path='../outputs/example_basic_result.png',
        save_intermediate=True
    )
    
    print("✓ Basic example complete!\n")


def example_high_quality():
    """High-quality example with more steps"""
    print("=" * 60)
    print("EXAMPLE 2: High Quality Object Move")
    print("=" * 60)
    
    pipeline = ObjectMovePipeline(use_fp16=True)
    
    result = pipeline.run(
        input_image_path='../raw_image.png',
        object_point=(1323, 832),
        target_point=(800, 400),
        output_path='../outputs/example_hq_result.png',
        save_intermediate=True,
        objectclear_steps=30,      # More steps for better quality
        objectclear_guidance=3.0,
        anydoor_steps=100,          # More steps for better quality
        anydoor_guidance=5.5
    )
    
    print("✓ High-quality example complete!\n")


def example_batch_processing():
    """Process multiple target locations"""
    print("=" * 60)
    print("EXAMPLE 3: Batch Processing (Multiple Locations)")
    print("=" * 60)
    
    # Initialize pipeline once (reuse models)
    pipeline = ObjectMovePipeline(use_fp16=True)
    
    # Define multiple target locations
    target_locations = [
        (300, 300, "top_left"),
        (800, 300, "top_right"),
        (300, 700, "bottom_left"),
        (800, 700, "bottom_right"),
    ]
    
    for i, (x, y, position_name) in enumerate(target_locations, 1):
        print(f"\n[{i}/{len(target_locations)}] Moving to {position_name}...")
        
        result = pipeline.run(
            input_image_path='../raw_image.png',
            object_point=(1323, 832),
            target_point=(x, y),
            output_path=f'../outputs/example_batch_{position_name}.png',
            save_intermediate=False,  # Skip intermediate for batch
            output_dir=f'../outputs/batch_{position_name}'
        )
    
    print("\n✓ Batch processing complete!\n")


def example_custom_models():
    """Example with pre-loaded models (for efficiency)"""
    print("=" * 60)
    print("EXAMPLE 4: Using Pre-loaded Models")
    print("=" * 60)
    
    from utils.sam_utils import ObjectExtractor
    from utils.objectclear_utils import ObjectRemover
    from utils.anydoor_utils import ObjectPlacer
    
    # Pre-load models
    print("Loading models...")
    sam = ObjectExtractor()
    objectclear = ObjectRemover(use_fp16=True)
    anydoor = ObjectPlacer()
    
    # Create pipeline with pre-loaded models
    pipeline = ObjectMovePipeline(
        sam_model=sam,
        objectclear_model=objectclear,
        anydoor_model=anydoor
    )
    
    # Now run multiple times without reloading
    for i in range(3):
        print(f"\nProcessing image {i+1}/3...")
        result = pipeline.run(
            input_image_path='../raw_image.png',
            object_point=(1323, 832),
            target_point=(400 + i*200, 400),
            output_path=f'../outputs/example_preloaded_{i+1}.png',
            save_intermediate=False
        )
    
    print("\n✓ Pre-loaded models example complete!\n")


def example_step_by_step():
    """Example using individual components"""
    print("=" * 60)
    print("EXAMPLE 5: Step-by-Step (Using Individual Components)")
    print("=" * 60)
    
    from utils.sam_utils import ObjectExtractor
    from utils.objectclear_utils import ObjectRemover
    from utils.anydoor_utils import ObjectPlacer
    import numpy as np
    
    # Load input
    input_image = Image.open('../raw_image.png').convert('RGB')
    object_point = (1323, 832)
    target_point = (600, 600)
    
    # Step 1: Extract object
    print("\n[Step 1] Extracting object with SAM...")
    extractor = ObjectExtractor()
    extraction = extractor.extract(
        input_image, 
        object_point,
        save_intermediate=True,
        output_dir='../outputs/step_by_step'
    )
    object_mask = extraction['mask']
    object_rgba = extraction['object_rgba']
    
    # Step 2: Remove object
    print("\n[Step 2] Removing object with ObjectClear...")
    remover = ObjectRemover()
    background = remover.remove(
        input_image,
        object_mask,
        save_intermediate=True,
        output_dir='../outputs/step_by_step'
    )
    
    # Step 3: Place object
    print("\n[Step 3] Placing object with AnyDoor...")
    placer = ObjectPlacer()
    
    # Create target mask
    h, w = input_image.size[1], input_image.size[0]
    obj_h, obj_w = extraction['object_cropped'].shape[0:2]
    target_mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = target_point
    x1 = max(0, cx - obj_w // 2)
    x2 = min(w, cx + obj_w // 2)
    y1 = max(0, cy - obj_h // 2)
    y2 = min(h, cy + obj_h // 2)
    target_mask[y1:y2, x1:x2] = 1
    
    final = placer.place(
        object_rgba[:, :, :3],
        (object_mask > 0).astype(np.uint8),
        np.array(background),
        target_mask,
        save_intermediate=True,
        output_dir='../outputs/step_by_step'
    )
    
    # Save final
    Image.fromarray(final.astype(np.uint8)).save(
        '../outputs/example_step_by_step.png'
    )
    
    print("\n✓ Step-by-step example complete!\n")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("OBJECT MOVE PIPELINE - EXAMPLES")
    print("="*60 + "\n")
    
    # Create output directory
    os.makedirs('../outputs', exist_ok=True)
    
    # Check if input image exists
    if not os.path.exists('../raw_image.png'):
        print("❌ Error: ../raw_image.png not found!")
        print("Please ensure you have an input image.")
        return
    
    try:
        # Run examples
        print("Running examples...\n")
        
        # Uncomment the examples you want to run:
        
        example_basic()
        # example_high_quality()
        # example_batch_processing()
        # example_custom_models()
        # example_step_by_step()
        
        print("\n" + "="*60)
        print("✓ ALL EXAMPLES COMPLETE!")
        print("="*60)
        print("\nCheck the outputs/ directory for results.")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

