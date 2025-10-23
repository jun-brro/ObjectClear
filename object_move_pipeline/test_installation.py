import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    tests = [
        ("PyTorch", "torch"),
        ("Torchvision", "torchvision"),
        ("NumPy", "numpy"),
        ("PIL/Pillow", "PIL"),
        ("OpenCV", "cv2"),
        ("Transformers", "transformers"),
        ("Diffusers", "diffusers"),
        ("OmegaConf", "omegaconf"),
        ("Einops", "einops"),
        ("PyTorch Lightning", "pytorch_lightning"),
        ("Albumentations", "albumentations"),
        ("Scikit-image", "skimage"),
    ]
    
    failed = []
    for name, module in tests:
        try:
            __import__(module)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - FAILED: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ {len(failed)} packages failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "=" * 60)
    print("TESTING CUDA")
    print("=" * 60)
    
    import torch
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print("\n✅ CUDA is ready!")
        return True
    else:
        print("\n⚠️ CUDA not available. Pipeline will run on CPU (slower).")
        return False


def test_pipeline_imports():
    """Test if pipeline modules can be imported"""
    print("\n" + "=" * 60)
    print("TESTING PIPELINE MODULES")
    print("=" * 60)
    
    import os
    import sys
    
    # Add pipeline to path
    pipeline_dir = os.path.dirname(__file__)
    sys.path.insert(0, pipeline_dir)
    
    tests = [
        ("Pipeline Main", "pipeline", "ObjectMovePipeline"),
        ("SAM Utils", "utils.sam_utils", "ObjectExtractor"),
        ("ObjectClear Utils", "utils.objectclear_utils", "ObjectRemover"),
        ("AnyDoor Utils", "utils.anydoor_utils", "ObjectPlacer"),
    ]
    
    failed = []
    for name, module, class_name in tests:
        try:
            mod = __import__(module, fromlist=[class_name])
            cls = getattr(mod, class_name)
            print(f"✓ {name:25s} - OK")
        except Exception as e:
            print(f"✗ {name:25s} - FAILED: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ {len(failed)} modules failed: {', '.join(failed)}")
        return False
    else:
        print("\n✅ All pipeline modules imported successfully!")
        return True


def test_parent_modules():
    """Test if parent ObjectClear and AnyDoor modules are accessible"""
    print("\n" + "=" * 60)
    print("TESTING PARENT MODULES")
    print("=" * 60)
    
    import os
    import sys
    
    # Add parent directories
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, os.path.join(parent_dir, 'AnyDoor'))
    
    tests = [
        ("ObjectClear Pipeline", "objectclear.pipelines", "ObjectClearPipeline"),
        ("ObjectClear Utils", "objectclear.utils", "resize_by_short_side"),
    ]
    
    # AnyDoor tests (may not be fully set up yet)
    anydoor_tests = [
        ("AnyDoor Model", "cldm.model", "create_model"),
        ("AnyDoor Data Utils", "datasets.data_utils", "get_bbox_from_mask"),
    ]
    
    failed = []
    for name, module, item in tests:
        try:
            mod = __import__(module, fromlist=[item])
            obj = getattr(mod, item)
            print(f"✓ {name:25s} - OK")
        except Exception as e:
            print(f"✗ {name:25s} - FAILED: {e}")
            failed.append(name)
    
    print("\nTesting AnyDoor modules (may require additional setup):")
    for name, module, item in anydoor_tests:
        try:
            mod = __import__(module, fromlist=[item])
            obj = getattr(mod, item)
            print(f"✓ {name:25s} - OK")
        except Exception as e:
            print(f"⚠ {name:25s} - Not available: {str(e)[:50]}")
    
    if failed:
        print(f"\n❌ {len(failed)} critical modules failed: {', '.join(failed)}")
        return False
    else:
        print("\n✅ Core modules accessible!")
        return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("OBJECT MOVE PIPELINE - INSTALLATION TEST")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("CUDA Availability", test_cuda()))
    results.append(("Pipeline Modules", test_pipeline_imports()))
    results.append(("Parent Modules", test_parent_modules()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nYou're ready to use the Object Move Pipeline!")
        print("\nQuick start:")
        print("  python pipeline.py -i image.png -op 100,200 -tp 500,300")
        return 0
    else:
        print("\n" + "="*60)
        print("⚠️ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("\nFor parent module issues:")
        print("  pip install -e .. (from ObjectClear root)")
        return 1


if __name__ == '__main__':
    sys.exit(main())

