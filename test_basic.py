#!/usr/bin/env python3

def test_imports():
    print("Testing imports...")
    try:
        import numpy as np
        print("✓ numpy imported successfully")
        
        import pygame
        print("✓ pygame imported successfully")
        
        from env import PixelLifeEnv
        print("✓ PixelLifeEnv imported successfully")
        
        from renderer import BasicRenderer
        print("✓ BasicRenderer imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_environment():
    print("\nTesting environment...")
    try:
        from env import PixelLifeEnv
        env = PixelLifeEnv(size=32)
        initial_state = env.reset()
        print(f"✓ Environment created with size {env.size}x{env.size}")
        
        # Test a few steps
        for i in range(3):
            state = env.step()
            print(f"✓ Step {i+1} completed")
        
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def main():
    print("=== Pixel Life Basic Test ===")
    
    imports_ok = test_imports()
    env_ok = test_environment()
    
    if imports_ok and env_ok:
        print("\n✓ All tests passed!")
        print("\nTo run with rendering:")
        print("  python train.py --render")
        print("\nTo run renderer only:")
        print("  python renderer.py")
    else:
        print("\n✗ Some tests failed!")

if __name__ == "__main__":
    main()