"""
Test script for the basic renderer to verify it works without requiring a display.
"""

import os
import sys
import numpy as np

# Set headless mode for pygame
os.environ['SDL_VIDEODRIVER'] = 'dummy'

try:
    from basic_renderer import BasicPixelRenderer, PixelLifeRenderer
    print("✓ Successfully imported renderer classes")
except ImportError as e:
    print(f"✗ Failed to import renderer: {e}")
    sys.exit(1)

def test_basic_renderer():
    """Test the BasicPixelRenderer class."""
    print("\nTesting BasicPixelRenderer...")
    
    try:
        # Create renderer (won't open a window in headless mode)
        renderer = BasicPixelRenderer(width=400, height=300, grid_size=20)
        print(f"✓ Created renderer with grid size: {renderer.grid_width}x{renderer.grid_height}")
        
        # Test pixel operations
        renderer.set_pixel(5, 5, True)
        renderer.set_pixel(10, 10, True)
        assert renderer.grid[5, 5] == True
        assert renderer.grid[10, 10] == True
        print("✓ Pixel setting works")
        
        # Test batch pixel operations
        coords = [(1, 1), (2, 2), (3, 3)]
        renderer.set_pixels(coords, True)
        assert all(renderer.grid[y, x] for x, y in coords)
        print("✓ Batch pixel setting works")
        
        # Test clear
        renderer.clear_grid()
        assert not renderer.grid.any()
        print("✓ Grid clearing works")
        
        # Test randomize
        renderer.randomize_grid()
        assert renderer.grid.any()  # Should have some pixels
        print("✓ Grid randomization works")
        
        # Test array update
        test_array = np.random.random((renderer.grid_height, renderer.grid_width)) > 0.5
        renderer.update_from_array(test_array)
        assert np.array_equal(renderer.grid, test_array)
        print("✓ Array update works")
        
        print("✓ All BasicPixelRenderer tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ BasicPixelRenderer test failed: {e}")
        return False

def test_pixel_life_renderer():
    """Test the PixelLifeRenderer class."""
    print("\nTesting PixelLifeRenderer...")
    
    try:
        # Create renderer without environment
        renderer = PixelLifeRenderer(width=400, height=300, grid_size=20)
        print("✓ Created PixelLifeRenderer")
        
        # Test basic functionality inherited from BasicPixelRenderer
        renderer.set_pixel(5, 5, True)
        assert renderer.grid[5, 5] == True
        print("✓ Inherited functionality works")
        
        # Test step counter
        assert renderer.step_count == 0
        print("✓ Step counter initialized")
        
        print("✓ All PixelLifeRenderer tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ PixelLifeRenderer test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Basic Renderer Tests")
    print("=" * 40)
    
    success = True
    success &= test_basic_renderer()
    success &= test_pixel_life_renderer()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! The renderer is working correctly.")
        print("\nTo run the actual renderer (requires display):")
        print("  python basic_renderer.py")
    else:
        print("✗ Some tests failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()