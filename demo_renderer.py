#!/usr/bin/env python3
"""
Demo script for the Basic Pixel Renderer.
Shows some simple patterns and animations.
"""

import time
import numpy as np
from basic_renderer import BasicPixelRenderer


def create_glider(renderer, start_x=10, start_y=10):
    """Create a simple glider pattern."""
    # Conway's Game of Life glider pattern
    glider = [
        (start_x + 1, start_y),
        (start_x + 2, start_y + 1),
        (start_x, start_y + 2),
        (start_x + 1, start_y + 2),
        (start_x + 2, start_y + 2)
    ]
    renderer.set_pixels(glider)


def create_blinker(renderer, start_x=5, start_y=5):
    """Create a simple blinker pattern."""
    blinker = [
        (start_x, start_y),
        (start_x + 1, start_y),
        (start_x + 2, start_y)
    ]
    renderer.set_pixels(blinker)


def main():
    """Run the demo."""
    print("Starting Basic Pixel Renderer Demo")
    print("=" * 40)
    print("Controls:")
    print("  SPACE - Pause/Unpause")
    print("  ESC or Q - Quit")
    print("  Click - Toggle pixels")
    print("  R - Random pixels")
    print("  C - Clear all pixels")
    print("=" * 40)
    
    # Create renderer with smaller grid for better visibility
    renderer = BasicPixelRenderer(width=800, height=600, grid_size=15)
    
    # Create some initial patterns
    create_glider(renderer, 10, 10)
    create_blinker(renderer, 5, 5)
    
    # Add some random scattered pixels
    for _ in range(20):
        x = np.random.randint(0, renderer.grid_width)
        y = np.random.randint(0, renderer.grid_height)
        renderer.set_pixel(x, y, True)
    
    print("\nStarting renderer... Close window or press ESC to exit.")
    
    try:
        renderer.run()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo error: {e}")
        print("Note: This demo requires a display. Run test_renderer.py for headless testing.")


if __name__ == "__main__":
    main()