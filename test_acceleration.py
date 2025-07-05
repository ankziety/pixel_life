#!/usr/bin/env python3
"""
Test script to demonstrate Apple Silicon acceleration command line options.
"""

import sys
import os
import time
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_acceleration_options():
    """Test the acceleration command line options."""
    print("🧪 Testing Apple Silicon Acceleration Options")
    print("=" * 50)
    
    # Test if acceleration is available
    try:
        from apple_acceleration import create_accelerated_env, benchmark_acceleration
        from accelerated_training import train_accelerated_pixel_life
        from accelerated_renderer import create_renderer
        APPLE_ACCELERATION_AVAILABLE = True
        print("✅ Apple Silicon acceleration is available")
    except ImportError as e:
        APPLE_ACCELERATION_AVAILABLE = False
        print(f"❌ Apple Silicon acceleration not available: {e}")
        return
    
    # Test basic environment creation
    print("\n🔧 Testing environment creation...")
    try:
        # Standard environment
        from env import PixelLifeEnv
        standard_env = PixelLifeEnv(H=30, W=30)
        print("✅ Standard environment created")
        
        # Accelerated environment
        accelerated_env = create_accelerated_env(H=30, W=30)
        print("✅ Accelerated environment created")
        
        # Test basic functionality
        obs_std = standard_env.reset()
        obs_acc = accelerated_env.reset()
        print("✅ Both environments reset successfully")
        
        # Test step functionality
        pixel_actions = {}
        for coord in standard_env.live_pixels:
            pixel_actions[coord] = (np.random.randint(0, 4), np.random.randint(0, 4))
        
        obs_std, rewards_std, done_std, truncated_std, info_std = standard_env.step(0, pixel_actions)
        obs_acc, rewards_acc, done_acc, truncated_acc, info_acc = accelerated_env.step(0, pixel_actions)
        print("✅ Both environments stepped successfully")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return
    
    # Test performance comparison
    print("\n⚡ Testing performance comparison...")
    try:
        # Benchmark standard environment
        start_time = time.time()
        for _ in range(1000):
            pixel_actions = {}
            for coord in standard_env.live_pixels:
                pixel_actions[coord] = (np.random.randint(0, 4), np.random.randint(0, 4))
            obs_std, rewards_std, done_std, truncated_std, info_std = standard_env.step(0, pixel_actions)
            if done_std:
                standard_env.reset()
        standard_time = time.time() - start_time
        
        # Benchmark accelerated environment
        start_time = time.time()
        for _ in range(1000):
            pixel_actions = {}
            for coord in accelerated_env.live_pixels:
                pixel_actions[coord] = (np.random.randint(0, 4), np.random.randint(0, 4))
            obs_acc, rewards_acc, done_acc, truncated_acc, info_acc = accelerated_env.step(0, pixel_actions)
            if done_acc:
                accelerated_env.reset()
        accelerated_time = time.time() - start_time
        
        print(f"   Standard environment: {standard_time:.3f}s")
        print(f"   Accelerated environment: {accelerated_time:.3f}s")
        if standard_time > 0:
            speedup = standard_time / accelerated_time
            print(f"   🚀 Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
    
    # Test renderer creation
    print("\n🎨 Testing renderer creation...")
    try:
        renderer = create_renderer(accelerated_env, renderer_type="auto")
        print("✅ Accelerated renderer created")
    except Exception as e:
        print(f"❌ Renderer test failed: {e}")
    
    print("\n✅ All tests completed!")

def show_usage_examples():
    """Show usage examples for the acceleration options."""
    print("\n📖 Usage Examples")
    print("=" * 30)
    
    print("Basic demo with acceleration:")
    print("  python pixel_life.py basic --size 50 --steps 200 --accelerated")
    
    print("\nAI demo with acceleration:")
    print("  python pixel_life.py ai --size 50 --steps 200 --accelerated")
    
    print("\nTraining with acceleration:")
    print("  python pixel_life.py train --timesteps 100000 --device mps --accelerated")
    
    print("\nPerformance benchmark:")
    print("  python pixel_life.py benchmark-demo --size 50")
    
    print("\nDedicated accelerated demo:")
    print("  python pixel_life.py accelerated --size 50 --steps 200")
    
    print("\nCheck available options:")
    print("  python pixel_life.py basic --help")
    print("  python pixel_life.py train --help")

if __name__ == "__main__":
    test_acceleration_options()
    show_usage_examples() 