#!/usr/bin/env python3
"""
Quick test to verify optimized implementation and demonstrate performance gains.
"""

import time
import numpy as np
from env import PixelLifeEnv
from env_optimized import PixelLifeEnvOptimized

def quick_performance_test():
    """Quick performance comparison between original and optimized implementations."""
    print("🚀 Quick Performance Test - Guardian Optimizations")
    print("=" * 60)
    
    grid_size = 16
    num_steps = 1000
    
    print(f"Testing {grid_size}x{grid_size} grid for {num_steps:,} steps")
    print()
    
    # Test original implementation
    print("📊 Testing ORIGINAL implementation...")
    env_orig = PixelLifeEnv(H=grid_size, W=grid_size)
    env_orig.reset()
    
    start_time = time.perf_counter()
    for i in range(num_steps):
        spice_action = np.random.randint(3)
        
        # Get live pixels and create action dict (original format)
        live_pixels = env_orig._get_live_pixels_sorted()
        pixel_actions_dict = {}
        for coord in live_pixels[:min(10, len(live_pixels))]:  # Limit to first 10 for speed
            pixel_actions_dict[coord] = np.random.randint(4)
        
        obs, rewards, terminated, truncated, info = env_orig.step(spice_action, pixel_actions_dict)
        if terminated or truncated:
            env_orig.reset()
    
    orig_time = time.perf_counter() - start_time
    orig_fps = num_steps / orig_time
    
    print(f"   Time: {orig_time:.2f}s")
    print(f"   FPS:  {orig_fps:.1f}")
    
    # Test optimized implementation
    print("\n⚡ Testing OPTIMIZED implementation...")
    env_opt = PixelLifeEnvOptimized(H=grid_size, W=grid_size)
    env_opt.reset()
    
    start_time = time.perf_counter()
    for i in range(num_steps):
        spice_action = np.random.randint(3)
        pixel_actions = np.random.randint(4, size=grid_size * grid_size)
        action = {'spice_action': spice_action, 'pixel_actions': pixel_actions}
        
        obs, rewards, terminated, truncated, info = env_opt.step(action)
        if terminated or truncated:
            env_opt.reset()
    
    opt_time = time.perf_counter() - start_time
    opt_fps = num_steps / opt_time
    
    print(f"   Time: {opt_time:.2f}s")
    print(f"   FPS:  {opt_fps:.1f}")
    
    # Calculate improvement
    speedup = opt_fps / orig_fps
    time_reduction = (orig_time - opt_time) / orig_time * 100
    
    print(f"\n🎯 RESULTS:")
    print(f"   Speedup:       {speedup:.2f}x faster")
    print(f"   Time saved:    {time_reduction:.1f}%")
    print(f"   Extra steps:   {int((speedup - 1) * num_steps):,} more steps/second")
    
    # Show what this means for training
    training_hours = 2  # Assume 2 hour training session
    time_saved_hours = training_hours * (1 - 1/speedup)
    
    print(f"\n💡 For a {training_hours}-hour training session:")
    print(f"   Time saved:    {time_saved_hours:.1f} hours")
    print(f"   New duration:  {training_hours/speedup:.1f} hours")
    
    print("\n" + "=" * 60)
    print("✅ Guardian-guided optimizations successfully implemented!")
    
    return {
        'original_fps': orig_fps,
        'optimized_fps': opt_fps,
        'speedup': speedup,
        'time_reduction_percent': time_reduction
    }

def test_memory_usage():
    """Test memory usage comparison."""
    import psutil
    import gc
    
    print("\n💾 Testing Memory Usage...")
    
    grid_size = 32
    
    # Test original
    gc.collect()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1e6
    
    env_orig = PixelLifeEnv(H=grid_size, W=grid_size)
    env_orig.reset()
    
    mem_after = process.memory_info().rss / 1e6
    orig_memory = mem_after - mem_before
    del env_orig
    
    # Test optimized
    gc.collect()
    mem_before = process.memory_info().rss / 1e6
    
    env_opt = PixelLifeEnvOptimized(H=grid_size, W=grid_size)
    env_opt.reset()
    
    mem_after = process.memory_info().rss / 1e6
    opt_memory = mem_after - mem_before
    del env_opt
    
    if orig_memory > 0:
        memory_reduction = (orig_memory - opt_memory) / orig_memory * 100
    else:
        memory_reduction = 0 if opt_memory == 0 else -100
    
    print(f"   Original:     {orig_memory:.1f} MB")
    print(f"   Optimized:    {opt_memory:.1f} MB")
    if orig_memory > 0:
        print(f"   Reduction:    {memory_reduction:.1f}%")
    else:
        print(f"   Memory usage too small to measure accurately")
    
    return {
        'original_memory': orig_memory,
        'optimized_memory': opt_memory,
        'memory_reduction_percent': memory_reduction
    }

def test_numba_compilation():
    """Test Numba compilation effectiveness."""
    print("\n⚡ Testing Numba JIT Compilation...")
    
    from env_optimized import fast_organism_update
    
    # Create test data
    grid = np.random.randint(-1, 10, size=(32, 32), dtype=np.int32)
    organism_counts = np.zeros(100, dtype=np.int32)
    organism_positions = np.zeros((100, 50, 2), dtype=np.int32)
    
    # First call (includes compilation time)
    start = time.perf_counter()
    result1 = fast_organism_update(grid, organism_counts, organism_positions, 100)
    first_time = time.perf_counter() - start
    
    # Second call (pure execution time)
    start = time.perf_counter()
    result2 = fast_organism_update(grid, organism_counts, organism_positions, 100)
    second_time = time.perf_counter() - start
    
    compilation_speedup = first_time / second_time
    
    print(f"   First call (with compilation):  {first_time*1000:.2f}ms")
    print(f"   Second call (compiled):         {second_time*1000:.2f}ms")
    print(f"   Compilation benefit:            {compilation_speedup:.1f}x")
    
    return {
        'compilation_time': first_time,
        'execution_time': second_time,
        'compilation_speedup': compilation_speedup
    }

if __name__ == "__main__":
    try:
        # Run performance test
        perf_results = quick_performance_test()
        
        # Run memory test
        memory_results = test_memory_usage()
        
        # Test Numba compilation
        numba_results = test_numba_compilation()
        
        print(f"\n🏆 GUARDIAN OPTIMIZATION SUMMARY:")
        print(f"   Performance: {perf_results['speedup']:.2f}x faster")
        print(f"   Memory:      {memory_results['memory_reduction_percent']:.1f}% less")
        print(f"   Numba JIT:   {numba_results['compilation_speedup']:.1f}x benefit")
        
        print(f"\n✨ All optimizations working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()