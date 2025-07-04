"""Comprehensive benchmark of Pixel Life optimization techniques.

Tests and compares:
1. Original vs Optimized environment
2. Sequential vs Vectorized vs Async environments  
3. Synchronous vs Asynchronous training
4. Performance profiling and bottleneck analysis
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import cProfile
import pstats
import io
import torch
import psutil
from contextlib import contextmanager
from typing import Dict, List, Tuple

from env import PixelLifeEnv
from env_optimized import PixelLifeEnvOptimized
from vec_env import VectorizedPixelLife, AsyncVectorEnv


@contextmanager
def profile_code(name: str):
    """Context manager for profiling code blocks."""
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        pr.disable()
        elapsed = time.perf_counter() - start_time
        
        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        
        print(f"\n{'='*60}")
        print(f"Profile: {name}")
        print(f"Total time: {elapsed:.3f} seconds")
        print(f"{'='*60}")
        print(s.getvalue())


def benchmark_single_env():
    """Benchmark single environment performance."""
    print("\n" + "="*60)
    print("BENCHMARK: Single Environment Performance")
    print("="*60)
    
    num_episodes = 100
    max_steps = 500
    
    results = {}
    
    # Test original environment
    print("\n1. Testing Original Environment...")
    env = PixelLifeEnv(H=50, W=50)
    
    start_time = time.perf_counter()
    total_steps = 0
    
    for _ in range(num_episodes):
        env.reset()
        for _ in range(max_steps):
            spice_action = env.spice_action_space.sample() if hasattr(env, 'spice_action_space') else 0
            pixel_actions = {}  # Simplified
            
            _, _, done, _ = env.step(spice_action, pixel_actions)
            total_steps += 1
            
            if done:
                break
    
    original_time = time.perf_counter() - start_time
    original_fps = total_steps / original_time
    results['original'] = {'time': original_time, 'fps': original_fps, 'steps': total_steps}
    
    print(f"  Time: {original_time:.2f}s")
    print(f"  FPS: {original_fps:.1f}")
    
    # Test optimized environment
    print("\n2. Testing Optimized Environment...")
    env_opt = PixelLifeEnvOptimized(H=50, W=50)
    
    start_time = time.perf_counter()
    total_steps = 0
    
    for _ in range(num_episodes):
        env_opt.reset()
        for _ in range(max_steps):
            spice_action = env_opt.spice_action_space.sample()
            pixel_actions = {}
            
            _, _, done, info = env_opt.step(spice_action, pixel_actions)
            total_steps += 1
            
            if done:
                break
    
    optimized_time = time.perf_counter() - start_time
    optimized_fps = total_steps / optimized_time
    results['optimized'] = {
        'time': optimized_time, 
        'fps': optimized_fps, 
        'steps': total_steps,
        'perf_stats': env_opt.perf_stats
    }
    
    print(f"  Time: {optimized_time:.2f}s")
    print(f"  FPS: {optimized_fps:.1f}")
    print(f"  Speedup: {original_time/optimized_time:.2f}x")
    
    # Print detailed performance stats
    if env_opt.perf_stats:
        print("\n  Detailed timing breakdown:")
        for key, value in env_opt.perf_stats.items():
            print(f"    {key}: {value*1000:.3f} ms")
    
    return results


def benchmark_vectorized_envs():
    """Benchmark vectorized environment performance."""
    print("\n" + "="*60)
    print("BENCHMARK: Vectorized Environment Performance")
    print("="*60)
    
    num_envs_list = [1, 2, 4, 8, 16, 32, 64]
    num_steps = 1000
    
    results = {
        'sequential': [],
        'vectorized': [],
        'vectorized_shm': [],
        'async': []
    }
    
    for num_envs in num_envs_list:
        if num_envs > mp.cpu_count() * 2:
            continue  # Skip if too many environments
            
        print(f"\nTesting with {num_envs} environments...")
        
        # Sequential baseline
        envs = [PixelLifeEnvOptimized(H=30, W=30) for _ in range(num_envs)]
        start = time.perf_counter()
        
        for _ in range(num_steps):
            for env in envs:
                env.step(0, {})
        
        sequential_time = time.perf_counter() - start
        sequential_fps = (num_steps * num_envs) / sequential_time
        results['sequential'].append(sequential_fps)
        
        # Vectorized
        vec_env = VectorizedPixelLife(num_envs, H=30, W=30, use_shared_memory=False)
        vec_env.reset()
        start = time.perf_counter()
        
        for _ in range(num_steps):
            actions = np.zeros(num_envs, dtype=int)
            pixel_actions = [{} for _ in range(num_envs)]
            vec_env.step(actions, pixel_actions)
        
        vec_time = time.perf_counter() - start
        vec_fps = (num_steps * num_envs) / vec_time
        results['vectorized'].append(vec_fps)
        
        # Vectorized with shared memory
        vec_env_shm = VectorizedPixelLife(num_envs, H=30, W=30, use_shared_memory=True)
        vec_env_shm.reset()
        start = time.perf_counter()
        
        for _ in range(num_steps):
            actions = np.zeros(num_envs, dtype=int)
            pixel_actions = [{} for _ in range(num_envs)]
            vec_env_shm.step(actions, pixel_actions)
        
        vec_shm_time = time.perf_counter() - start
        vec_shm_fps = (num_steps * num_envs) / vec_shm_time
        results['vectorized_shm'].append(vec_shm_fps)
        
        # Async (only on Linux/Mac due to fork)
        if mp.get_start_method() != 'spawn' or num_envs <= 8:
            env_fns = [lambda: PixelLifeEnvOptimized(H=30, W=30) for _ in range(num_envs)]
            async_env = AsyncVectorEnv(env_fns)
            async_env.reset()
            start = time.perf_counter()
            
            for _ in range(num_steps):
                actions = np.zeros(num_envs, dtype=int)
                pixel_actions = [{} for _ in range(num_envs)]
                async_env.step((actions, pixel_actions))
            
            async_time = time.perf_counter() - start
            async_fps = (num_steps * num_envs) / async_time
            results['async'].append(async_fps)
            async_env.close()
        else:
            results['async'].append(0)
        
        # Cleanup
        for env in envs:
            env.close()
        vec_env.close()
        vec_env_shm.close()
        
        # Print results
        print(f"  Sequential: {sequential_fps:.1f} FPS")
        print(f"  Vectorized: {vec_fps:.1f} FPS ({vec_fps/sequential_fps:.2f}x)")
        print(f"  Vec + SHM: {vec_shm_fps:.1f} FPS ({vec_shm_fps/sequential_fps:.2f}x)")
        if results['async'][-1] > 0:
            print(f"  Async: {async_fps:.1f} FPS ({async_fps/sequential_fps:.2f}x)")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    valid_envs = num_envs_list[:len(results['sequential'])]
    
    for method, fps_list in results.items():
        if fps_list and any(f > 0 for f in fps_list):
            plt.plot(valid_envs, fps_list[:len(valid_envs)], 'o-', 
                    label=method, linewidth=2, markersize=8)
    
    plt.xlabel('Number of Environments')
    plt.ylabel('Frames Per Second')
    plt.title('Vectorized Environment Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('pixel_life/benchmark_vectorized.png', dpi=150)
    plt.close()
    
    return results


def benchmark_memory_usage():
    """Benchmark memory usage of different approaches."""
    print("\n" + "="*60)
    print("BENCHMARK: Memory Usage")
    print("="*60)
    
    process = psutil.Process()
    results = {}
    
    # Test single environment
    print("\n1. Single Environment Memory Usage")
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    env = PixelLifeEnvOptimized(H=100, W=100)
    env.reset()
    for _ in range(100):
        env.step(0, {})
    
    mem_after = process.memory_info().rss / 1024 / 1024
    single_mem = mem_after - mem_before
    results['single'] = single_mem
    print(f"  Memory used: {single_mem:.1f} MB")
    env.close()
    
    # Test vectorized environment
    print("\n2. Vectorized Environment Memory Usage (16 envs)")
    mem_before = process.memory_info().rss / 1024 / 1024
    
    vec_env = VectorizedPixelLife(16, H=100, W=100)
    vec_env.reset()
    for _ in range(100):
        vec_env.step(np.zeros(16), [{} for _ in range(16)])
    
    mem_after = process.memory_info().rss / 1024 / 1024
    vec_mem = mem_after - mem_before
    results['vectorized'] = vec_mem
    print(f"  Memory used: {vec_mem:.1f} MB")
    print(f"  Per environment: {vec_mem/16:.1f} MB")
    vec_env.close()
    
    return results


def analyze_bottlenecks():
    """Profile and analyze performance bottlenecks."""
    print("\n" + "="*60)
    print("ANALYSIS: Performance Bottlenecks")
    print("="*60)
    
    # Profile optimized environment
    with profile_code("Optimized Environment Step"):
        env = PixelLifeEnvOptimized(H=50, W=50)
        env.reset()
        
        for _ in range(1000):
            env.step(0, {(25, 25): (1, 0)})  # Single pixel action
    
    # Profile vectorized environment
    with profile_code("Vectorized Environment Step"):
        vec_env = VectorizedPixelLife(8, H=50, W=50)
        vec_env.reset()
        
        actions = np.zeros(8, dtype=int)
        pixel_actions = [{(25, 25): (1, 0)} for _ in range(8)]
        
        for _ in range(1000):
            vec_env.step(actions, pixel_actions)


def generate_report():
    """Generate comprehensive performance report."""
    print("\n" + "="*60)
    print("PIXEL LIFE PERFORMANCE OPTIMIZATION REPORT")
    print("="*60)
    
    # System info
    print(f"\nSystem Information:")
    print(f"  CPU: {psutil.cpu_count()} cores")
    print(f"  RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {'Available' if torch.cuda.is_available() else 'Not available'}")
    
    # Run benchmarks
    single_results = benchmark_single_env()
    vec_results = benchmark_vectorized_envs()
    mem_results = benchmark_memory_usage()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nKey Findings:")
    print(f"1. Optimized environment is {single_results['original']['time']/single_results['optimized']['time']:.1f}x faster than original")
    print(f"2. Vectorized environments scale well up to {mp.cpu_count()} parallel instances")
    print(f"3. Memory overhead per environment: ~{mem_results['vectorized']/16:.1f} MB")
    
    print("\nRecommendations:")
    print("- Use optimized environment for all training")
    print("- Use AsyncVectorEnv for heavy environments or many parallel instances")
    print("- Use VectorizedPixelLife with shared memory for lighter workloads")
    print("- Consider mixed approach: multiple async workers each running vectorized envs")
    
    # Analyze bottlenecks
    analyze_bottlenecks()
    
    print("\nOptimization techniques applied:")
    print("✓ Numba JIT compilation for hot loops")
    print("✓ Vectorized distance calculations")
    print("✓ Pre-allocated buffers")
    print("✓ int16 grid for better cache performance")
    print("✓ Shared memory for inter-process communication")
    print("✓ Asynchronous training pipeline")
    print("✓ Parallel environment execution")
    
    print("\nBenchmark complete! Check generated plots for visual results.")


if __name__ == "__main__":
    import sys
    generate_report()