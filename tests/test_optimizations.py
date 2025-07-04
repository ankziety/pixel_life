"""Test script to verify all optimizations work correctly."""

import numpy as np
import time
import sys

def test_optimized_env():
    """Test optimized environment functionality."""
    print("\n1. Testing Optimized Environment...")
    
    try:
        from env_optimized import PixelLifeEnvOptimized
        
        env = PixelLifeEnvOptimized(H=30, W=30)
        obs = env.reset()
        
        # Test basic functionality
        for _ in range(10):
            spice_action = env.spice_action_space.sample()
            pixel_actions = {(15, 15): (1, 0)}  # Simple action
            obs, rewards, done, info = env.step(spice_action, pixel_actions)
        
        print("✓ Optimized environment works correctly")
        print(f"  Performance stats: {info.get('perf_stats', {})}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Optimized environment failed: {e}")
        return False


def test_vectorized_env():
    """Test vectorized environment."""
    print("\n2. Testing Vectorized Environment...")
    
    try:
        from vec_env import VectorizedPixelLife
        
        # Test without shared memory
        vec_env = VectorizedPixelLife(4, H=20, W=20, use_shared_memory=False)
        obs = vec_env.reset()
        
        for _ in range(5):
            spice_actions = np.random.randint(0, 3, size=4)
            pixel_actions = [{} for _ in range(4)]
            obs, rewards, dones, infos = vec_env.step(spice_actions, pixel_actions)
        
        print("✓ Vectorized environment (no SHM) works correctly")
        
        vec_env.close()
        
        # Test with shared memory
        vec_env_shm = VectorizedPixelLife(4, H=20, W=20, use_shared_memory=True)
        obs = vec_env_shm.reset()
        
        for _ in range(5):
            spice_actions = np.random.randint(0, 3, size=4)
            pixel_actions = [{} for _ in range(4)]
            obs, rewards, dones, infos = vec_env_shm.step(spice_actions, pixel_actions)
        
        print("✓ Vectorized environment (with SHM) works correctly")
        
        vec_env_shm.close()
        return True
        
    except Exception as e:
        print(f"✗ Vectorized environment failed: {e}")
        return False


def test_async_env():
    """Test async vectorized environment."""
    print("\n3. Testing Async Vectorized Environment...")
    
    try:
        from vec_env import AsyncVectorEnv
        from env_optimized import PixelLifeEnvOptimized
        
        env_fns = [lambda: PixelLifeEnvOptimized(H=20, W=20) for _ in range(2)]
        async_env = AsyncVectorEnv(env_fns)
        
        obs = async_env.reset()
        
        for _ in range(5):
            spice_actions = np.random.randint(0, 3, size=2)
            pixel_actions = [{} for _ in range(2)]
            obs, rewards, dones, infos = async_env.step((spice_actions, pixel_actions))
        
        async_env.close()
        
        print("✓ Async vectorized environment works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Async vectorized environment failed: {e}")
        return False


def test_numba_functions():
    """Test Numba-compiled functions."""
    print("\n4. Testing Numba Functions...")
    
    try:
        from env_optimized import find_empty_neighbors_fast, count_neighbors_fast, compute_distances_batch
        
        # Test grid
        grid = np.array([
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ], dtype=np.int16)
        
        # Test find_empty_neighbors_fast
        neighbors = find_empty_neighbors_fast(grid, 1, 1, 3, 3)
        assert len(neighbors) > 0, "Should find empty neighbors"
        
        # Test count_neighbors_fast
        count = count_neighbors_fast(grid, 1, 1, 3, 3, 1)
        assert count == 2, f"Should have 2 neighbors, got {count}"
        
        # Test compute_distances_batch
        coords = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
        distances = compute_distances_batch(coords, 1, 1)
        assert len(distances) == 3, "Should compute 3 distances"
        
        print("✓ Numba functions work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Numba functions failed: {e}")
        return False


def test_performance_comparison():
    """Quick performance comparison."""
    print("\n5. Performance Comparison...")
    
    try:
        from env import PixelLifeEnv
        from env_optimized import PixelLifeEnvOptimized
        
        # Test original
        env_orig = PixelLifeEnv(H=30, W=30)
        env_orig.reset()
        
        start = time.perf_counter()
        for _ in range(100):
            env_orig.step(0, {})
        orig_time = time.perf_counter() - start
        
        # Test optimized
        env_opt = PixelLifeEnvOptimized(H=30, W=30)
        env_opt.reset()
        
        start = time.perf_counter()
        for _ in range(100):
            env_opt.step(0, {})
        opt_time = time.perf_counter() - start
        
        speedup = orig_time / opt_time
        print(f"✓ Optimized environment is {speedup:.1f}x faster")
        
        env_orig.close()
        env_opt.close()
        return True
        
    except Exception as e:
        print(f"✗ Performance comparison failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("PIXEL LIFE OPTIMIZATION TESTS")
    print("="*60)
    
    tests = [
        test_optimized_env,
        test_vectorized_env,
        test_async_env,
        test_numba_functions,
        test_performance_comparison
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Optimizations are working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())