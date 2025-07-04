"""Vectorized environment wrapper for Pixel Life.

This implements efficient parallel environment execution using:
- Batch operations with NumPy
- Minimal Python loops
- Shared memory for fast data transfer
"""

import numpy as np
import gym
from gym import spaces
import multiprocessing as mp
from multiprocessing import shared_memory
import time
from typing import List, Tuple, Dict, Optional
import numba
from numba import njit, prange

from env_optimized import PixelLifeEnvOptimized


class VectorizedPixelLife:
    """Vectorized wrapper that runs multiple PixelLife environments in parallel.
    
    Uses NumPy operations and shared memory for efficient batch processing.
    """
    
    def __init__(self, num_envs: int, H: int = 50, W: int = 50, 
                 max_size: int = 200, use_shared_memory: bool = True):
        """Initialize vectorized environments.
        
        Args:
            num_envs: Number of parallel environments
            H, W: Initial grid dimensions
            max_size: Maximum grid size
            use_shared_memory: Whether to use shared memory for observations
        """
        self.num_envs = num_envs
        self.H = H
        self.W = W
        self.max_size = max_size
        self.use_shared_memory = use_shared_memory
        
        # Create individual environments
        self.envs = [
            PixelLifeEnvOptimized(H=H, W=W, max_size=max_size)
            for _ in range(num_envs)
        ]
        
        # Pre-allocate arrays for batch operations
        self.obs_shape = (max_size, max_size)
        self.batch_obs_main = np.zeros((num_envs, H, W), dtype=np.int16)
        self.batch_obs_spice = np.zeros((num_envs, max_size, max_size), dtype=np.int16)
        self.batch_rewards = np.zeros((num_envs, 2), dtype=np.float32)
        self.batch_dones = np.zeros(num_envs, dtype=bool)
        
        # Shared memory for observations (optional)
        if use_shared_memory:
            self._setup_shared_memory()
        
        # Performance tracking
        self.step_times = []
        self.reset_times = []
    
    def _setup_shared_memory(self):
        """Set up shared memory buffers for observations."""
        # Main agent observations
        self.shm_obs_main = shared_memory.SharedMemory(
            create=True,
            size=self.batch_obs_main.nbytes
        )
        self.shared_obs_main = np.ndarray(
            self.batch_obs_main.shape,
            dtype=self.batch_obs_main.dtype,
            buffer=self.shm_obs_main.buf
        )
        
        # Spice agent observations
        self.shm_obs_spice = shared_memory.SharedMemory(
            create=True,
            size=self.batch_obs_spice.nbytes
        )
        self.shared_obs_spice = np.ndarray(
            self.batch_obs_spice.shape,
            dtype=self.batch_obs_spice.dtype,
            buffer=self.shm_obs_spice.buf
        )
    
    def reset(self, env_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Reset specified environments or all if none specified.
        
        Args:
            env_indices: Indices of environments to reset (None = all)
            
        Returns:
            Tuple of (main_observations, spice_observations)
        """
        start_time = time.perf_counter()
        
        if env_indices is None:
            env_indices = range(self.num_envs)
        
        for i in env_indices:
            obs_main, obs_spice = self.envs[i].reset()
            
            # Copy to batch arrays
            if self.use_shared_memory:
                self.shared_obs_main[i, :self.H, :self.W] = obs_main[:self.H, :self.W]
                self.shared_obs_spice[i] = obs_spice
            else:
                self.batch_obs_main[i, :self.H, :self.W] = obs_main[:self.H, :self.W]
                self.batch_obs_spice[i] = obs_spice
        
        self.reset_times.append(time.perf_counter() - start_time)
        
        if self.use_shared_memory:
            return self.shared_obs_main.copy(), self.shared_obs_spice.copy()
        else:
            return self.batch_obs_main.copy(), self.batch_obs_spice.copy()
    
    def step(self, spice_actions: np.ndarray, 
             pixel_actions: List[Dict[Tuple[int, int], Tuple[int, int]]]) -> Tuple:
        """Execute actions in all environments.
        
        Args:
            spice_actions: Array of spice actions for each environment
            pixel_actions: List of pixel action dictionaries for each environment
            
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        start_time = time.perf_counter()
        
        infos = []
        
        # Execute steps in all environments
        for i in range(self.num_envs):
            obs, rewards, done, info = self.envs[i].step(
                spice_actions[i], 
                pixel_actions[i]
            )
            
            # Update batch arrays
            obs_main, obs_spice = obs
            
            if self.use_shared_memory:
                self.shared_obs_main[i, :self.H, :self.W] = obs_main[:self.H, :self.W]
                self.shared_obs_spice[i] = obs_spice
            else:
                self.batch_obs_main[i, :self.H, :self.W] = obs_main[:self.H, :self.W]
                self.batch_obs_spice[i] = obs_spice
            
            self.batch_rewards[i] = rewards
            self.batch_dones[i] = done
            
            infos.append(info)
            
            # Auto-reset if done
            if done:
                obs_main, obs_spice = self.envs[i].reset()
                if self.use_shared_memory:
                    self.shared_obs_main[i, :self.H, :self.W] = obs_main[:self.H, :self.W]
                    self.shared_obs_spice[i] = obs_spice
                else:
                    self.batch_obs_main[i, :self.H, :self.W] = obs_main[:self.H, :self.W]
                    self.batch_obs_spice[i] = obs_spice
        
        self.step_times.append(time.perf_counter() - start_time)
        
        # Return copies to avoid external modifications
        if self.use_shared_memory:
            obs = (self.shared_obs_main.copy(), self.shared_obs_spice.copy())
        else:
            obs = (self.batch_obs_main.copy(), self.batch_obs_spice.copy())
        
        return obs, self.batch_rewards.copy(), self.batch_dones.copy(), infos
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        stats = {}
        
        if self.step_times:
            stats['avg_step_time'] = np.mean(self.step_times)
            stats['total_steps'] = len(self.step_times)
            stats['steps_per_second'] = self.num_envs / stats['avg_step_time']
        
        if self.reset_times:
            stats['avg_reset_time'] = np.mean(self.reset_times)
            stats['total_resets'] = len(self.reset_times)
        
        # Get individual environment stats
        env_stats = []
        for env in self.envs:
            if hasattr(env, 'perf_stats'):
                env_stats.append(env.perf_stats)
        
        if env_stats:
            for key in env_stats[0].keys():
                values = [s[key] for s in env_stats if key in s]
                if values:
                    stats[f'env_{key}_avg'] = np.mean(values)
        
        return stats
    
    def render(self, env_idx: int = 0, mode: str = 'human'):
        """Render a specific environment."""
        return self.envs[env_idx].render(mode)
    
    def close(self):
        """Clean up resources."""
        for env in self.envs:
            env.close()
        
        if self.use_shared_memory:
            self.shm_obs_main.close()
            self.shm_obs_main.unlink()
            self.shm_obs_spice.close()
            self.shm_obs_spice.unlink()


class AsyncVectorEnv:
    """Asynchronous vectorized environment using subprocesses.
    
    Each environment runs in its own process for true parallelism.
    """
    
    def __init__(self, env_fns: List, start_method: str = 'fork'):
        """Initialize async vector environment.
        
        Args:
            env_fns: List of functions that create environments
            start_method: Multiprocessing start method ('fork', 'spawn', 'forkserver')
        """
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        
        ctx = mp.get_context(start_method)
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[
            ctx.Pipe() for _ in range(self.num_envs)
        ])
        
        # Start worker processes
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            
        # Close worker remotes
        for remote in self.work_remotes:
            remote.close()
        
        # Get observation and action spaces
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset all environments."""
        for remote in self.remotes:
            remote.send(('reset', None))
        
        obs_main_list = []
        obs_spice_list = []
        
        for remote in self.remotes:
            obs_main, obs_spice = remote.recv()
            obs_main_list.append(obs_main)
            obs_spice_list.append(obs_spice)
        
        return np.stack(obs_main_list), np.stack(obs_spice_list)
    
    def step_async(self, actions: Tuple[np.ndarray, List]):
        """Send actions to environments asynchronously."""
        if self.waiting:
            raise RuntimeError('Already waiting for step to complete')
        
        self.waiting = True
        spice_actions, pixel_actions = actions
        
        for remote, spice_action, pixel_action in zip(
            self.remotes, spice_actions, pixel_actions
        ):
            remote.send(('step', (spice_action, pixel_action)))
    
    def step_wait(self) -> Tuple:
        """Wait for step results."""
        if not self.waiting:
            raise RuntimeError('Not waiting for step')
        
        self.waiting = False
        
        results = [remote.recv() for remote in self.remotes]
        
        obs_main_list = []
        obs_spice_list = []
        rewards_list = []
        dones_list = []
        infos_list = []
        
        for (obs_main, obs_spice), rewards, done, info in results:
            obs_main_list.append(obs_main)
            obs_spice_list.append(obs_spice)
            rewards_list.append(rewards)
            dones_list.append(done)
            infos_list.append(info)
        
        obs = (np.stack(obs_main_list), np.stack(obs_spice_list))
        rewards = np.array(rewards_list)
        dones = np.array(dones_list)
        
        return obs, rewards, dones, infos_list
    
    def step(self, actions: Tuple[np.ndarray, List]) -> Tuple:
        """Step all environments synchronously."""
        self.step_async(actions)
        return self.step_wait()
    
    def close(self):
        """Close all environments and processes."""
        if self.closed:
            return
        
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        
        for remote in self.remotes:
            remote.send(('close', None))
        
        for process in self.processes:
            process.join()
        
        self.closed = True


def _worker(remote, parent_remote, env_fn):
    """Worker function for async environments."""
    parent_remote.close()
    env = env_fn()
    
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                spice_action, pixel_actions = data
                obs, rewards, done, info = env.step(spice_action, pixel_actions)
                if done:
                    obs = env.reset()
                remote.send((obs, rewards, done, info))
                
            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs)
                
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
                
            elif cmd == 'close':
                env.close()
                break
                
            else:
                raise NotImplementedError(f'Unknown command: {cmd}')
                
        except EOFError:
            break
    
    remote.close()


def benchmark_vec_envs():
    """Benchmark different vectorization strategies."""
    import matplotlib.pyplot as plt
    
    num_envs_list = [1, 2, 4, 8, 16, 32]
    num_steps = 1000
    
    # Results storage
    results = {
        'sequential': [],
        'vectorized': [],
        'vectorized_shm': [],
        'async': []
    }
    
    for num_envs in num_envs_list:
        print(f"\nBenchmarking with {num_envs} environments...")
        
        # Sequential baseline
        envs = [PixelLifeEnvOptimized(H=30, W=30) for _ in range(num_envs)]
        start = time.perf_counter()
        
        for _ in range(num_steps):
            for env in envs:
                obs = env.reset()
                done = False
                while not done:
                    action = env.spice_action_space.sample()
                    pixel_actions = {}
                    obs, reward, done, info = env.step(action, pixel_actions)
        
        sequential_time = time.perf_counter() - start
        results['sequential'].append(sequential_time)
        print(f"Sequential: {sequential_time:.2f}s")
        
        # Vectorized without shared memory
        vec_env = VectorizedPixelLife(num_envs, H=30, W=30, use_shared_memory=False)
        start = time.perf_counter()
        
        for _ in range(num_steps):
            vec_env.reset()
            done = False
            while not done:
                spice_actions = np.random.randint(0, 3, size=num_envs)
                pixel_actions = [{} for _ in range(num_envs)]
                _, _, dones, _ = vec_env.step(spice_actions, pixel_actions)
                done = dones.all()
        
        vec_time = time.perf_counter() - start
        results['vectorized'].append(vec_time)
        print(f"Vectorized: {vec_time:.2f}s ({sequential_time/vec_time:.2f}x speedup)")
        
        # Vectorized with shared memory
        vec_env_shm = VectorizedPixelLife(num_envs, H=30, W=30, use_shared_memory=True)
        start = time.perf_counter()
        
        for _ in range(num_steps):
            vec_env_shm.reset()
            done = False
            while not done:
                spice_actions = np.random.randint(0, 3, size=num_envs)
                pixel_actions = [{} for _ in range(num_envs)]
                _, _, dones, _ = vec_env_shm.step(spice_actions, pixel_actions)
                done = dones.all()
        
        vec_shm_time = time.perf_counter() - start
        results['vectorized_shm'].append(vec_shm_time)
        print(f"Vectorized (SHM): {vec_shm_time:.2f}s ({sequential_time/vec_shm_time:.2f}x speedup)")
        
        # Async vectorized
        env_fns = [lambda: PixelLifeEnvOptimized(H=30, W=30) for _ in range(num_envs)]
        async_env = AsyncVectorEnv(env_fns, start_method='spawn')
        start = time.perf_counter()
        
        for _ in range(num_steps):
            async_env.reset()
            done = False
            while not done:
                spice_actions = np.random.randint(0, 3, size=num_envs)
                pixel_actions = [{} for _ in range(num_envs)]
                _, _, dones, _ = async_env.step((spice_actions, pixel_actions))
                done = dones.all()
        
        async_time = time.perf_counter() - start
        results['async'].append(async_time)
        print(f"Async: {async_time:.2f}s ({sequential_time/async_time:.2f}x speedup)")
        
        # Cleanup
        for env in envs:
            env.close()
        vec_env.close()
        vec_env_shm.close()
        async_env.close()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    for method, times in results.items():
        speedups = [results['sequential'][i] / times[i] for i in range(len(times))]
        plt.plot(num_envs_list, speedups, 'o-', label=method, linewidth=2, markersize=8)
    
    plt.plot(num_envs_list, num_envs_list, 'k--', label='Ideal speedup', alpha=0.5)
    
    plt.xlabel('Number of Environments')
    plt.ylabel('Speedup vs Sequential')
    plt.title('Vectorized Environment Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pixel_life/vec_env_benchmark.png', dpi=150)
    plt.close()
    
    print("\nBenchmark complete! Results saved to vec_env_benchmark.png")


if __name__ == "__main__":
    # Run benchmark when executed directly
    benchmark_vec_envs()