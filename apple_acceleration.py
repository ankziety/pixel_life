"""
Apple Silicon Acceleration for Pixel Life

This module provides optimized implementations leveraging Apple Silicon's GPU and NPU
capabilities for faster computation in the Pixel Life environment.
"""

import os
import numpy as np
import torch
from numba import jit, prange
from typing import Dict, Tuple, List, Optional
import platform
import gymnasium as gym

# Check if we're on Apple Silicon
def is_apple_silicon():
    """Check if running on Apple Silicon Mac."""
    return platform.machine() == 'arm64' and platform.system() == 'Darwin'

def get_optimal_device():
    """Get the optimal device for computation on Apple Silicon."""
    if not is_apple_silicon():
        return 'cpu'
    
    if torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

class AppleSiliconAccelerator:
    """Main accelerator class for Apple Silicon optimizations."""
    
    def __init__(self):
        self.device = get_optimal_device()
        self.use_mps = self.device == 'mps'
        self.use_numba = True  # Always use Numba for CPU operations
        
        print(f"🚀 Apple Silicon Accelerator initialized:")
        print(f"   Device: {self.device}")
        print(f"   MPS Available: {torch.backends.mps.is_available()}")
        print(f"   Numba Available: {self.use_numba}")
        
        if self.use_mps:
            print(f"   Using Metal Performance Shaders for GPU acceleration")
    
    def to_device(self, tensor):
        """Move tensor to optimal device."""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        return tensor
    
    def from_device(self, tensor):
        """Move tensor from device to CPU."""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu()
        return tensor

# Global accelerator instance
accelerator = AppleSiliconAccelerator()

@jit(nopython=True, parallel=True)
def accelerated_grid_update(grid, live_pixels, pixel_energy, params):
    """
    Numba-accelerated grid update function.
    
    Args:
        grid: Current grid state
        live_pixels: Set of live pixel coordinates
        pixel_energy: Dictionary of pixel energy levels
        params: Game parameters
    
    Returns:
        Updated grid, new live pixels set, updated energy levels
    """
    H, W = grid.shape
    new_grid = grid.copy()
    new_live_pixels = set()
    new_pixel_energy = {}
    
    # Apply energy decay and check for deaths
    for y, x in live_pixels:
        if 0 <= y < H and 0 <= x < W:
            current_energy = pixel_energy.get((y, x), 0.0)
            new_energy = current_energy - params['energy_decay']
            
            if new_energy > 0:
                new_live_pixels.add((y, x))
                new_pixel_energy[(y, x)] = new_energy
                new_grid[y, x] = 1
            else:
                new_grid[y, x] = -1  # Dead cell
    
    return new_grid, new_live_pixels, new_pixel_energy

@jit(nopython=True)
def accelerated_pixel_movement(y, x, direction, grid, pixel_energy, params):
    """
    Numba-accelerated pixel movement with collision detection.
    
    Args:
        y, x: Current pixel position
        direction: Movement direction (0=up, 1=right, 2=down, 3=left)
        grid: Current grid state
        pixel_energy: Current pixel energy
        params: Game parameters
    
    Returns:
        (new_y, new_x, success, energy_cost)
    """
    H, W = grid.shape
    
    # Calculate new position
    if direction == 0:  # Up
        new_y, new_x = y - 1, x
    elif direction == 1:  # Right
        new_y, new_x = y, x + 1
    elif direction == 2:  # Down
        new_y, new_x = y + 1, x
    else:  # Left
        new_y, new_x = y, x - 1
    
    # Check bounds and collision
    if (0 <= new_y < H and 0 <= new_x < W and 
        grid[new_y, new_x] == 0 and  # Empty space
        pixel_energy >= params['min_energy_to_move']):
        
        energy_cost = params['move_cost']
        return new_y, new_x, True, energy_cost
    
    return y, x, False, 0.0

@jit(nopython=True)
def accelerated_pixel_consume(y, x, direction, grid, pixel_energy, params):
    """
    Numba-accelerated pixel consumption.
    
    Args:
        y, x: Current pixel position
        direction: Consumption direction
        grid: Current grid state
        pixel_energy: Current pixel energy
        params: Game parameters
    
    Returns:
        (success, energy_gain, consumed_y, consumed_x)
    """
    H, W = grid.shape
    
    # Calculate target position
    if direction == 0:  # Up
        target_y, target_x = y - 1, x
    elif direction == 1:  # Right
        target_y, target_x = y, x + 1
    elif direction == 2:  # Down
        target_y, target_x = y + 1, x
    else:  # Left
        target_y, target_x = y, x - 1
    
    # Check if target is valid and contains a live pixel
    if (0 <= target_y < H and 0 <= target_x < W and 
        grid[target_y, target_x] == 1):
        
        energy_gain = params['consume_gain']
        return True, energy_gain, target_y, target_x
    
    return False, 0.0, y, x

@jit(nopython=True)
def accelerated_pixel_reproduce(y, x, direction, grid, pixel_energy, params):
    """
    Numba-accelerated pixel reproduction.
    
    Args:
        y, x: Current pixel position
        direction: Reproduction direction
        grid: Current grid state
        pixel_energy: Current pixel energy
        params: Game parameters
    
    Returns:
        (success, energy_cost, child_y, child_x)
    """
    H, W = grid.shape
    
    if pixel_energy < params['min_energy_to_reproduce']:
        return False, 0.0, y, x
    
    # Calculate child position
    if direction == 0:  # Up
        child_y, child_x = y - 1, x
    elif direction == 1:  # Right
        child_y, child_x = y, x + 1
    elif direction == 2:  # Down
        child_y, child_x = y + 1, x
    else:  # Left
        child_y, child_x = y, x - 1
    
    # Check if child position is valid and empty
    if (0 <= child_y < H and 0 <= child_x < W and 
        grid[child_y, child_x] == 0):
        
        energy_cost = params['reproduce_cost']
        return True, energy_cost, child_y, child_x
    
    return False, 0.0, y, x

class AcceleratedPixelLifeEnv:
    """Accelerated version of PixelLifeEnv using Apple Silicon optimizations."""
    
    def __init__(self, H=50, W=50, max_size=200):
        self.H = H
        self.W = W
        self.max_size = max_size
        
        # Initialize accelerator
        self.accelerator = accelerator
        
        # Game parameters (same as original)
        self.default_params = {
            'move_cost': 0.1,
            'consume_gain': 1.5,
            'reproduce_cost': 1.5,
            'max_energy': 8.0,
            'energy_decay': 0.02,
            'min_energy_to_move': 0.3,
            'min_energy_to_reproduce': 2.0,
        }
        self.params = self.default_params.copy()
        
        # Convert params to numpy array for Numba
        self.params_array = np.array([
            self.params['move_cost'],
            self.params['consume_gain'],
            self.params['reproduce_cost'],
            self.params['max_energy'],
            self.params['energy_decay'],
            self.params['min_energy_to_move'],
            self.params['min_energy_to_reproduce'],
        ], dtype=np.float32)
        
        # State tracking
        self.grid = np.zeros((H, W), dtype=np.int32)
        self.live_pixels = set()
        self.pixel_energy = {}
        self.tick_count = 0
        self.done = False
        
        # Action spaces (for compatibility with original env)
        self.spice_action_space = gym.spaces.Discrete(6)
        self.action_space = gym.spaces.Discrete(20)
        
        # Observation space (for compatibility)
        self.observation_space = gym.spaces.Dict({
            'grid': gym.spaces.Box(low=-1, high=1, shape=(H, W), dtype=np.int32),
            'energy': gym.spaces.Box(low=0, high=10, shape=(H, W), dtype=np.float32),
            'params': gym.spaces.Box(low=0, high=10, shape=(len(self.params),), dtype=np.float32),
            'tick': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        })
        
        # PyTorch tensors for GPU acceleration (if available)
        if self.accelerator.use_mps:
            self.grid_tensor = torch.zeros((H, W), dtype=torch.int32, device='mps')
            self.energy_tensor = torch.zeros((H, W), dtype=torch.float32, device='mps')
    
    def reset(self, seed=None):
        """Reset environment with acceleration."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset grid
        self.grid.fill(0)
        self.live_pixels.clear()
        self.pixel_energy.clear()
        
        # Place initial pixels
        start_y, start_x = self.H // 2, self.W // 2
        initial_positions = [
            (start_y, start_x),
            (start_y, start_x + 1),
            (start_y + 1, start_x)
        ]
        
        for y, x in initial_positions:
            if 0 <= y < self.H and 0 <= x < self.W:
                self.live_pixels.add((y, x))
                self.grid[y, x] = 1
                self.pixel_energy[(y, x)] = self.params['max_energy'] / 2
        
        # Update GPU tensors if using MPS
        if self.accelerator.use_mps:
            self._update_gpu_tensors()
        
        self.tick_count = 0
        self.done = False
        
        return self._get_observation()
    
    def step(self, spice_action, pixel_actions):
        """Execute one environment step with acceleration."""
        self.tick_count += 1
        
        # Apply spice action (simplified for now)
        if spice_action == 5:  # Tweak rule
            self._apply_tweak()
        
        # Apply energy decay using accelerated function
        self.grid, self.live_pixels, self.pixel_energy = accelerated_grid_update(
            self.grid, self.live_pixels, self.pixel_energy, self.params_array
        )
        
        # Execute pixel actions
        for (y, x), (action_type, direction) in pixel_actions.items():
            if (y, x) not in self.live_pixels:
                continue
            
            current_energy = self.pixel_energy.get((y, x), 0.0)
            
            if action_type == 1:  # Move
                new_y, new_x, success, energy_cost = accelerated_pixel_movement(
                    y, x, direction, self.grid, current_energy, self.params_array
                )
                if success:
                    self._move_pixel(y, x, new_y, new_x, energy_cost)
            
            elif action_type == 2:  # Consume
                success, energy_gain, consumed_y, consumed_x = accelerated_pixel_consume(
                    y, x, direction, self.grid, current_energy, self.params_array
                )
                if success:
                    self._consume_pixel(y, x, consumed_y, consumed_x, energy_gain)
            
            elif action_type == 3:  # Reproduce
                success, energy_cost, child_y, child_x = accelerated_pixel_reproduce(
                    y, x, direction, self.grid, current_energy, self.params_array
                )
                if success:
                    self._reproduce_pixel(y, x, child_y, child_x, energy_cost)
        
        # Update GPU tensors if using MPS
        if self.accelerator.use_mps:
            self._update_gpu_tensors()
        
        # Check termination
        if len(self.live_pixels) == 0:
            self.done = True
        
        return self._get_observation(), self._calculate_rewards(), self.done, False, {}
        
        # Execute pixel actions
        for (y, x), (action_type, direction) in pixel_actions.items():
            if (y, x) not in self.live_pixels:
                continue
            
            current_energy = self.pixel_energy.get((y, x), 0.0)
            
            if action_type == 1:  # Move
                new_y, new_x, success, energy_cost = accelerated_pixel_movement(
                    y, x, direction, self.grid, current_energy, self.params_array
                )
                if success:
                    self._move_pixel(y, x, new_y, new_x, energy_cost)
            
            elif action_type == 2:  # Consume
                success, energy_gain, consumed_y, consumed_x = accelerated_pixel_consume(
                    y, x, direction, self.grid, current_energy, self.params_array
                )
                if success:
                    self._consume_pixel(y, x, consumed_y, consumed_x, energy_gain)
            
            elif action_type == 3:  # Reproduce
                success, energy_cost, child_y, child_x = accelerated_pixel_reproduce(
                    y, x, direction, self.grid, current_energy, self.params_array
                )
                if success:
                    self._reproduce_pixel(y, x, child_y, child_x, energy_cost)
        
        # Update GPU tensors if using MPS
        if self.accelerator.use_mps:
            self._update_gpu_tensors()
        
        # Check termination
        if len(self.live_pixels) == 0:
            self.done = True
        
        return self._get_observation(), self._calculate_rewards(), self.done, False, {}
    
    def _move_pixel(self, old_y, old_x, new_y, new_x, energy_cost):
        """Move pixel with energy cost."""
        if (old_y, old_x) in self.pixel_energy:
            current_energy = self.pixel_energy[(old_y, old_x)]
            new_energy = current_energy - energy_cost
            
            if new_energy > 0:
                self.grid[old_y, old_x] = 0
                self.grid[new_y, new_x] = 1
                self.live_pixels.remove((old_y, old_x))
                self.live_pixels.add((new_y, new_x))
                self.pixel_energy[(new_y, new_x)] = new_energy
                del self.pixel_energy[(old_y, old_x)]
    
    def _consume_pixel(self, consumer_y, consumer_x, target_y, target_x, energy_gain):
        """Consume another pixel."""
        if (target_y, target_x) in self.live_pixels:
            # Kill consumed pixel
            self.grid[target_y, target_x] = -1
            self.live_pixels.remove((target_y, target_x))
            if (target_y, target_x) in self.pixel_energy:
                del self.pixel_energy[(target_y, target_x)]
            
            # Give energy to consumer
            if (consumer_y, consumer_x) in self.pixel_energy:
                self.pixel_energy[(consumer_y, consumer_x)] += energy_gain
    
    def _reproduce_pixel(self, parent_y, parent_x, child_y, child_x, energy_cost):
        """Reproduce a new pixel."""
        if (parent_y, parent_x) in self.pixel_energy:
            current_energy = self.pixel_energy[(parent_y, parent_x)]
            new_energy = current_energy - energy_cost
            
            if new_energy > 0:
                # Create child
                self.grid[child_y, child_x] = 1
                self.live_pixels.add((child_y, child_x))
                self.pixel_energy[(child_y, child_x)] = self.params['max_energy'] / 2
                
                # Update parent energy
                self.pixel_energy[(parent_y, parent_x)] = new_energy
    
    def _apply_tweak(self):
        """Apply random parameter tweak."""
        # Define parameter ranges for tweaking
        param_ranges = {
            'move_cost': (0.05, 0.5),
            'consume_gain': (0.5, 2.0),
            'reproduce_cost': (1.0, 5.0),
            'max_energy': (3.0, 10.0),
            'energy_decay': (0.01, 0.2),
            'min_energy_to_move': (0.2, 1.0),
            'min_energy_to_reproduce': (2.0, 6.0),
        }
        
        param_names = list(self.params.keys())
        param_name = np.random.choice(param_names)
        param_range = param_ranges.get(param_name, (0.5, 2.0))
        
        # Random tweak within range
        new_value = np.random.uniform(param_range[0], param_range[1])
        self.params[param_name] = new_value
        
        # Update params array for Numba
        self.params_array = np.array([
            self.params['move_cost'],
            self.params['consume_gain'],
            self.params['reproduce_cost'],
            self.params['max_energy'],
            self.params['energy_decay'],
            self.params['min_energy_to_move'],
            self.params['min_energy_to_reproduce'],
        ], dtype=np.float32)
    
    def _update_gpu_tensors(self):
        """Update GPU tensors for MPS acceleration."""
        if self.accelerator.use_mps:
            self.grid_tensor = torch.from_numpy(self.grid).to('mps')
            
            # Create energy tensor
            energy_array = np.zeros((self.H, self.W), dtype=np.float32)
            for (y, x), energy in self.pixel_energy.items():
                energy_array[y, x] = energy
            self.energy_tensor = torch.from_numpy(energy_array).to('mps')
    
    def _get_observation(self):
        """Get current observation."""
        # Create energy grid
        energy_grid = np.zeros((self.H, self.W), dtype=np.float32)
        for (y, x), energy in self.pixel_energy.items():
            energy_grid[y, x] = energy
        
        # Create params array
        params_array = np.array(list(self.params.values()), dtype=np.float32)
        
        return {
            'grid': self.grid.copy(),
            'energy': energy_grid,
            'params': params_array,
            'tick': np.array([self.tick_count], dtype=np.int32)
        }
    
    def _calculate_rewards(self):
        """Calculate rewards for both agents."""
        main_reward = len(self.live_pixels) * 0.1
        spice_reward = 0.0  # Simplified for now
        
        return main_reward, spice_reward
    
    def render(self, mode='human'):
        """Render the environment (compatibility method)."""
        if mode == 'human':
            # Simple text-based rendering for compatibility
            print(f"Grid size: {self.H}x{self.W}, Live pixels: {len(self.live_pixels)}")
            return None
        return None

def create_accelerated_env(H=50, W=50, max_size=200):
    """Factory function to create accelerated environment."""
    return AcceleratedPixelLifeEnv(H=H, W=W, max_size=max_size)

# Performance benchmarking
def benchmark_acceleration(env_size=50, steps=1000):
    """Benchmark the acceleration improvements."""
    print("🏃‍♂️ Benchmarking Apple Silicon Acceleration...")
    
    # Test regular environment (if available)
    try:
        from env import PixelLifeEnv
        regular_env = PixelLifeEnv(H=env_size, W=env_size)
        
        import time
        start_time = time.time()
        
        obs = regular_env.reset()
        for _ in range(steps):
            spice_action = regular_env.spice_action_space.sample()
            pixel_actions = {}
            for coord in regular_env.live_pixels:
                pixel_actions[coord] = (np.random.randint(0, 4), np.random.randint(0, 4))
            
            obs, rewards, done, truncated, info = regular_env.step(spice_action, pixel_actions)
            if done:
                break
        
        regular_time = time.time() - start_time
        print(f"   Regular environment: {regular_time:.3f}s")
    except ImportError:
        regular_time = None
        print("   Regular environment not available for comparison")
    
    # Test accelerated environment
    accelerated_env = create_accelerated_env(H=env_size, W=env_size)
    
    start_time = time.time()
    
    obs = accelerated_env.reset()
    for _ in range(steps):
        spice_action = 0  # No spice action for simplicity
        pixel_actions = {}
        for coord in accelerated_env.live_pixels:
            pixel_actions[coord] = (np.random.randint(0, 4), np.random.randint(0, 4))
        
        obs, rewards, done, truncated, info = accelerated_env.step(spice_action, pixel_actions)
        if done:
            break
    
    accelerated_time = time.time() - start_time
    print(f"   Accelerated environment: {accelerated_time:.3f}s")
    
    if regular_time is not None:
        speedup = regular_time / accelerated_time
        print(f"   🚀 Speedup: {speedup:.2f}x")
    
    return accelerated_time

if __name__ == "__main__":
    # Run benchmark
    benchmark_acceleration() 