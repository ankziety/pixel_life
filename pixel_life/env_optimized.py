#!/usr/bin/env python3
"""
Optimized PixelLife Environment using Numba JIT compilation and vectorized operations.
Implements guardian best practices for high-performance RL environments.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
import numba
from numba import jit, types
from numba.typed import Dict as NumbaDict, List as NumbaList
import warnings

# Suppress numba warnings for cleaner output
warnings.filterwarnings("ignore", category=numba.NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=numba.NumbaWarning)


@jit(nopython=True, cache=True)
def fast_organism_update(grid: np.ndarray, organism_counts: np.ndarray, 
                        organism_positions: np.ndarray, max_organisms: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated organism state update.
    
    Args:
        grid: 2D grid with organism IDs
        organism_counts: Count of pixels per organism
        organism_positions: Positions of each organism
        max_organisms: Maximum number of organisms
    
    Returns:
        Updated organism_counts and organism_positions
    """
    H, W = grid.shape
    
    # Reset counts
    organism_counts.fill(0)
    
    # Count organisms and track positions
    for y in range(H):
        for x in range(W):
            org_id = grid[y, x]
            if org_id > 0:
                organism_counts[org_id] += 1
                # Store position (simple linear indexing for speed)
                pos_idx = organism_counts[org_id] - 1
                if pos_idx < organism_positions.shape[1]:
                    organism_positions[org_id, pos_idx, 0] = y
                    organism_positions[org_id, pos_idx, 1] = x
    
    return organism_counts, organism_positions


@jit(nopython=True, cache=True)
def fast_action_execution(grid: np.ndarray, action_grid: np.ndarray, 
                         organism_counts: np.ndarray, next_org_id: int,
                         params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Numba-accelerated action execution with vectorized operations.
    
    Args:
        grid: Current grid state
        action_grid: Grid of actions to execute
        organism_counts: Count per organism
        next_org_id: Next available organism ID
        params: Environment parameters [split_threshold, max_age, death_prob]
    
    Returns:
        Updated grid, organism_counts, and next_org_id
    """
    H, W = grid.shape
    new_grid = grid.copy()
    split_threshold, max_age, death_prob = params[0], params[1], params[2]
    
    # Process all actions in parallel-friendly order
    for y in range(H):
        for x in range(W):
            if grid[y, x] <= 0:  # Skip empty/dead cells
                continue
                
            action = action_grid[y, x]
            org_id = grid[y, x]
            
            if action == 0:  # Split
                if organism_counts[org_id] >= split_threshold:
                    # Find empty adjacent cell
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < H and 0 <= nx < W and 
                                new_grid[ny, nx] == 0 and (dy != 0 or dx != 0)):
                                new_grid[ny, nx] = next_org_id
                                organism_counts[next_org_id] = 1
                                next_org_id += 1
                                break
                        else:
                            continue
                        break
            
            elif action == 1:  # Consume (kill adjacent cells)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < H and 0 <= nx < W and 
                            new_grid[ny, nx] > 0 and new_grid[ny, nx] != org_id):
                            killed_org = new_grid[ny, nx]
                            new_grid[ny, nx] = -1  # Mark as dead
                            organism_counts[killed_org] = max(0, organism_counts[killed_org] - 1)
            
            elif action == 3:  # Forfeit (die)
                new_grid[y, x] = -1
                organism_counts[org_id] = max(0, organism_counts[org_id] - 1)
    
    return new_grid, organism_counts, next_org_id


@jit(nopython=True, cache=True)
def fast_distance_sort(positions: np.ndarray, origin: np.ndarray, 
                      grid: np.ndarray) -> np.ndarray:
    """
    Fast distance-based sorting of live pixels from origin.
    
    Args:
        positions: Array of (y, x) positions
        origin: Origin position (y, x)
        grid: Current grid state
    
    Returns:
        Sorted indices by distance from origin
    """
    H, W = grid.shape
    live_positions = []
    
    # Collect live positions
    for y in range(H):
        for x in range(W):
            if grid[y, x] > 0:
                dist = np.sqrt((y - origin[0])**2 + (x - origin[1])**2)
                live_positions.append((dist, y, x))
    
    # Sort by distance (simple bubble sort for numba compatibility)
    n = len(live_positions)
    indices = np.zeros(n, dtype=np.int32)
    
    if n > 0:
        # Convert to arrays for sorting
        distances = np.array([pos[0] for pos in live_positions])
        y_coords = np.array([pos[1] for pos in live_positions])
        x_coords = np.array([pos[2] for pos in live_positions])
        
        # Argsort equivalent
        for i in range(n):
            indices[i] = i
        
        # Bubble sort indices by distance
        for i in range(n):
            for j in range(0, n - i - 1):
                if distances[indices[j]] > distances[indices[j + 1]]:
                    indices[j], indices[j + 1] = indices[j + 1], indices[j]
    
    return indices


class PixelLifeEnvOptimized(gym.Env):
    """
    Highly optimized PixelLife environment using Numba JIT compilation,
    vectorized operations, and memory-efficient data structures.
    """
    
    def __init__(self, H: int = 32, W: int = 32, max_steps: int = 1000, 
                 max_organisms: int = 1000, **kwargs):
        super().__init__()
        
        # Grid dimensions and limits
        self.H, self.W = H, W
        self.max_steps = max_steps
        self.max_organisms = max_organisms
        self.current_step = 0
        
        # Pre-allocate arrays for speed
        self.grid = np.zeros((H, W), dtype=np.int32)
        self.organism_counts = np.zeros(max_organisms, dtype=np.int32)
        self.organism_positions = np.zeros((max_organisms, 100, 2), dtype=np.int32)  # Max 100 pixels per organism
        
        # Action/observation spaces
        self.action_space = gym.spaces.Dict({
            'spice_action': gym.spaces.Discrete(3),  # no-op, expand, tweak
            'pixel_actions': gym.spaces.Box(
                low=0, high=3, shape=(H * W,), dtype=np.int32
            )
        })
        
        self.observation_space = gym.spaces.Dict({
            'main': gym.spaces.Box(0, 255, (H, W, 3), dtype=np.uint8),
            'spice': gym.spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32)
        })
        
        # Optimized parameters (research-backed values)
        self.params = np.array([
            3.0,    # split_threshold  
            50.0,   # max_age
            0.01,   # death_probability
            0.1,    # tweak_strength
            2,      # expansion_size
        ], dtype=np.float32)
        
        # Performance tracking
        self.total_live_pixels = 0
        self.next_org_id = 1
        self.origin = np.array([H // 2, W // 2], dtype=np.int32)
        
        # Vectorized reward computation
        self.reward_history = np.zeros(100, dtype=np.float32)  # Ring buffer
        self.reward_idx = 0
    
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Optimized reset with minimal memory allocation."""
        if seed is not None:
            np.random.seed(seed)
        
        # Fast reset using fill operations
        self.grid.fill(0)
        self.organism_counts.fill(0)
        self.organism_positions.fill(0)
        
        self.current_step = 0
        self.next_org_id = 1
        self.total_live_pixels = 0
        self.reward_history.fill(0.0)
        self.reward_idx = 0
        
        # Initialize with a small seed organism
        seed_size = np.random.randint(2, 5)
        for i in range(seed_size):
            y = self.origin[0] + np.random.randint(-1, 2)
            x = self.origin[1] + np.random.randint(-1, 2)
            y = np.clip(y, 0, self.H - 1)
            x = np.clip(x, 0, self.W - 1)
            if self.grid[y, x] == 0:
                self.grid[y, x] = self.next_org_id
                self.organism_counts[self.next_org_id] += 1
        
        if self.organism_counts[self.next_org_id] > 0:
            self.next_org_id += 1
        
        # Update organism tracking
        self.organism_counts, self.organism_positions = fast_organism_update(
            self.grid, self.organism_counts, self.organism_positions, self.max_organisms
        )
        
        self.total_live_pixels = np.sum(self.organism_counts)
        
        obs = self._get_observation()
        info = {'total_pixels': self.total_live_pixels, 'organisms': self.next_org_id - 1}
        
        return obs, info
    
    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, Dict[str, Any]]:
        """Optimized step function using Numba-accelerated operations."""
        spice_action = int(action['spice_action'])
        
        # Handle dynamic grid sizes after expansion
        expected_size = self.H * self.W
        pixel_actions_array = action['pixel_actions']
        
        if len(pixel_actions_array) != expected_size:
            # Grid has expanded, pad or truncate actions as needed
            if len(pixel_actions_array) < expected_size:
                # Pad with no-op actions (0)
                padded_actions = np.zeros(expected_size, dtype=pixel_actions_array.dtype)
                padded_actions[:len(pixel_actions_array)] = pixel_actions_array
                pixel_actions_array = padded_actions
            else:
                # Truncate to current grid size
                pixel_actions_array = pixel_actions_array[:expected_size]
        
        pixel_actions = pixel_actions_array.reshape(self.H, self.W)
        
        # 1. Execute spice action
        if spice_action == 1:  # expand
            self._expand_universe_fast()
        elif spice_action == 2:  # tweak
            self._apply_tweak_fast()
        
        # 2. Execute pixel actions using fast implementation
        self.grid, self.organism_counts, self.next_org_id = fast_action_execution(
            self.grid, pixel_actions, self.organism_counts, 
            self.next_org_id, self.params
        )
        
        # 3. Update organism tracking
        self.organism_counts, self.organism_positions = fast_organism_update(
            self.grid, self.organism_counts, self.organism_positions, self.max_organisms
        )
        
        # 4. Fast reward computation
        prev_total = self.total_live_pixels
        self.total_live_pixels = np.sum(self.organism_counts)
        
        r_main = float(self.total_live_pixels + 1)
        
        # Spice reward based on main agent performance
        if self.total_live_pixels < prev_total:
            r_spice = 1.0
        elif self.total_live_pixels > prev_total:
            r_spice = -1.0
        elif self.total_live_pixels == 0:
            r_spice = 10.0
        else:
            r_spice = 0.0
        
        # Update reward history (ring buffer)
        self.reward_history[self.reward_idx] = r_main
        self.reward_idx = (self.reward_idx + 1) % len(self.reward_history)
        
        # 5. Check termination
        self.current_step += 1
        terminated = (self.total_live_pixels == 0)
        truncated = (self.current_step >= self.max_steps)
        
        # 6. Build observation and info
        obs = self._get_observation()
        rewards = {'main': r_main, 'spice': r_spice}
        info = {
            'total_pixels': self.total_live_pixels,
            'organisms': self.next_org_id - 1,
            'step': self.current_step,
            'avg_reward': float(np.mean(self.reward_history))
        }
        
        return obs, rewards, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Fast observation generation with minimal allocations."""
        # Main agent observation (3-channel image)
        main_obs = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        
        # Channel 0: Organism IDs (normalized)
        organism_mask = self.grid > 0
        main_obs[:, :, 0] = np.where(organism_mask, 
                                   np.clip(self.grid * 20, 0, 255), 0)
        
        # Channel 1: Live/dead state
        main_obs[:, :, 1] = np.where(self.grid > 0, 255,
                                   np.where(self.grid == -1, 128, 0))
        
        # Channel 2: Distance from origin
        y_coords, x_coords = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
        distances = np.sqrt((y_coords - self.origin[0])**2 + (x_coords - self.origin[1])**2)
        main_obs[:, :, 2] = np.clip(distances * 10, 0, 255).astype(np.uint8)
        
        # Spice agent observation (aggregated statistics)
        spice_obs = np.array([
            self.total_live_pixels / (self.H * self.W),  # Density
            (self.next_org_id - 1) / 100.0,             # Organism count (normalized)
            self.current_step / self.max_steps,         # Progress
            np.mean(self.params[:3]),                   # Avg parameters
            np.std(self.organism_counts[:self.next_org_id]), # Organism size variance
            float(np.sum(self.grid == -1)) / (self.H * self.W), # Death density
            float(np.sum(organism_mask)) / (self.H * self.W),   # Live density
            np.mean(self.reward_history),               # Average reward
            float(np.max(self.organism_counts)),        # Largest organism
            float(self.total_live_pixels > 0)           # Any life indicator
        ], dtype=np.float32)
        
        return {'main': main_obs, 'spice': spice_obs}
    
    def _expand_universe_fast(self) -> None:
        """Fast universe expansion with minimal memory operations."""
        expansion_size = int(self.params[4])
        direction = np.random.randint(4)  # 0=up, 1=down, 2=left, 3=right
        
        if direction == 0:  # Up
            new_grid = np.zeros((self.H + expansion_size, self.W), dtype=np.int32)
            new_grid[expansion_size:] = self.grid
            self.origin[0] += expansion_size
        elif direction == 1:  # Down  
            new_grid = np.zeros((self.H + expansion_size, self.W), dtype=np.int32)
            new_grid[:self.H] = self.grid  # Copy existing grid to top
        elif direction == 2:  # Left
            new_grid = np.zeros((self.H, self.W + expansion_size), dtype=np.int32)
            new_grid[:, expansion_size:] = self.grid
            self.origin[1] += expansion_size
        else:  # Right
            new_grid = np.zeros((self.H, self.W + expansion_size), dtype=np.int32)
            new_grid[:, :self.W] = self.grid  # Copy existing grid to left side
        
        self.grid = new_grid
        self.H, self.W = new_grid.shape
    
    def _apply_tweak_fast(self) -> None:
        """Fast parameter tweaking with bounds checking."""
        param_idx = np.random.randint(len(self.params))
        change = np.random.normal(0, self.params[3])  # tweak_strength
        
        # Apply change with bounds
        old_val = self.params[param_idx]
        new_val = max(0.1, old_val + change)
        
        # Keep integer parameters as integers
        if param_idx in [4]:  # expansion_size
            new_val = int(new_val)
        
        self.params[param_idx] = new_val
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Fast rendering for visualization."""
        if mode == 'rgb_array':
            return self._get_observation()['main']
        elif mode == 'human':
            print(f"Step {self.current_step}: {self.total_live_pixels} pixels, {self.next_org_id-1} organisms")
            return None