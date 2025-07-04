"""Optimized Pixel Life Environment with performance enhancements."""

import gym
from gym import spaces
import numpy as np
import numba
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
import random
import time


# Numba-accelerated helper functions
@njit
def find_empty_neighbors_fast(grid, y, x, H, W):
    """Find empty neighboring cells using Numba."""
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] == 0:
            neighbors.append((ny, nx))
    
    return neighbors


@njit
def count_neighbors_fast(grid, y, x, H, W, org_id):
    """Count neighboring cells of same organism using Numba."""
    count = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] == org_id:
            count += 1
    
    return count


@njit
def manhattan_distance(y1, x1, y2, x2):
    """Calculate Manhattan distance."""
    return abs(y1 - y2) + abs(x1 - x2)


@njit(parallel=True)
def compute_distances_batch(coords, origin_y, origin_x):
    """Compute distances for all coordinates in parallel."""
    n = len(coords)
    distances = np.empty(n, dtype=np.int32)
    
    for i in prange(n):
        y, x = coords[i]
        distances[i] = manhattan_distance(y, x, origin_y, origin_x)
    
    return distances


class PixelLifeEnvOptimized(gym.Env):
    """Optimized 2D pixel life environment with performance enhancements."""
    
    def __init__(self, H=50, W=50, max_size=200):
        super().__init__()
        
        # Initial and maximum grid dimensions
        self.initial_H = H
        self.initial_W = W
        self.max_size = max_size
        self.H = H
        self.W = W
        
        # Pre-allocate arrays for better memory performance
        self.max_organisms = 10000
        self.max_pixels = H * W * 4  # Assume max 4x expansion
        
        # Core data structures
        self.grid = None  # int16 for better cache performance
        self.organisms = {}
        self.pixel_to_org = {}
        self.origin = None
        
        # Pre-allocated buffers for fast operations
        self.coords_buffer = np.zeros((self.max_pixels, 2), dtype=np.int32)
        self.distances_buffer = np.zeros(self.max_pixels, dtype=np.int32)
        self.actions_buffer = np.zeros((self.max_pixels, 2), dtype=np.int32)
        
        # Game parameters
        self.params = {
            'split_cost': 0.1,
            'consume_efficiency': 0.8,
            'combine_threshold': 2,
            'forfeit_penalty': 0.5,
            'max_org_size': 100,
            'initial_org_size': 3
        }
        
        # Tweakable parameters for spice
        self.tweakable_params = ['split_cost', 'consume_efficiency', 'combine_threshold']
        self.param_ranges = {
            'split_cost': (0.05, 0.5),
            'consume_efficiency': (0.5, 1.0),
            'combine_threshold': (1, 5)
        }
        
        # Action spaces
        self.pixel_action_types = 5  # no-op, split, consume, combine, forfeit
        self.directions = 4  # up, right, down, left
        self.spice_actions = 3  # no-op, expand, tweak
        
        # Define discrete action space for spice
        self.spice_action_space = spaces.Discrete(self.spice_actions)
        
        # Define action space for main agent (pixel actions)
        self.action_space = spaces.MultiDiscrete([5, 4])  # action_type, direction
        
        # Define observation spaces (simplified for now)
        self.observation_space = spaces.Box(
            low=-1, high=self.max_organisms,
            shape=(self.max_size, self.max_size),
            dtype=np.int16
        )
        
        # Direction vectors
        self.dir_vectors = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=np.int32)
        
        # Game state
        self.next_org_id = 1
        self.tick_count = 0
        self.tweak_timer = 0
        self.tweak_cooldown = 50
        self.expand_timer = 0
        self.expand_cooldown = 100
        
        # Performance monitoring
        self.perf_stats = {
            'reset_time': 0,
            'step_time': 0,
            'action_time': 0,
            'obs_time': 0
        }
    
    def reset(self):
        """Reset environment to initial state."""
        start_time = time.perf_counter()
        
        # Reset dimensions
        self.H = self.initial_H
        self.W = self.initial_W
        
        # Initialize grid with int16 for better cache performance
        self.grid = np.zeros((self.max_size, self.max_size), dtype=np.int16)
        
        # Clear data structures
        self.organisms.clear()
        self.pixel_to_org.clear()
        
        # Reset game state
        self.next_org_id = 1
        self.tick_count = 0
        self.tweak_timer = 0
        self.expand_timer = 0
        
        # Place initial organism at center
        cy, cx = self.H // 2, self.W // 2
        self.origin = (cy, cx)
        
        # Create initial organism efficiently
        org_id = self.next_org_id
        self.next_org_id += 1
        
        # Generate initial cluster
        positions = [(cy, cx)]
        self.grid[cy, cx] = org_id
        self.pixel_to_org[(cy, cx)] = org_id
        
        # Add adjacent cells
        for dy, dx in [(-1, 0), (0, 1), (1, 0)]:
            if self.params['initial_org_size'] > len(positions):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.H and 0 <= nx < self.W:
                    positions.append((ny, nx))
                    self.grid[ny, nx] = org_id
                    self.pixel_to_org[(ny, nx)] = org_id
        
        self.organisms[org_id] = set(positions)
        
        self.perf_stats['reset_time'] = time.perf_counter() - start_time
        
        # Return observations
        return self._get_observations()
    
    def _get_observations(self):
        """Get observations for both agents efficiently."""
        start_time = time.perf_counter()
        
        # Crop grid to active area for main agent
        obs_main = self.grid[:self.H, :self.W].copy()
        
        # Full grid for spice agent  
        obs_spice = self.grid.copy()
        
        self.perf_stats['obs_time'] = time.perf_counter() - start_time
        
        return obs_main, obs_spice
    
    def step(self, spice_action, pixel_actions):
        """Execute one environment tick with optimized operations."""
        start_time = time.perf_counter()
        self.tick_count += 1
        
        # Track previous state for reward calculation
        prev_live_count = len(self.pixel_to_org)
        
        # 1. Apply spice action
        if spice_action == 1 and self.expand_timer <= 0:  # Expand
            direction = random.randint(0, 3)
            self._expand_universe(direction)
            self.expand_timer = self.expand_cooldown
        elif spice_action == 2 and self.tweak_timer <= 0:  # Tweak
            self._apply_tweak()
            self.tweak_timer = self.tweak_cooldown
        
        # Decrement timers
        self.tweak_timer = max(0, self.tweak_timer - 1)
        self.expand_timer = max(0, self.expand_timer - 1)
        
        # 2. Sort pixels by distance using vectorized operations
        action_start = time.perf_counter()
        
        live_pixels = list(self.pixel_to_org.keys())
        n_pixels = len(live_pixels)
        
        if n_pixels > 0:
            # Copy to pre-allocated buffer
            for i, (y, x) in enumerate(live_pixels):
                self.coords_buffer[i, 0] = y
                self.coords_buffer[i, 1] = x
            
            # Compute distances in parallel
            coords_view = self.coords_buffer[:n_pixels]
            distances = compute_distances_batch(coords_view, self.origin[0], self.origin[1])
            
            # Sort by distance
            sorted_indices = np.argsort(distances)
            sorted_pixels = [live_pixels[i] for i in sorted_indices]
            
            # 3. Execute pixel actions
            for y, x in sorted_pixels:
                if (y, x) not in self.pixel_to_org:
                    continue
                
                if (y, x) in pixel_actions:
                    action_type, direction = pixel_actions[(y, x)]
                    
                    if action_type == 1:  # Split
                        self._do_split_optimized(y, x, direction)
                    elif action_type == 2:  # Consume
                        self._do_consume_optimized(y, x, direction)
                    elif action_type == 3:  # Combine
                        self._do_combine_optimized(y, x, direction)
                    elif action_type == 4:  # Forfeit
                        self.do_forfeit(y, x)
        
        self.perf_stats['action_time'] = time.perf_counter() - action_start
        
        # 4. Build observations
        obs_main, obs_spice = self._get_observations()
        
        # 5. Compute rewards
        curr_live_count = len(self.pixel_to_org)
        reward_main = curr_live_count + 1
        
        # Spice reward based on main's performance
        if curr_live_count < prev_live_count:
            reward_spice = 1  # Good for spice
        elif curr_live_count > prev_live_count:
            reward_spice = -1  # Bad for spice
        else:
            reward_spice = 0
        
        # Check if main agent is dead
        done = curr_live_count == 0
        if done:
            reward_spice = 10  # Big reward for killing main
        
        self.perf_stats['step_time'] = time.perf_counter() - start_time
        
        info = {
            'live_pixels': curr_live_count,
            'organisms': len(self.organisms),
            'grid_size': (self.H, self.W),
            'perf_stats': self.perf_stats.copy()
        }
        
        return (obs_main, obs_spice), (reward_main, reward_spice), done, info
    
    def _do_split_optimized(self, y, x, direction):
        """Optimized split operation using Numba functions."""
        org_id = self.pixel_to_org.get((y, x))
        if org_id is None:
            return False
        
        # Calculate new position
        dy, dx = self.dir_vectors[direction]
        ny, nx = y + dy, x + dx
        
        # Check bounds and availability
        if not (0 <= ny < self.H and 0 <= nx < self.W):
            return False
        
        if self.grid[ny, nx] != 0:
            return False
        
        # Apply split cost
        org_cells = self.organisms[org_id]
        if len(org_cells) <= 1:
            return False
        
        cost = int(len(org_cells) * self.params['split_cost'])
        if cost >= len(org_cells):
            return False
        
        # Create new organism efficiently
        new_id = self.next_org_id
        self.next_org_id += 1
        
        self.grid[ny, nx] = new_id
        self.pixel_to_org[(ny, nx)] = new_id
        self.organisms[new_id] = {(ny, nx)}
        
        # Remove cells for cost (vectorized operation)
        if cost > 0:
            cells_list = list(org_cells)
            cells_to_remove = random.sample(cells_list, cost)
            for cy, cx in cells_to_remove:
                self.grid[cy, cx] = -1
                del self.pixel_to_org[(cy, cx)]
                org_cells.remove((cy, cx))
        
        return True
    
    def _do_consume_optimized(self, y, x, direction):
        """Optimized consume operation."""
        org_id = self.pixel_to_org.get((y, x))
        if org_id is None:
            return False
        
        # Calculate target position
        dy, dx = self.dir_vectors[direction]
        ty, tx = y + dy, x + dx
        
        # Check bounds
        if not (0 <= ty < self.H and 0 <= tx < self.W):
            return False
        
        target_val = self.grid[ty, tx]
        
        # Can consume dead cells or smaller organisms
        if target_val == -1:
            # Consume dead cell
            self.grid[ty, tx] = org_id
            self.pixel_to_org[(ty, tx)] = org_id
            self.organisms[org_id].add((ty, tx))
            return True
        
        elif target_val > 0 and target_val != org_id:
            # Check if can consume other organism
            target_size = len(self.organisms.get(target_val, []))
            my_size = len(self.organisms[org_id])
            
            if my_size > target_size:
                # Consume the organism efficiently
                target_cells = self.organisms[target_val]
                efficiency = self.params['consume_efficiency']
                cells_to_convert = int(len(target_cells) * efficiency)
                
                if cells_to_convert > 0:
                    cells_list = list(target_cells)
                    for i in range(min(cells_to_convert, len(cells_list))):
                        cy, cx = cells_list[i]
                        self.grid[cy, cx] = org_id
                        self.pixel_to_org[(cy, cx)] = org_id
                        self.organisms[org_id].add((cy, cx))
                
                # Remove the consumed organism
                del self.organisms[target_val]
                
                # Mark remaining cells as dead
                for i in range(cells_to_convert, len(cells_list)):
                    cy, cx = cells_list[i]
                    self.grid[cy, cx] = -1
                    if (cy, cx) in self.pixel_to_org:
                        del self.pixel_to_org[(cy, cx)]
                
                return True
        
        return False
    
    def _do_combine_optimized(self, y, x, direction):
        """Optimized combine operation."""
        org_id = self.pixel_to_org.get((y, x))
        if org_id is None or org_id not in self.organisms:
            return False
        
        # Find adjacent cells of same organism efficiently
        adjacent_count = count_neighbors_fast(self.grid, y, x, self.H, self.W, org_id)
        
        if adjacent_count >= self.params['combine_threshold']:
            # Merge with adjacent cells to form new organism
            new_id = self.next_org_id
            self.next_org_id += 1
            
            # Convert this cell and adjacent cells
            cells_to_convert = [(y, x)]
            
            for dy, dx in self.dir_vectors:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.H and 0 <= nx < self.W and 
                    self.grid[ny, nx] == org_id):
                    cells_to_convert.append((ny, nx))
            
            # Create new organism
            self.organisms[new_id] = set()
            for cy, cx in cells_to_convert:
                self.grid[cy, cx] = new_id
                self.pixel_to_org[(cy, cx)] = new_id
                self.organisms[new_id].add((cy, cx))
                if (cy, cx) in self.organisms[org_id]:
                    self.organisms[org_id].remove((cy, cx))
            
            # Clean up old organism if empty
            if not self.organisms[org_id]:
                del self.organisms[org_id]
            
            return True
        
        return False
    
    def do_forfeit(self, y, x):
        """Forfeit action - sacrifice pixel."""
        org_id = self.pixel_to_org.get((y, x))
        if org_id is None:
            return False
        
        # Mark as dead
        self.grid[y, x] = -1
        del self.pixel_to_org[(y, x)]
        
        if org_id in self.organisms:
            self.organisms[org_id].discard((y, x))
            if not self.organisms[org_id]:
                del self.organisms[org_id]
        
        return True
    
    def _expand_universe(self, direction):
        """Expand the universe in given direction."""
        if direction == 0 and self.H < self.max_size:  # Up
            self.H += 5
        elif direction == 1 and self.W < self.max_size:  # Right
            self.W += 5
        elif direction == 2 and self.H < self.max_size:  # Down
            # Shift everything down
            new_grid = np.zeros_like(self.grid)
            new_grid[5:, :] = self.grid[:-5, :]
            self.grid = new_grid
            
            # Update coordinates
            new_pixel_to_org = {}
            for (y, x), org_id in self.pixel_to_org.items():
                new_y = y + 5
                new_pixel_to_org[(new_y, x)] = org_id
            self.pixel_to_org = new_pixel_to_org
            
            # Update organisms
            for org_id, cells in self.organisms.items():
                self.organisms[org_id] = {(y + 5, x) for y, x in cells}
            
            # Update origin
            self.origin = (self.origin[0] + 5, self.origin[1])
            self.H += 5
            
        elif direction == 3 and self.W < self.max_size:  # Left
            # Shift everything right
            new_grid = np.zeros_like(self.grid)
            new_grid[:, 5:] = self.grid[:, :-5]
            self.grid = new_grid
            
            # Update coordinates
            new_pixel_to_org = {}
            for (y, x), org_id in self.pixel_to_org.items():
                new_x = x + 5
                new_pixel_to_org[(y, new_x)] = org_id
            self.pixel_to_org = new_pixel_to_org
            
            # Update organisms
            for org_id, cells in self.organisms.items():
                self.organisms[org_id] = {(y, x + 5) for y, x in cells}
            
            # Update origin
            self.origin = (self.origin[0], self.origin[1] + 5)
            self.W += 5
    
    def _apply_tweak(self):
        """Apply random parameter tweak."""
        param = random.choice(self.tweakable_params)
        min_val, max_val = self.param_ranges[param]
        
        # Random walk within bounds
        delta = random.uniform(-0.1, 0.1) * (max_val - min_val)
        new_val = self.params[param] + delta
        self.params[param] = np.clip(new_val, min_val, max_val)
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            # Create color map
            colors = ['white', 'black'] + plt.cm.tab20.colors
            n_colors = len(colors)
            cmap = mcolors.ListedColormap(colors)
            
            # Plot grid
            plt.figure(figsize=(10, 10))
            grid_display = self.grid[:self.H, :self.W].copy()
            
            # Normalize organism IDs for coloring
            for y in range(self.H):
                for x in range(self.W):
                    if grid_display[y, x] > 0:
                        grid_display[y, x] = (grid_display[y, x] % (n_colors - 2)) + 2
            
            plt.imshow(grid_display, cmap=cmap, interpolation='nearest')
            plt.title(f'Pixel Life - Tick {self.tick_count} - Organisms: {len(self.organisms)}')
            plt.axis('off')
            
            # Add grid lines
            for i in range(self.W + 1):
                plt.axvline(x=i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            for i in range(self.H + 1):
                plt.axhline(y=i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            
            # Mark origin
            if self.origin:
                plt.plot(self.origin[1], self.origin[0], 'r*', markersize=15)
            
            plt.tight_layout()
            plt.show()
    
    def close(self):
        """Clean up resources."""
        plt.close('all')