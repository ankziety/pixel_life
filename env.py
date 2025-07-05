import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
import random


class PixelLifeEnv(gym.Env):
    """2D pixel life environment where individual pixels compete for survival.
    
    Main agent controls individual pixels that can move, consume, and reproduce.
    Spice agent can expand the universe and tweak game parameters to challenge the main agent.
    """
    
    def __init__(self, H=50, W=50, max_size=200):
        super().__init__()
        
        # Initial and maximum grid dimensions
        self.initial_H = H
        self.initial_W = W
        self.max_size = max_size
        self.H = H  # Current height
        self.W = W  # Current width
        
        # Core data structures - simplified for pixel-focused approach
        self.grid = None  # np.array: 0=empty, 1=live pixel, -1=dead
        self.live_pixels = set()  # Set of (y, x) tuples for live pixels
        self.dead_cells = set()  # Track dead cell positions
        self.pixel_ages = {}  # Track how long each pixel has been alive
        
        # Game balance parameters (what spice can tweak)
        self.default_params = {
            'move_cost': 0.1,           # Energy cost for moving
            'consume_gain': 1.5,        # Energy gained from consuming
            'reproduce_cost': 1.5,      # Energy cost for reproduction
            'max_energy': 8.0,          # Maximum energy per pixel
            'energy_decay': 0.02,       # Energy lost per tick
            'min_energy_to_move': 0.3,  # Minimum energy required to move
            'min_energy_to_reproduce': 2.0,  # Minimum energy to reproduce
        }
        self.params = self.default_params.copy()
        self.param_ranges = {
            'move_cost': (0.05, 0.5),
            'consume_gain': (0.5, 2.0),
            'reproduce_cost': (1.0, 5.0),
            'max_energy': (3.0, 10.0),
            'energy_decay': (0.01, 0.2),
            'min_energy_to_move': (0.2, 1.0),
            'min_energy_to_reproduce': (2.0, 6.0),
        }
        
        # State tracking
        self.tick_count = 0
        self.done = False
        self.last_main_reward = 0  # Track for spice reward calculation
        self.pixel_energy = {}  # Energy level for each pixel
        
        # Action spaces
        # Pixel action: (action_type, direction)
        # action_type: 0=no-op, 1=move, 2=consume, 3=reproduce
        # direction: 0=up, 1=right, 2=down, 3=left (ignored for no-op)
        self.pixel_action_space = spaces.MultiDiscrete([4, 4])
        
        # Spice actions: 0=no-op, 1=expand_up, 2=expand_down, 
        #                3=expand_left, 4=expand_right, 5=tweak_rule
        self.spice_action_space = spaces.Discrete(6)
        
        # Main action space (for compatibility with gym interfaces)
        # 20 possible actions: 5 action types * 4 directions
        self.action_space = spaces.Discrete(20)
        
        # Observation space
        # Main agent sees local views, spice sees global view
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=-1, high=1, shape=(H, W), dtype=np.int32),
            'energy': spaces.Box(low=0, high=10, shape=(H, W), dtype=np.float32),
            'params': spaces.Box(low=0, high=10, shape=(len(self.params),), dtype=np.float32),
            'tick': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        })
        
        # Configuration
        self.initial_pixels = 3  # Number of pixels to start with
        self.start_center = True  # Start at center vs random position
        
        # Rendering
        self.fig = None
        self.ax = None
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # Reset dimensions
        self.H = self.initial_H
        self.W = self.initial_W
        
        # Initialize empty grid
        self.grid = np.zeros((self.H, self.W), dtype=np.int32)
        
        # Clear all data structures
        self.live_pixels.clear()
        self.dead_cells.clear()
        self.pixel_ages.clear()
        self.pixel_energy.clear()
        
        # Reset parameters to defaults
        self.params = self.default_params.copy()
        
        # Reset counters
        self.tick_count = 0
        self.done = False
        self.last_main_reward = 0
        
        # Place initial pixels
        if self.start_center:
            start_y, start_x = self.H // 2, self.W // 2
        else:
            # Random position with margin
            margin = 5
            start_y = np.random.randint(margin, self.H - margin)
            start_x = np.random.randint(margin, self.W - margin)
        
        # Create initial pixels in a small cluster
        initial_positions = [
            (start_y, start_x),
            (start_y, start_x + 1),
            (start_y + 1, start_x)
        ]
        
        for y, x in initial_positions:
            if 0 <= y < self.H and 0 <= x < self.W:
                self.live_pixels.add((y, x))
                self.grid[y, x] = 1
                self.pixel_ages[(y, x)] = 0
                self.pixel_energy[(y, x)] = self.params['max_energy'] / 2
        
        # Return initial observations
        obs_main = self._get_main_observation()
        obs_spice = self._get_spice_observation()
        return (obs_main, obs_spice), {}
    
    def step(self, spice_action, pixel_actions):
        """Execute one environment tick.
        
        Args:
            spice_action: int - single action from spice agent
            pixel_actions: dict {(y,x): (action_type, direction)} for each living pixel
            
        Returns:
            observations: (obs_main, obs_spice)
            rewards: (reward_main, reward_spice)
            done: bool
            info: dict with additional information
        """
        self.tick_count += 1
        
        # Store previous state for reward calculation
        prev_live_pixels = len(self.live_pixels)
        
        # 1. Apply spice action
        spice_success = False
        if spice_action == 0:  # No-op
            pass
        elif 1 <= spice_action <= 4:  # Expand universe
            direction = spice_action - 1  # 0=up, 1=right, 2=down, 3=left
            spice_success = self._expand_universe(direction)
        elif spice_action == 5:  # Tweak rule
            tweaked_param = self._apply_tweak()
            spice_success = True
            
        # 2. Apply energy decay to all pixels
        for (y, x) in list(self.live_pixels):
            self.pixel_energy[(y, x)] -= self.params['energy_decay']
            self.pixel_ages[(y, x)] += 1
            
            # Pixel dies if energy reaches zero
            if self.pixel_energy[(y, x)] <= 0:
                self._kill_pixel(y, x)
        
        # 3. Execute each pixel's action
        action_results = {}
        for y, x in list(self.live_pixels):
            if (y, x) not in pixel_actions:
                action_results[(y, x)] = 'no_action'
                continue
                
            action_type, direction = pixel_actions[(y, x)]
            
            if action_type == 0:  # No-op
                action_results[(y, x)] = 'no_op'
            elif action_type == 1:  # Move
                success = self.do_move(y, x, direction)
                action_results[(y, x)] = 'move_success' if success else 'move_fail'
            elif action_type == 2:  # Consume
                success = self.do_consume(y, x, direction)
                action_results[(y, x)] = 'consume_success' if success else 'consume_fail'
            elif action_type == 3:  # Reproduce
                success = self.do_reproduce(y, x, direction)
                action_results[(y, x)] = 'reproduce_success' if success else 'reproduce_fail'
        
        # 4. Build observations
        obs_main = self._get_main_observation()
        obs_spice = self._get_spice_observation()
        
        # 5. Compute rewards
        current_live_pixels = len(self.live_pixels)
        
        # Main reward: survival time + energy efficiency
        survival_bonus = current_live_pixels * 1.0  # Base survival reward
        energy_efficiency = sum(self.pixel_energy.values()) / max(current_live_pixels, 1)
        age_bonus = sum(self.pixel_ages.values()) / max(current_live_pixels, 1) * 0.1
        
        reward_main = survival_bonus + energy_efficiency + age_bonus
        
        # Spice reward based on main agent's performance
        if current_live_pixels == 0:  # Main agent dead
            reward_spice = 10
            self.done = True
        elif current_live_pixels < prev_live_pixels:  # Main agent weakened
            reward_spice = 1
        elif current_live_pixels > prev_live_pixels:  # Main agent grew
            reward_spice = -1
        else:  # No change
            reward_spice = 0
            
        # Store for next step
        self.last_main_reward = reward_main
        
        # 6. Check done conditions
        if not self.live_pixels:  # No pixels left
            self.done = True
        elif self.tick_count > 10000:  # Max episode length
            self.done = True
        elif self.H * self.W > 50000:  # Grid too large
            self.done = True
            
        # Additional info
        info = {
            'tick': self.tick_count,
            'live_pixels': current_live_pixels,
            'dead_pixels': len(self.dead_cells),
            'grid_size': (self.H, self.W),
            'spice_success': spice_success,
            'action_results': action_results,
            'params': self.params.copy(),
            'avg_energy': np.mean(list(self.pixel_energy.values())) if self.pixel_energy else 0,
            'avg_age': np.mean(list(self.pixel_ages.values())) if self.pixel_ages else 0
        }
        
        return (obs_main, obs_spice), (reward_main, reward_spice), self.done, False, info
    
    def render(self, mode='human'):
        """Visualize current state using matplotlib."""
        if mode == 'human':
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(10, 10))
                
            # Create color map: -1=black (dead), 0=white (empty), 1=live pixel
            display_grid = self.grid.copy().astype(float)
            display_grid[display_grid == -1] = -0.5  # Dead cells as dark gray
            
            # Normalize energy for coloring
            if self.live_pixels:
                min_energy = min(self.pixel_energy.values())
                max_energy = max(self.pixel_energy.values())
                if max_energy > min_energy:
                    for (y, x) in self.live_pixels:
                        display_grid[y, x] = (self.pixel_energy[(y, x)] - min_energy) / (max_energy - min_energy)
                else:
                    for (y, x) in self.live_pixels:
                        display_grid[y, x] = 0.5 # Equal energy
            
            self.ax.clear()
            self.ax.imshow(display_grid, cmap='viridis', vmin=-0.5, vmax=1)
            self.ax.set_title(f'Tick: {self.tick_count}, Live Pixels: {len(self.live_pixels)}')
            self.ax.grid(True, alpha=0.3)
            
            # No origin marking needed in pixel-focused approach
            pass
            
            plt.pause(0.01)
            
        return self.grid.copy()
    

    
    def _get_main_observation(self):
        """Build observation for main agent."""
        # For now, return the full grid view
        # TODO: Implement local views per organism
        if self.grid is None:
            self.grid = np.zeros((self.H, self.W), dtype=np.int32)
        return {
            'grid': self.grid.copy(),
            'energy': np.array([self.pixel_energy.get((y, x), 0) for y in range(self.H) for x in range(self.W)], dtype=np.float32).reshape(self.H, self.W),
            'params': np.array(list(self.params.values()), dtype=np.float32),
            'tick': np.array([self.tick_count], dtype=np.int32)
        }
    
    def _get_spice_observation(self):
        """Build observation for spice agent (full grid view)."""
        if self.grid is None:
            self.grid = np.zeros((self.H, self.W), dtype=np.int32)
        return {
            'grid': self.grid.copy(),
            'energy': np.array([self.pixel_energy.get((y, x), 0) for y in range(self.H) for x in range(self.W)], dtype=np.float32).reshape(self.H, self.W),
            'params': np.array(list(self.params.values()), dtype=np.float32),
            'tick': np.array([self.tick_count], dtype=np.int32)
        }
    
    def _get_active_bounds(self, margin=5):
        """Get bounding box of all living cells plus margin."""
        living_coords = list(self.live_pixels)
        if not living_coords:
            return 0, 0, self.H-1, self.W-1
            
        ys, xs = zip(*living_coords)
        min_y = max(0, min(ys) - margin)
        max_y = min(self.H - 1, max(ys) + margin)
        min_x = max(0, min(xs) - margin)
        max_x = min(self.W - 1, max(xs) + margin)
        
        return min_y, min_x, max_y, max_x
    
    # ========== Atomic Actions ==========
    
    def do_move(self, y, x, direction):
        """Move a pixel in the given direction.
        
        Args:
            y, x: Coordinates of the pixel initiating the move
            direction: 0=up, 1=right, 2=down, 3=left
            
        Returns:
            bool: Success of the action
        """
        if (y, x) not in self.live_pixels:
            return False
            
        if self.pixel_energy[(y, x)] < self.params['min_energy_to_move']:
            return False
            
        dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
        new_y, new_x = y + dy, x + dx
        
        if not (0 <= new_y < self.H and 0 <= new_x < self.W):
            return False
        if self.grid[new_y, new_x] != 0:
            return False
            
        # Transfer energy and age
        self.pixel_energy[(new_y, new_x)] = self.pixel_energy[(y, x)]
        self.pixel_ages[(new_y, new_x)] = self.pixel_ages[(y, x)]
        
        # Remove from old position
        self.grid[y, x] = 0
        self.live_pixels.remove((y, x))
        del self.pixel_energy[(y, x)]
        del self.pixel_ages[(y, x)]
        
        # Add to new position
        self.grid[new_y, new_x] = 1
        self.live_pixels.add((new_y, new_x))
        self.pixel_energy[(new_y, new_x)] = self.pixel_energy[(new_y, new_x)] - self.params['move_cost']
        self.pixel_ages[(new_y, new_x)] = 0 # Reset age at new position
        
        return True
    
    def _kill_pixel(self, y, x):
        """Kill a pixel and mark it as dead."""
        if (y, x) in self.live_pixels:
            self.grid[y, x] = -1
            self.dead_cells.add((y, x))
            self.live_pixels.remove((y, x))
            del self.pixel_energy[(y, x)]
            del self.pixel_ages[(y, x)]
    
    def do_consume(self, y, x, direction):
        """Consume dead cells or smaller organisms in the given direction.
        
        Args:
            y, x: Coordinates of the pixel initiating consumption
            direction: 0=up, 1=right, 2=down, 3=left
            
        Returns:
            bool: Success of the action
        """
        if (y, x) not in self.live_pixels:
            return False
            
        energy_before = self.pixel_energy[(y, x)]
        self.pixel_energy[(y, x)] += self.params['consume_gain']
        
        # Find targets within consume range
        dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
        consumed = []
        
        # Check immediate neighbor in direction
        target_y = y + dy
        target_x = x + dx
        
        if (0 <= target_y < self.H and 0 <= target_x < self.W):
            cell_value = self.grid[target_y, target_x]
            
            # Consume dead cell
            if cell_value == -1:
                consumed.append((target_y, target_x))
                self.grid[target_y, target_x] = 1 # Live pixel
                self.live_pixels.add((target_y, target_x))
                self.pixel_energy[(target_y, target_x)] = self.params['max_energy'] / 2
                self.pixel_ages[(target_y, target_x)] = 0
                self.dead_cells.discard((target_y, target_x))
                
            # Consume other live pixels (simplified - just consume individual pixels)
            elif cell_value == 1: # Live pixel
                # Simple pixel-to-pixel consumption
                consumed.append((target_y, target_x))
                # Remove the consumed pixel
                self.live_pixels.remove((target_y, target_x))
                del self.pixel_energy[(target_y, target_x)]
                del self.pixel_ages[(target_y, target_x)]
                self.grid[target_y, target_x] = 0 # Empty space
                
        # Energy change after consumption
        self.pixel_energy[(y, x)] = energy_before + self.params['consume_gain'] - self.params['consume_gain'] * len(consumed) # Subtract energy cost for each consumed pixel
        
        return len(consumed) > 0
    
    def do_reproduce(self, y, x, direction):
        """Reproduce a pixel in the given direction.
        
        Args:
            y, x: Coordinates of the pixel initiating reproduction
            direction: 0=up, 1=right, 2=down, 3=left
            
        Returns:
            bool: Success of the action
        """
        if (y, x) not in self.live_pixels:
            return False
            
        if self.pixel_energy[(y, x)] < self.params['min_energy_to_reproduce']:
            return False
            
        if self.params['reproduce_cost'] > self.pixel_energy[(y, x)]: # Cannot afford reproduction cost
            return False
            
        # Calculate target position for new pixel
        dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
        new_y, new_x = y + dy, x + dx
        
        if not (0 <= new_y < self.H and 0 <= new_x < self.W):
            return False
        if self.grid[new_y, new_x] != 0:
            return False
            
        # Create new pixel at target position
        self.grid[new_y, new_x] = 1
        self.live_pixels.add((new_y, new_x))
        self.pixel_energy[(new_y, new_x)] = self.params['max_energy'] / 2
        self.pixel_ages[(new_y, new_x)] = 0
        
        # Reduce parent's energy
        self.pixel_energy[(y, x)] -= self.params['reproduce_cost']
        
        return True
    
    def do_forfeit(self, y, x):
        """Forfeit action - pixel dies and may affect neighbors.
        
        Args:
            y, x: Coordinates of the forfeiting pixel
            
        Returns:
            bool: Success of the action
        """
        if (y, x) not in self.live_pixels:
            return False
            
        # Mark pixel as dead
        self.grid[y, x] = -1
        self.dead_cells.add((y, x))
        self.live_pixels.remove((y, x))
        del self.pixel_energy[(y, x)]
        del self.pixel_ages[(y, x)]
        
        # Spread death to nearby pixels
        spread_count = 0
        for dy in range(-self.params['forfeit_spread'], self.params['forfeit_spread'] + 1):
            for dx in range(-self.params['forfeit_spread'], self.params['forfeit_spread'] + 1):
                if dy == 0 and dx == 0:
                    continue
                    
                spread_y, spread_x = y + dy, x + dx
                if (0 <= spread_y < self.H and 0 <= spread_x < self.W and
                    (spread_y, spread_x) in self.live_pixels and
                    random.random() < 0.5):  # 50% chance to spread
                    
                    self.grid[spread_y, spread_x] = -1
                    self.dead_cells.add((spread_y, spread_x))
                    self.live_pixels.remove((spread_y, spread_x))
                    del self.pixel_energy[(spread_y, spread_x)]
                    del self.pixel_ages[(spread_y, spread_x)]
                    spread_count += 1
                    
                    if spread_count >= self.params['forfeit_spread']:
                        break
            if spread_count >= self.params['forfeit_spread']:
                break
        
        # No organism cleanup needed in pixel-focused approach
        pass
            
        return True
    
    def _expand_universe(self, direction):
        """Expand the universe in the given direction.
        
        Args:
            direction: 0=up, 1=right, 2=down, 3=left
            
        Returns:
            bool: Success of expansion
        """
        # Check if we've hit max size
        if self.H >= self.max_size or self.W >= self.max_size:
            return False
            
        expansion_size = 5  # Expand by 5 cells at a time
        
        if direction == 0:  # Up
            if self.H + expansion_size > self.max_size:
                return False
            new_grid = np.zeros((self.H + expansion_size, self.W), dtype=np.int32)
            new_grid[expansion_size:, :] = self.grid
            self.grid = new_grid
            # Update coordinates
            self._shift_coordinates(expansion_size, 0)
            self.H += expansion_size
            
        elif direction == 1:  # Right
            if self.W + expansion_size > self.max_size:
                return False
            new_grid = np.zeros((self.H, self.W + expansion_size), dtype=np.int32)
            new_grid[:, :self.W] = self.grid
            self.grid = new_grid
            self.W += expansion_size
            
        elif direction == 2:  # Down
            if self.H + expansion_size > self.max_size:
                return False
            new_grid = np.zeros((self.H + expansion_size, self.W), dtype=np.int32)
            new_grid[:self.H, :] = self.grid
            self.grid = new_grid
            self.H += expansion_size
            
        elif direction == 3:  # Left
            if self.W + expansion_size > self.max_size:
                return False
            new_grid = np.zeros((self.H, self.W + expansion_size), dtype=np.int32)
            new_grid[:, expansion_size:] = self.grid
            self.grid = new_grid
            # Update coordinates
            self._shift_coordinates(0, expansion_size)
            self.W += expansion_size
            
        return True
    
    def _apply_tweak(self):
        """Randomly tweak one game parameter within allowed range.
        
        Returns:
            str: Name of tweaked parameter
        """
        param_name = random.choice(list(self.params.keys()))
        min_val, max_val = self.param_ranges[param_name]
        
        # Random walk: current value +/- 1, clamped to range
        current = self.params[param_name]
        change = random.choice([-1, 1])
        new_val = max(min_val, min(max_val, current + change))
        
        self.params[param_name] = new_val
        return param_name
    
    def _shift_coordinates(self, dy, dx):
        """Shift all coordinates after universe expansion."""
        # Update live pixels
        new_live_pixels = {(y + dy, x + dx) for y, x in self.live_pixels}
        self.live_pixels = new_live_pixels
        
        # Update pixel energy and ages
        new_pixel_energy = {}
        new_pixel_ages = {}
        for (y, x), energy in self.pixel_energy.items():
            new_pixel_energy[(y + dy, x + dx)] = energy
        for (y, x), age in self.pixel_ages.items():
            new_pixel_ages[(y + dy, x + dx)] = age
        self.pixel_energy = new_pixel_energy
        self.pixel_ages = new_pixel_ages
        
        # Update dead cells
        new_dead_cells = {(y + dy, x + dx) for y, x in self.dead_cells}
        self.dead_cells = new_dead_cells
    
