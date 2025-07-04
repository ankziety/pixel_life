import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
import random


class PixelLifeEnv(gym.Env):
    """2D pixel life environment where organisms compete for survival.
    
    Main agent controls pixel organisms that can split, consume, combine, and forfeit.
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
        
        # Core data structures
        self.grid = None  # np.array: 0=empty, >0=org ID, -1=dead
        self.organisms = {}  # {org_id: set of (y, x) tuples}
        self.pixel_to_org = {}  # {(y, x): org_id}
        self.origin = None  # (y, x) of first seed
        self.dead_cells = set()  # Track dead cell positions
        
        # Game balance parameters (what spice can tweak)
        self.default_params = {
            'split_min_size': 3,      # Min organism size to split
            'split_offspring_size': 2,  # Size of new organism after split
            'consume_range': 1,        # How far can consume reach
            'combine_max_distance': 2,  # Max distance between organisms to combine
            'dead_cell_bonus': 1,      # Extra pixels gained from consuming dead cells
            'forfeit_spread': 1,       # How many pixels are lost on forfeit
        }
        self.params = self.default_params.copy()
        self.param_ranges = {
            'split_min_size': (2, 10),
            'split_offspring_size': (1, 5),
            'consume_range': (1, 3),
            'combine_max_distance': (1, 5),
            'dead_cell_bonus': (0, 3),
            'forfeit_spread': (1, 3),
        }
        
        # State tracking
        self.tick_count = 0
        self.next_org_id = 1
        self.done = False
        self.last_main_reward = 0  # Track for spice reward calculation
        
        # Action spaces
        # Pixel action: (action_type, direction)
        # action_type: 0=no-op, 1=split, 2=consume, 3=combine, 4=forfeit
        # direction: 0=up, 1=right, 2=down, 3=left (ignored for no-op/forfeit)
        self.pixel_action_space = spaces.MultiDiscrete([5, 4])
        
        # Spice actions: 0=no-op, 1=expand_up, 2=expand_down, 
        #                3=expand_left, 4=expand_right, 5=tweak_rule
        self.spice_action_space = spaces.Discrete(6)
        
        # Observation space
        # Main agent sees local views, spice sees global view
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=-1, high=10000, shape=(H, W), dtype=np.int32),
            'params': spaces.Box(low=0, high=10, shape=(len(self.params),), dtype=np.float32),
            'tick': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        })
        
        # Configuration
        self.initial_pixels = 3  # Number of pixels in starting organism
        self.start_center = True  # Start at center vs random position
        
        # Rendering
        self.fig = None
        self.ax = None
        
    def reset(self):
        """Reset environment to initial state."""
        # Reset dimensions
        self.H = self.initial_H
        self.W = self.initial_W
        
        # Initialize empty grid
        self.grid = np.zeros((self.H, self.W), dtype=np.int32)
        
        # Clear all data structures
        self.organisms.clear()
        self.pixel_to_org.clear()
        self.dead_cells.clear()
        
        # Reset parameters to defaults
        self.params = self.default_params.copy()
        
        # Reset counters
        self.tick_count = 0
        self.next_org_id = 1
        self.done = False
        self.last_main_reward = 0
        
        # Place initial organism
        if self.start_center:
            start_y, start_x = self.H // 2, self.W // 2
        else:
            # Random position with margin
            margin = 5
            start_y = np.random.randint(margin, self.H - margin)
            start_x = np.random.randint(margin, self.W - margin)
        
        self.origin = (start_y, start_x)
        
        # Create L-shaped organism (common pattern for 3 pixels)
        org_id = self._create_organism([(start_y, start_x), 
                                       (start_y, start_x + 1), 
                                       (start_y + 1, start_x)])
        
        # Return initial observations
        obs_main = self._get_main_observation()
        obs_spice = self._get_spice_observation()
        
        return (obs_main, obs_spice)
    
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
        prev_live_pixels = len(self.pixel_to_org)
        
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
            
        # 2. Sort live pixels by distance from origin
        live_pixels = list(self.pixel_to_org.keys())
        if self.origin and live_pixels:
            # Manhattan distance from origin
            live_pixels.sort(key=lambda p: abs(p[0] - self.origin[0]) + abs(p[1] - self.origin[1]))
        
        # 3. Execute each pixel's action
        action_results = {}
        for y, x in live_pixels:
            if (y, x) not in self.pixel_to_org:  # Pixel might have been consumed
                continue
                
            if (y, x) not in pixel_actions:
                action_results[(y, x)] = 'no_action'
                continue
                
            action_type, direction = pixel_actions[(y, x)]
            
            if action_type == 0:  # No-op
                action_results[(y, x)] = 'no_op'
            elif action_type == 1:  # Split
                success = self.do_split(y, x, direction)
                action_results[(y, x)] = 'split_success' if success else 'split_fail'
            elif action_type == 2:  # Consume
                success = self.do_consume(y, x, direction)
                action_results[(y, x)] = 'consume_success' if success else 'consume_fail'
            elif action_type == 3:  # Combine
                success = self.do_combine(y, x, direction)
                action_results[(y, x)] = 'combine_success' if success else 'combine_fail'
            elif action_type == 4:  # Forfeit
                success = self.do_forfeit(y, x)
                action_results[(y, x)] = 'forfeit'
        
        # 4. Build observations
        obs_main = self._get_main_observation()
        obs_spice = self._get_spice_observation()
        
        # 5. Compute rewards
        current_live_pixels = len(self.pixel_to_org)
        reward_main = current_live_pixels + 1  # +1 baseline reward
        
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
        if not self.organisms:  # No organisms left
            self.done = True
        elif self.tick_count > 10000:  # Max episode length
            self.done = True
        elif self.H * self.W > 50000:  # Grid too large
            self.done = True
            
        # Additional info
        info = {
            'tick': self.tick_count,
            'organisms': len(self.organisms),
            'live_pixels': current_live_pixels,
            'dead_pixels': len(self.dead_cells),
            'grid_size': (self.H, self.W),
            'spice_success': spice_success,
            'action_results': action_results,
            'params': self.params.copy()
        }
        
        return (obs_main, obs_spice), (reward_main, reward_spice), self.done, info
    
    def render(self, mode='human'):
        """Visualize current state using matplotlib."""
        if mode == 'human':
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(10, 10))
                
            # Create color map: -1=black (dead), 0=white (empty), >0=colors (organisms)
            display_grid = self.grid.copy().astype(float)
            display_grid[display_grid == -1] = -0.5  # Dead cells as dark gray
            
            # Normalize organism IDs for coloring
            mask = display_grid > 0
            if mask.any():
                max_id = display_grid[mask].max()
                display_grid[mask] = display_grid[mask] / max_id
            
            self.ax.clear()
            self.ax.imshow(display_grid, cmap='viridis', vmin=-0.5, vmax=1)
            self.ax.set_title(f'Tick: {self.tick_count}, Organisms: {len(self.organisms)}')
            self.ax.grid(True, alpha=0.3)
            
            # Mark origin
            if self.origin:
                self.ax.plot(self.origin[1], self.origin[0], 'r*', markersize=15)
            
            plt.pause(0.01)
            
        return self.grid.copy()
    
    def _create_organism(self, pixels):
        """Create a new organism from a list of pixel coordinates."""
        org_id = self.next_org_id
        self.next_org_id += 1
        
        self.organisms[org_id] = set(pixels)
        for y, x in pixels:
            self.grid[y, x] = org_id
            self.pixel_to_org[(y, x)] = org_id
            
        return org_id
    
    def _get_main_observation(self):
        """Build observation for main agent."""
        # For now, return the full grid view
        # TODO: Implement local views per organism
        return {
            'grid': self.grid.copy(),
            'params': np.array(list(self.params.values()), dtype=np.float32),
            'tick': np.array([self.tick_count], dtype=np.int32)
        }
    
    def _get_spice_observation(self):
        """Build observation for spice agent (full grid view)."""
        return {
            'grid': self.grid.copy(),
            'params': np.array(list(self.params.values()), dtype=np.float32),
            'tick': np.array([self.tick_count], dtype=np.int32)
        }
    
    def _get_active_bounds(self, margin=5):
        """Get bounding box of all living cells plus margin."""
        living_coords = list(self.pixel_to_org.keys())
        if not living_coords:
            return 0, 0, self.H-1, self.W-1
            
        ys, xs = zip(*living_coords)
        min_y = max(0, min(ys) - margin)
        max_y = min(self.H - 1, max(ys) + margin)
        min_x = max(0, min(xs) - margin)
        max_x = min(self.W - 1, max(xs) + margin)
        
        return min_y, min_x, max_y, max_x
    
    # ========== Atomic Actions ==========
    
    def do_split(self, y, x, direction):
        """Split an organism by creating a new offspring in the given direction.
        
        Args:
            y, x: Coordinates of the pixel initiating the split
            direction: 0=up, 1=right, 2=down, 3=left
            
        Returns:
            bool: Success of the action
        """
        if (y, x) not in self.pixel_to_org:
            return False
            
        org_id = self.pixel_to_org[(y, x)]
        org_pixels = self.organisms[org_id]
        
        # Check minimum size requirement
        if len(org_pixels) < self.params['split_min_size']:
            return False
            
        # Calculate target position for new organism
        dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
        new_y, new_x = y + dy, x + dx
        
        # Check if target position is valid and empty
        if not (0 <= new_y < self.H and 0 <= new_x < self.W):
            return False
        if self.grid[new_y, new_x] != 0:
            return False
            
        # Find pixels to transfer to offspring (BFS from split point)
        offspring_pixels = set()
        to_check = [(y, x)]
        checked = set()
        
        while to_check and len(offspring_pixels) < self.params['split_offspring_size']:
            curr_y, curr_x = to_check.pop(0)
            if (curr_y, curr_x) in checked:
                continue
            checked.add((curr_y, curr_x))
            
            if (curr_y, curr_x) in org_pixels:
                offspring_pixels.add((curr_y, curr_x))
                # Add neighbors
                for dy2, dx2 in [(-1,0), (0,1), (1,0), (0,-1)]:
                    ny, nx = curr_y + dy2, curr_x + dx2
                    if (ny, nx) not in checked:
                        to_check.append((ny, nx))
        
        # Ensure parent keeps connected component
        if len(org_pixels) - len(offspring_pixels) < 1:
            return False
            
        # Create new organism
        new_pixels = list(offspring_pixels) + [(new_y, new_x)]
        new_org_id = self._create_organism(new_pixels)
        
        # Remove transferred pixels from parent
        for py, px in offspring_pixels:
            org_pixels.remove((py, px))
            del self.pixel_to_org[(py, px)]
            self.grid[py, px] = 0
            
        # Re-assign remaining parent pixels
        for py, px in org_pixels:
            self.grid[py, px] = org_id
            self.pixel_to_org[(py, px)] = org_id
            
        return True
    
    def do_consume(self, y, x, direction):
        """Consume dead cells or smaller organisms in the given direction.
        
        Args:
            y, x: Coordinates of the pixel initiating consumption
            direction: 0=up, 1=right, 2=down, 3=left
            
        Returns:
            bool: Success of the action
        """
        if (y, x) not in self.pixel_to_org:
            return False
            
        org_id = self.pixel_to_org[(y, x)]
        org_pixels = self.organisms[org_id]
        
        # Find targets within consume range
        dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
        consumed = []
        
        for dist in range(1, self.params['consume_range'] + 1):
            target_y = y + dy * dist
            target_x = x + dx * dist
            
            if not (0 <= target_y < self.H and 0 <= target_x < self.W):
                break
                
            cell_value = self.grid[target_y, target_x]
            
            # Consume dead cell
            if cell_value == -1:
                consumed.append((target_y, target_x))
                self.grid[target_y, target_x] = org_id
                self.pixel_to_org[(target_y, target_x)] = org_id
                org_pixels.add((target_y, target_x))
                self.dead_cells.discard((target_y, target_x))
                
                # Bonus pixels for dead cell consumption
                for _ in range(self.params['dead_cell_bonus']):
                    # Try to expand in random direction
                    for attempt in range(4):
                        rand_dir = random.randint(0, 3)
                        dy2, dx2 = [(-1, 0), (0, 1), (1, 0), (0, -1)][rand_dir]
                        bonus_y = target_y + dy2
                        bonus_x = target_x + dx2
                        if (0 <= bonus_y < self.H and 0 <= bonus_x < self.W and 
                            self.grid[bonus_y, bonus_x] == 0):
                            self.grid[bonus_y, bonus_x] = org_id
                            self.pixel_to_org[(bonus_y, bonus_x)] = org_id
                            org_pixels.add((bonus_y, bonus_x))
                            break
                            
            # Consume smaller organism
            elif cell_value > 0 and cell_value != org_id:
                target_org_pixels = self.organisms.get(cell_value, set())
                if len(target_org_pixels) < len(org_pixels):
                    # Consume entire organism
                    for ty, tx in list(target_org_pixels):
                        self.grid[ty, tx] = org_id
                        self.pixel_to_org[(ty, tx)] = org_id
                        org_pixels.add((ty, tx))
                    if cell_value in self.organisms:
                        del self.organisms[cell_value]
                    consumed.extend(target_org_pixels)
                    
            if consumed:
                break  # Only consume first valid target
                
        return len(consumed) > 0
    
    def do_combine(self, y, x, direction):
        """Combine with nearby organisms within range.
        
        Args:
            y, x: Coordinates of the pixel initiating combination
            direction: Direction to look for organisms to combine with
            
        Returns:
            bool: Success of the action
        """
        if (y, x) not in self.pixel_to_org:
            return False
            
        org_id = self.pixel_to_org[(y, x)]
        
        # Find nearby organisms in the given direction
        dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
        found_orgs = set()
        
        # Search in cone pattern in the given direction
        for dist in range(1, self.params['combine_max_distance'] + 1):
            for offset in range(-dist//2, dist//2 + 1):
                if direction in [0, 2]:  # up/down
                    check_y = y + dy * dist
                    check_x = x + offset
                else:  # left/right
                    check_y = y + offset
                    check_x = x + dx * dist
                    
                if (0 <= check_y < self.H and 0 <= check_x < self.W):
                    cell_org = self.grid[check_y, check_x]
                    if cell_org > 0 and cell_org != org_id:
                        found_orgs.add(cell_org)
        
        if not found_orgs:
            return False
            
        # Combine with all found organisms
        combined_pixels = set(self.organisms[org_id])
        for other_id in found_orgs:
            if other_id in self.organisms:
                other_pixels = self.organisms[other_id]
                # Transfer all pixels to main organism
                for py, px in other_pixels:
                    self.grid[py, px] = org_id
                    self.pixel_to_org[(py, px)] = org_id
                    combined_pixels.add((py, px))
                del self.organisms[other_id]
                
        self.organisms[org_id] = combined_pixels
        return True
    
    def do_forfeit(self, y, x):
        """Forfeit action - pixel dies and may affect neighbors.
        
        Args:
            y, x: Coordinates of the forfeiting pixel
            
        Returns:
            bool: Success of the action
        """
        if (y, x) not in self.pixel_to_org:
            return False
            
        org_id = self.pixel_to_org[(y, x)]
        org_pixels = self.organisms[org_id]
        
        # Mark pixel as dead
        self.grid[y, x] = -1
        self.dead_cells.add((y, x))
        org_pixels.remove((y, x))
        del self.pixel_to_org[(y, x)]
        
        # Spread death to nearby pixels
        spread_count = 0
        for dy in range(-self.params['forfeit_spread'], self.params['forfeit_spread'] + 1):
            for dx in range(-self.params['forfeit_spread'], self.params['forfeit_spread'] + 1):
                if dy == 0 and dx == 0:
                    continue
                    
                spread_y, spread_x = y + dy, x + dx
                if (0 <= spread_y < self.H and 0 <= spread_x < self.W and
                    (spread_y, spread_x) in org_pixels and
                    random.random() < 0.5):  # 50% chance to spread
                    
                    self.grid[spread_y, spread_x] = -1
                    self.dead_cells.add((spread_y, spread_x))
                    org_pixels.remove((spread_y, spread_x))
                    del self.pixel_to_org[(spread_y, spread_x)]
                    spread_count += 1
                    
                    if spread_count >= self.params['forfeit_spread']:
                        break
            if spread_count >= self.params['forfeit_spread']:
                break
        
        # Clean up organism if empty
        if not org_pixels:
            del self.organisms[org_id]
        else:
            # Check if organism split into multiple components
            self._check_organism_connectivity(org_id)
            
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
        # Update organisms
        new_organisms = {}
        for org_id, pixels in self.organisms.items():
            new_pixels = {(y + dy, x + dx) for y, x in pixels}
            new_organisms[org_id] = new_pixels
        self.organisms = new_organisms
        
        # Update pixel_to_org
        new_pixel_to_org = {}
        for (y, x), org_id in self.pixel_to_org.items():
            new_pixel_to_org[(y + dy, x + dx)] = org_id
        self.pixel_to_org = new_pixel_to_org
        
        # Update dead cells
        new_dead_cells = {(y + dy, x + dx) for y, x in self.dead_cells}
        self.dead_cells = new_dead_cells
        
        # Update origin
        if self.origin:
            self.origin = (self.origin[0] + dy, self.origin[1] + dx)
    
    def _check_organism_connectivity(self, org_id):
        """Check if organism is still connected, split if not."""
        if org_id not in self.organisms:
            return
            
        pixels = self.organisms[org_id]
        if not pixels:
            return
            
        # Find connected components using BFS
        unvisited = set(pixels)
        components = []
        
        while unvisited:
            start = unvisited.pop()
            component = set()
            queue = [start]
            
            while queue:
                y, x = queue.pop(0)
                if (y, x) in component:
                    continue
                    
                component.add((y, x))
                unvisited.discard((y, x))
                
                # Check neighbors
                for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in unvisited:
                        queue.append((ny, nx))
                        
            components.append(component)
        
        # If disconnected, keep largest component with original ID
        if len(components) > 1:
            components.sort(key=len, reverse=True)
            # Keep largest with original ID
            self.organisms[org_id] = components[0]
            
            # Create new organisms for other components
            for component in components[1:]:
                new_org_id = self._create_organism(list(component))