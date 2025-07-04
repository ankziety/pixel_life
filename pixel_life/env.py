import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict


class PixelLifeEnv(gym.Env):
    """
    2D Pixel Life Environment where organisms compete for survival
    against a "spice" agent that can modify the environment.
    """
    
    def __init__(self, H: int = 32, W: int = 32, max_steps: int = 1000, **kwargs):
        super().__init__()
        
        # Grid dimensions
        self.H, self.W = H, W
        self.max_steps = max_steps
        
        # Core data members
        self.grid = np.zeros((H, W), dtype=np.int32)  # 0=empty, >0=org_id, -1=dead
        self.organisms = {}  # dict: org_id -> set of (y,x) coordinates
        self.pixel_to_org = {}  # dict: (y,x) -> org_id
        self.origin = (H // 2, W // 2)  # Initial seed coordinate
        self.next_org_id = 1  # Counter for generating unique organism IDs
        
        # Timing/rule state
        self.step_count = 0
        self.last_tweak_step = 0
        self.prev_total_pixels = 0  # For spice reward calculation
        
        # Tweakable environment parameters
        self.params = {
            # Action costs/benefits
            'split_energy_cost': 1,
            'consume_energy_gain': 2,
            'combine_threshold': 3,
            'forfeit_penalty': -1,
            
            # Rule dynamics
            'tweak_interval': 100,
            'tweak_strength': 0.1,
            
            # Physics
            'expansion_size': 5,
            'max_organisms': 50,
            'energy_decay': 0.99
        }
        self.params.update(kwargs)
        
        # Action spaces
        self.spice_action_space = gym.spaces.Discrete(3)  # [no-op, expand, tweak]
        # Note: pixel_action_space is dynamic based on current live pixels
        
        # Observation spaces  
        self.main_observation_space = gym.spaces.Box(
            low=0, high=255, shape=(H, W, 4), dtype=np.uint8
        )
        self.spice_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
            
        options = options or {}
        
        # Reset core state
        self.grid.fill(0)
        self.organisms.clear()
        self.pixel_to_org.clear()
        self.step_count = 0
        self.last_tweak_step = 0
        self.next_org_id = 1
        
        # Initialize with seed organism
        start_mode = options.get('start_mode', 'single_seed')
        
        if start_mode == 'single_seed':
            self._create_organism([self.origin])
        elif start_mode == 'random_seeds':
            num_seeds = options.get('num_seeds', 3)
            for _ in range(num_seeds):
                y, x = np.random.randint(0, self.H), np.random.randint(0, self.W)
                if self.grid[y, x] == 0:  # Only place on empty cells
                    self._create_organism([(y, x)])
        elif start_mode == 'empty':
            pass  # Start with empty grid
        
        self.prev_total_pixels = self._get_total_live_pixels()
        
        return self._get_observations(), {}
    
    def step(self, spice_action: int, pixel_actions: Dict[Tuple[int, int], int]) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """Execute one environment step."""
        
        # 1. Apply spice action
        self._apply_spice_action(spice_action)
        
        # 2. Sort live pixels by distance from origin and execute actions
        live_pixels = self._get_live_pixels_sorted()
        
        for pixel_coord in live_pixels:
            if pixel_coord in pixel_actions:
                action = pixel_actions[pixel_coord]
                self._execute_pixel_action(pixel_coord, action)
        
        # 3. Build observations
        observations = self._get_observations()
        
        # 4. Compute rewards
        rewards = self._compute_rewards()
        
        # 5. Check termination conditions
        self.step_count += 1
        terminated = (self.step_count >= self.max_steps or 
                     len(self.organisms) == 0)
        truncated = False
        
        # 6. Update state for next step
        self.prev_total_pixels = self._get_total_live_pixels()
        
        info = {
            'total_organisms': len(self.organisms),
            'total_pixels': self._get_total_live_pixels(),
            'step_count': self.step_count
        }
        
        return observations, rewards, terminated, truncated, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the current state."""
        if mode == 'human':
            # Simple text representation
            display_grid = np.where(self.grid == -1, 'X',  # Dead
                                  np.where(self.grid == 0, '.',   # Empty
                                          self.grid.astype(str)))  # Organism IDs
            
            print(f"\nStep {self.step_count}, Organisms: {len(self.organisms)}")
            for row in display_grid:
                print(''.join(f'{cell:>2}' for cell in row))
            print()
            
        elif mode == 'rgb_array':
            # Return visual representation as numpy array
            rgb_array = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            
            # Color mapping: empty=black, dead=red, alive=green shades
            empty_mask = (self.grid == 0)
            dead_mask = (self.grid == -1)
            alive_mask = (self.grid > 0)
            
            rgb_array[empty_mask] = [0, 0, 0]  # Black
            rgb_array[dead_mask] = [255, 0, 0]  # Red
            
            # Different green shades for different organisms
            if np.any(alive_mask):
                unique_orgs = np.unique(self.grid[alive_mask])
                for i, org_id in enumerate(unique_orgs):
                    org_mask = (self.grid == org_id)
                    intensity = 128 + (i * 30) % 128  # Vary green intensity
                    rgb_array[org_mask] = [0, intensity, 0]
            
            return rgb_array
    
    # Core helper methods
    
    def _create_organism(self, pixel_coords: List[Tuple[int, int]]) -> int:
        """Create new organism with given pixel coordinates."""
        org_id = self.next_org_id
        self.next_org_id += 1
        
        self.organisms[org_id] = set(pixel_coords)
        for coord in pixel_coords:
            y, x = coord
            self.grid[y, x] = org_id
            self.pixel_to_org[coord] = org_id
            
        return org_id
    
    def _remove_organism(self, org_id: int) -> None:
        """Remove organism completely."""
        if org_id not in self.organisms:
            return
            
        # Clear from grid and mappings
        for coord in self.organisms[org_id]:
            y, x = coord
            self.grid[y, x] = 0
            if coord in self.pixel_to_org:
                del self.pixel_to_org[coord]
        
        del self.organisms[org_id]
    
    def _kill_pixel(self, coord: Tuple[int, int]) -> None:
        """Mark pixel as dead."""
        y, x = coord
        if coord in self.pixel_to_org:
            org_id = self.pixel_to_org[coord]
            
            # Remove from organism
            if org_id in self.organisms:
                self.organisms[org_id].discard(coord)
                
                # If organism has no living pixels, remove it
                if not self.organisms[org_id]:
                    del self.organisms[org_id]
            
            del self.pixel_to_org[coord]
        
        self.grid[y, x] = -1  # Mark as dead
    
    def _get_live_pixels_sorted(self) -> List[Tuple[int, int]]:
        """Get live pixels sorted by distance from origin."""
        live_pixels = list(self.pixel_to_org.keys())
        
        def distance_from_origin(coord):
            y, x = coord
            oy, ox = self.origin
            return abs(y - oy) + abs(x - ox)  # Manhattan distance
            
        return sorted(live_pixels, key=distance_from_origin)
    
    def _get_total_live_pixels(self) -> int:
        """Get total number of living pixels."""
        return len(self.pixel_to_org)
    
    def _get_neighbors(self, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring coordinates (4-connected)."""
        y, x = coord
        neighbors = []
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.H and 0 <= nx < self.W:
                neighbors.append((ny, nx))
                
        return neighbors
    
    def _apply_spice_action(self, action: int) -> None:
        """Apply spice agent action."""
        if action == 0:  # no-op
            pass
        elif action == 1:  # expand universe
            self._expand_universe()
        elif action == 2:  # apply tweak
            if self.step_count - self.last_tweak_step >= self.params['tweak_interval']:
                self._apply_tweak()
                self.last_tweak_step = self.step_count
    
    def _expand_universe(self) -> None:
        """Expand universe by adding empty space."""
        expansion_size = int(self.params['expansion_size'])
        
        # Choose random direction to expand
        direction = np.random.choice(['up', 'down', 'left', 'right'])
        
        if direction == 'up':
            new_grid = np.zeros((self.H + expansion_size, self.W), dtype=np.int32)
            new_grid[expansion_size:] = self.grid
            self.origin = (self.origin[0] + expansion_size, self.origin[1])
            
            # Update coordinates in data structures
            new_organisms = {}
            new_pixel_to_org = {}
            
            for org_id, coords in self.organisms.items():
                new_coords = set()
                for y, x in coords:
                    new_coord = (y + expansion_size, x)
                    new_coords.add(new_coord)
                    new_pixel_to_org[new_coord] = org_id
                new_organisms[org_id] = new_coords
                
            self.organisms = new_organisms
            self.pixel_to_org = new_pixel_to_org
            self.H += expansion_size
            
        elif direction == 'down':
            new_grid = np.zeros((self.H + expansion_size, self.W), dtype=np.int32)
            new_grid[:self.H] = self.grid
            # Origin stays the same, just grid gets bigger downward
            
            # No coordinate updates needed for downward expansion
            self.H += expansion_size
            
        elif direction == 'left':
            new_grid = np.zeros((self.H, self.W + expansion_size), dtype=np.int32)
            new_grid[:, expansion_size:] = self.grid
            self.origin = (self.origin[0], self.origin[1] + expansion_size)
            
            # Update coordinates in data structures
            new_organisms = {}
            new_pixel_to_org = {}
            
            for org_id, coords in self.organisms.items():
                new_coords = set()
                for y, x in coords:
                    new_coord = (y, x + expansion_size)
                    new_coords.add(new_coord)
                    new_pixel_to_org[new_coord] = org_id
                new_organisms[org_id] = new_coords
                
            self.organisms = new_organisms
            self.pixel_to_org = new_pixel_to_org
            self.W += expansion_size
            
        elif direction == 'right':
            new_grid = np.zeros((self.H, self.W + expansion_size), dtype=np.int32)
            new_grid[:, :self.W] = self.grid
            # Origin stays the same, just grid gets bigger rightward
            
            # No coordinate updates needed for rightward expansion
            self.W += expansion_size
        
        self.grid = new_grid
        
        # Update observation spaces to match new grid size
        self.main_observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 4), dtype=np.uint8
        )
    
    def _apply_tweak(self) -> None:
        """Apply random tweak to environment parameters."""
        # Randomly modify one parameter slightly
        param_keys = list(self.params.keys())
        if param_keys:
            key = np.random.choice(param_keys)
            if isinstance(self.params[key], (int, float)):
                change = np.random.normal(0, self.params['tweak_strength'])
                new_value = max(0, self.params[key] + change)
                
                # Keep certain parameters as integers
                integer_params = {'expansion_size', 'max_organisms', 'combine_threshold', 'tweak_interval'}
                if key in integer_params:
                    new_value = int(round(new_value))
                    
                self.params[key] = new_value
    
    def _execute_pixel_action(self, coord: Tuple[int, int], action: int) -> None:
        """Execute action for a specific pixel."""
        if coord not in self.pixel_to_org:
            return  # Pixel no longer alive
            
        if action == 0:  # split
            self._do_split(coord)
        elif action == 1:  # consume
            self._do_consume(coord)
        elif action == 2:  # combine
            self._do_combine(coord)
        elif action == 3:  # forfeit
            self._do_forfeit(coord)
    
    def _do_split(self, coord: Tuple[int, int]) -> None:
        """Attempt to split pixel to neighboring empty cell."""
        neighbors = self._get_neighbors(coord)
        empty_neighbors = [n for n in neighbors if self.grid[n[0], n[1]] == 0]
        
        if not empty_neighbors:
            return  # No place to split
            
        # Choose random empty neighbor
        target = np.random.choice(len(empty_neighbors))
        new_coord = empty_neighbors[target]
        
        # Get organism ID
        org_id = self.pixel_to_org[coord]
        
        # Add new pixel to organism
        self.organisms[org_id].add(new_coord)
        self.pixel_to_org[new_coord] = org_id
        ny, nx = new_coord
        self.grid[ny, nx] = org_id
    
    def _do_consume(self, coord: Tuple[int, int]) -> None:
        """Consume neighboring dead pixels for energy."""
        neighbors = self._get_neighbors(coord)
        dead_neighbors = [n for n in neighbors if self.grid[n[0], n[1]] == -1]
        
        for dead_coord in dead_neighbors:
            # Clear dead pixel
            dy, dx = dead_coord
            self.grid[dy, dx] = 0
            # TODO: Add energy mechanics if needed
    
    def _do_combine(self, coord: Tuple[int, int]) -> None:
        """Combine with neighboring organisms if conditions met."""
        neighbors = self._get_neighbors(coord)
        org_id = self.pixel_to_org[coord]
        
        for neighbor in neighbors:
            ny, nx = neighbor
            if (neighbor in self.pixel_to_org and 
                self.pixel_to_org[neighbor] != org_id):
                
                other_org_id = self.pixel_to_org[neighbor]
                
                # Check if combination conditions are met
                if (len(self.organisms[org_id]) >= self.params['combine_threshold'] and
                    len(self.organisms[other_org_id]) >= self.params['combine_threshold']):
                    
                    # Merge organisms
                    self._merge_organisms(org_id, other_org_id)
                    return
    
    def _do_forfeit(self, coord: Tuple[int, int]) -> None:
        """Forfeit pixel (mark as dead)."""
        self._kill_pixel(coord)
    
    def _merge_organisms(self, org_id1: int, org_id2: int) -> None:
        """Merge two organisms into one."""
        if org_id1 not in self.organisms or org_id2 not in self.organisms:
            return
            
        # Keep the larger organism, merge smaller into it
        if len(self.organisms[org_id1]) >= len(self.organisms[org_id2]):
            target_id, source_id = org_id1, org_id2
        else:
            target_id, source_id = org_id2, org_id1
        
        # Transfer all pixels
        for coord in self.organisms[source_id]:
            self.organisms[target_id].add(coord)
            self.pixel_to_org[coord] = target_id
            y, x = coord
            self.grid[y, x] = target_id
        
        # Remove source organism
        del self.organisms[source_id]
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Build observations for both agents."""
        main_obs = self._build_main_observation()
        spice_obs = self._build_spice_observation()
        
        return {
            'main': main_obs,
            'spice': spice_obs
        }
    
    def _build_main_observation(self) -> np.ndarray:
        """Build observation for main agent (pixel-level view)."""
        obs = np.zeros((self.H, self.W, 4), dtype=np.uint8)
        
        # Channel 0: Organism IDs (scaled for visibility)
        obs[:, :, 0] = np.clip(self.grid * 10, 0, 255)
        
        # Channel 1: Dead pixels
        obs[:, :, 1] = np.where(self.grid == -1, 255, 0)
        
        # Channel 2: Distance from origin (normalized)
        oy, ox = self.origin
        for y in range(self.H):
            for x in range(self.W):
                dist = abs(y - oy) + abs(x - ox)
                obs[y, x, 2] = min(255, dist * 5)  # Scale distance
        
        # Channel 3: Step count (normalized)
        step_val = min(255, (self.step_count * 255) // self.max_steps)
        obs[:, :, 3].fill(step_val)
        
        return obs
    
    def _build_spice_observation(self) -> np.ndarray:
        """Build observation for spice agent (aggregate statistics)."""
        obs = np.zeros(10, dtype=np.float32)
        
        obs[0] = len(self.organisms)  # Number of organisms
        obs[1] = self._get_total_live_pixels()  # Total living pixels
        obs[2] = np.sum(self.grid == -1)  # Total dead pixels
        obs[3] = self.step_count / self.max_steps  # Progress through episode
        obs[4] = (self.step_count - self.last_tweak_step) / self.params['tweak_interval']  # Time since last tweak
        
        # Organism size statistics
        if self.organisms:
            sizes = [len(coords) for coords in self.organisms.values()]
            obs[5] = np.mean(sizes)  # Average organism size
            obs[6] = np.max(sizes)   # Largest organism size
            obs[7] = np.min(sizes)   # Smallest organism size
        else:
            obs[5:8] = 0
        
        # Spatial distribution (center of mass distance from origin)
        if self.pixel_to_org:
            oy, ox = self.origin
            total_dist = sum(abs(y - oy) + abs(x - ox) for y, x in self.pixel_to_org.keys())
            obs[8] = total_dist / len(self.pixel_to_org)
        else:
            obs[8] = 0
            
        obs[9] = len(self.organisms) / self.params['max_organisms'] if self.params['max_organisms'] > 0 else 0  # Organism density
        
        return obs
    
    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards for both agents."""
        current_pixels = self._get_total_live_pixels()
        
        # Main agent reward: total live pixels + growth bonus
        r_main = current_pixels + 1
        
        # Spice agent reward: competing objective
        if current_pixels < self.prev_total_pixels:
            r_spice = 1.0  # Population decreased
        elif current_pixels > self.prev_total_pixels:
            r_spice = -1.0  # Population increased
        elif len(self.organisms) == 0:
            r_spice = 10.0  # Main agent extinct
        else:
            r_spice = 0.0  # Population stable
        
        return {
            'main': r_main,
            'spice': r_spice
        }
    
    def get_action_space(self, agent: str) -> gym.Space:
        """Get action space for specified agent."""
        if agent == 'spice':
            return self.spice_action_space
        elif agent == 'main':
            # Dynamic action space based on current live pixels
            live_pixels = list(self.pixel_to_org.keys())
            if not live_pixels:
                return gym.spaces.Dict({})
            
            # Return dict space mapping coordinates to actions
            return gym.spaces.Dict({
                f"{y},{x}": gym.spaces.Discrete(4)  # [split, consume, combine, forfeit]
                for y, x in live_pixels
            })
        else:
            raise ValueError(f"Unknown agent: {agent}")
    
    def get_observation_space(self, agent: str) -> gym.Space:
        """Get observation space for specified agent."""
        if agent == 'main':
            return self.main_observation_space
        elif agent == 'spice':
            return self.spice_observation_space
        else:
            raise ValueError(f"Unknown agent: {agent}")