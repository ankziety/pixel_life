import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Tuple, Optional, Any, Dict


class PixelLifeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, grid_size: int = 32, max_steps: int = 1000, render_mode: Optional[str] = None):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Action space: 5 actions (up, down, left, right, place_cell)
        self.action_space = spaces.Discrete(5)
        
        # Observation space: grid + agent position + step count
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(grid_size, grid_size, 3),  # 3 channels: cells, agent, age
            dtype=np.float32
        )
        
        # Initialize pygame for rendering
        self.window_size = 512
        self.window = None
        self.clock = None
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset environment state
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        self.step_count = 0
        self.cell_age = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place some initial random cells
        initial_cells = np.random.choice(2, size=(self.grid_size, self.grid_size), p=[0.85, 0.15])
        self.grid = initial_cells.astype(np.int32)
        
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        
        # Execute agent action
        reward = self._execute_action(action)
        
        # Update cellular automata
        self._update_cells()
        
        # Check termination conditions
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        # Calculate additional reward based on population
        population = np.sum(self.grid)
        stability_reward = self._calculate_stability_reward()
        
        total_reward = reward + 0.1 * population + stability_reward
        
        return self._get_obs(), total_reward, terminated, truncated, {'population': population}
    
    def _execute_action(self, action: int) -> float:
        reward = 0.0
        
        if action == 0:  # Move up
            new_pos = self.agent_pos + np.array([-1, 0])
        elif action == 1:  # Move down
            new_pos = self.agent_pos + np.array([1, 0])
        elif action == 2:  # Move left
            new_pos = self.agent_pos + np.array([0, -1])
        elif action == 3:  # Move right
            new_pos = self.agent_pos + np.array([0, 1])
        elif action == 4:  # Place/remove cell
            x, y = self.agent_pos
            if self.grid[x, y] == 0:
                self.grid[x, y] = 1
                self.cell_age[x, y] = 0
                reward = 1.0
            else:
                self.grid[x, y] = 0
                self.cell_age[x, y] = 0
                reward = -0.5
            return reward
        else:
            new_pos = self.agent_pos
        
        # Validate and update agent position
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)
        if not np.array_equal(new_pos, self.agent_pos):
            reward = 0.1  # Small reward for movement
        self.agent_pos = new_pos
        
        return reward
    
    def _update_cells(self):
        new_grid = self.grid.copy()
        new_age = self.cell_age.copy()
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                neighbors = self._count_neighbors(i, j)
                
                if self.grid[i, j] == 1:  # Cell is alive
                    new_age[i, j] += 1
                    # Conway's Game of Life rules with modifications
                    if neighbors < 2 or neighbors > 3:
                        new_grid[i, j] = 0  # Dies
                        new_age[i, j] = 0
                else:  # Cell is dead
                    if neighbors == 3:
                        new_grid[i, j] = 1  # Born
                        new_age[i, j] = 0
        
        self.grid = new_grid
        self.cell_age = new_age
    
    def _count_neighbors(self, x: int, y: int) -> int:
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    count += self.grid[nx, ny]
        return count
    
    def _calculate_stability_reward(self) -> float:
        # Reward for maintaining stable patterns
        if self.step_count < 10:
            return 0.0
        
        population = np.sum(self.grid)
        if 50 <= population <= 200:  # Ideal population range
            return 0.5
        elif population > 300:  # Overpopulation penalty
            return -1.0
        elif population < 10:  # Extinction penalty
            return -2.0
        else:
            return 0.0
    
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # Channel 0: Cell presence
        obs[:, :, 0] = self.grid.astype(np.float32)
        
        # Channel 1: Agent position
        obs[self.agent_pos[0], self.agent_pos[1], 1] = 1.0
        
        # Channel 2: Cell age (normalized)
        max_age = np.max(self.cell_age) if np.max(self.cell_age) > 0 else 1
        obs[:, :, 2] = self.cell_age.astype(np.float32) / max_age
        
        return obs
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()
            
            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((255, 255, 255))
            
            cell_size = self.window_size // self.grid_size
            
            # Draw cells
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    x = j * cell_size
                    y = i * cell_size
                    
                    if self.grid[i, j] == 1:
                        # Color based on age
                        age_ratio = min(self.cell_age[i, j] / 10.0, 1.0)
                        color = (int(255 * (1 - age_ratio)), int(255 * age_ratio), 0)
                        pygame.draw.rect(canvas, color, (x, y, cell_size, cell_size))
            
            # Draw agent
            agent_x = self.agent_pos[1] * cell_size + cell_size // 2
            agent_y = self.agent_pos[0] * cell_size + cell_size // 2
            pygame.draw.circle(canvas, (255, 0, 0), (agent_x, agent_y), cell_size // 3)
            
            # Draw grid lines
            for i in range(self.grid_size + 1):
                pygame.draw.line(canvas, (200, 200, 200), 
                               (0, i * cell_size), (self.window_size, i * cell_size))
                pygame.draw.line(canvas, (200, 200, 200), 
                               (i * cell_size, 0), (i * cell_size, self.window_size))
            
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
    
    def _render_frame(self):
        canvas = np.ones((self.window_size, self.window_size, 3), dtype=np.uint8) * 255
        cell_size = self.window_size // self.grid_size
        
        # Draw cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 1:
                    x_start = j * cell_size
                    y_start = i * cell_size
                    
                    age_ratio = min(self.cell_age[i, j] / 10.0, 1.0)
                    color = [int(255 * (1 - age_ratio)), int(255 * age_ratio), 0]
                    
                    canvas[y_start:y_start+cell_size, x_start:x_start+cell_size] = color
        
        # Draw agent
        agent_x = self.agent_pos[1] * cell_size + cell_size // 2
        agent_y = self.agent_pos[0] * cell_size + cell_size // 2
        
        # Simple circle approximation
        for dy in range(-cell_size//3, cell_size//3):
            for dx in range(-cell_size//3, cell_size//3):
                if dx*dx + dy*dy <= (cell_size//3)**2:
                    y, x = agent_y + dy, agent_x + dx
                    if 0 <= y < self.window_size and 0 <= x < self.window_size:
                        canvas[y, x] = [255, 0, 0]
        
        return canvas
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None