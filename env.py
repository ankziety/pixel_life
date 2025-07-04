#!/usr/bin/env python3
import numpy as np

class PixelLifeEnv:
    def __init__(self, size=64):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.uint8)
        self.reset()
    
    def reset(self):
        # Initialize with random white pixels (1 = white, 0 = black)
        self.grid = np.random.choice([0, 1], size=(self.size, self.size), p=[0.7, 0.3])
        return self.grid.copy()
    
    def step(self):
        # Simple cellular automata rule - just for demonstration
        # Conway's Game of Life rules
        new_grid = self.grid.copy()
        
        for i in range(1, self.size-1):
            for j in range(1, self.size-1):
                # Count neighbors
                neighbors = np.sum(self.grid[i-1:i+2, j-1:j+2]) - self.grid[i, j]
                
                if self.grid[i, j] == 1:  # Alive
                    if neighbors < 2 or neighbors > 3:
                        new_grid[i, j] = 0  # Dies
                else:  # Dead
                    if neighbors == 3:
                        new_grid[i, j] = 1  # Born
        
        self.grid = new_grid
        return self.grid.copy()
    
    def get_state(self):
        return self.grid.copy()
    
    def set_pixel(self, x, y, value):
        if 0 <= x < self.size and 0 <= y < self.size:
            self.grid[x, y] = value