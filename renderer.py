#!/usr/bin/env python3
import pygame
import sys
import numpy as np

class BasicRenderer:
    def __init__(self, env, window_size=512):
        self.env = env
        self.window_size = window_size
        self.cell_size = window_size // env.size
        self.running = True
        self.paused = False
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Pixel Life - Press SPACE to pause, ESC to quit")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
    
    def render(self):
        self.screen.fill(self.BLACK)
        
        grid = self.env.get_state()
        
        # Draw white pixels
        for i in range(self.env.size):
            for j in range(self.env.size):
                if grid[i, j] == 1:  # White pixel
                    x = j * self.cell_size
                    y = i * self.cell_size
                    pygame.draw.rect(self.screen, self.WHITE, 
                                   (x, y, self.cell_size, self.cell_size))
        
        pygame.display.flip()
    
    def run(self):
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  ESC   - Quit")
        
        while self.running:
            self.handle_events()
            
            if not self.paused:
                self.env.step()
            
            self.render()
            self.clock.tick(10)  # 10 FPS
        
        pygame.quit()

if __name__ == "__main__":
    from env import PixelLifeEnv
    env = PixelLifeEnv(size=64)
    renderer = BasicRenderer(env)
    renderer.run()