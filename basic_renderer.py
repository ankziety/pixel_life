"""
Basic Pixel Life Renderer
A simple renderer that displays white pixels in a window with pause/stop controls.
"""

import pygame
import sys
import numpy as np
from typing import Optional


class BasicPixelRenderer:
    def __init__(self, width: int = 800, height: int = 600, grid_size: int = 20):
        """Initialize the basic pixel renderer.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels  
            grid_size: Size of each grid cell in pixels
        """
        pygame.init()
        
        self.width = width
        self.height = height
        self.grid_size = grid_size
        
        # Calculate grid dimensions
        self.grid_width = width // grid_size
        self.grid_height = height // grid_size
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        
        # Create display
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Basic Pixel Life Renderer")
        
        # Initialize grid - True means white pixel, False means black
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        # Control state
        self.running = True
        self.paused = False
        
        # Clock for frame rate control
        self.clock = pygame.time.Clock()
        
        print("Basic Pixel Renderer Controls:")
        print("SPACE - Pause/Unpause")
        print("ESC or Q - Quit")
        print("Click - Toggle pixels")
        print("R - Random pixels")
        print("C - Clear all pixels")
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Unpaused'}")
                elif event.key == pygame.K_r:
                    self.randomize_grid()
                elif event.key == pygame.K_c:
                    self.clear_grid()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.toggle_pixel_at_mouse(event.pos)
    
    def toggle_pixel_at_mouse(self, pos):
        """Toggle pixel at mouse position."""
        grid_x = pos[0] // self.grid_size
        grid_y = pos[1] // self.grid_size
        
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            self.grid[grid_y, grid_x] = not self.grid[grid_y, grid_x]
    
    def randomize_grid(self):
        """Fill grid with random pixels."""
        self.grid = np.random.random((self.grid_height, self.grid_width)) > 0.7
        print("Randomized pixels")
    
    def clear_grid(self):
        """Clear all pixels."""
        self.grid.fill(False)
        print("Cleared all pixels")
    
    def set_pixel(self, x: int, y: int, value: bool = True):
        """Set a pixel at grid coordinates."""
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            self.grid[y, x] = value
    
    def set_pixels(self, coords: list, value: bool = True):
        """Set multiple pixels."""
        for x, y in coords:
            self.set_pixel(x, y, value)
    
    def update_from_array(self, array: np.ndarray):
        """Update grid from a numpy array."""
        if array.shape == self.grid.shape:
            self.grid = array.astype(bool)
        else:
            print(f"Warning: Array shape {array.shape} doesn't match grid shape {self.grid.shape}")
    
    def render(self):
        """Render the current grid state."""
        # Fill background
        self.screen.fill(self.BLACK)
        
        # Draw grid
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y, x]:
                    # Draw white pixel
                    rect = pygame.Rect(
                        x * self.grid_size,
                        y * self.grid_size,
                        self.grid_size,
                        self.grid_size
                    )
                    pygame.draw.rect(self.screen, self.WHITE, rect)
        
        # Draw grid lines (optional, very faint)
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(self.screen, (32, 32, 32), (x, 0), (x, self.height))
        for y in range(0, self.height, self.grid_size):
            pygame.draw.line(self.screen, (32, 32, 32), (0, y), (self.width, y))
        
        # Show pause state
        if self.paused:
            font = pygame.font.Font(None, 36)
            text = font.render("PAUSED", True, self.WHITE)
            text_rect = text.get_rect(center=(self.width // 2, 30))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()
    
    def run(self):
        """Main loop for the renderer."""
        self.randomize_grid()  # Start with some random pixels
        
        while self.running:
            self.handle_events()
            
            if not self.paused:
                # Here you could add simple animation logic
                # For now, we'll just occasionally flip some random pixels
                if np.random.random() < 0.02:  # 2% chance per frame
                    x, y = np.random.randint(0, self.grid_width), np.random.randint(0, self.grid_height)
                    self.grid[y, x] = not self.grid[y, x]
            
            self.render()
            self.clock.tick(30)  # 30 FPS
        
        pygame.quit()
        sys.exit()


class PixelLifeRenderer(BasicPixelRenderer):
    """Extended renderer that can interface with PixelLife environment."""
    
    def __init__(self, env=None, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.step_count = 0
    
    def update_from_env(self):
        """Update display from the environment state."""
        if self.env is not None:
            # Convert environment grid to boolean array for display
            # Assuming non-zero values should be displayed as white pixels
            env_grid = getattr(self.env, 'grid', None)
            if env_grid is not None:
                # Resize if needed
                if env_grid.shape != self.grid.shape:
                    import cv2
                    resized = cv2.resize(env_grid.astype(np.uint8), 
                                       (self.grid_width, self.grid_height), 
                                       interpolation=cv2.INTER_NEAREST)
                    self.grid = resized > 0
                else:
                    self.grid = env_grid > 0
    
    def run_with_env(self):
        """Run the renderer with environment integration."""
        while self.running:
            self.handle_events()
            
            if not self.paused and self.env is not None:
                # Step the environment
                try:
                    # This would need to be adapted based on your environment's interface
                    self.env.step({}, {})  # Placeholder for actual actions
                    self.update_from_env()
                    self.step_count += 1
                except Exception as e:
                    print(f"Environment step error: {e}")
                    self.paused = True
            
            self.render()
            
            # Show step count
            if self.env is not None:
                font = pygame.font.Font(None, 24)
                text = font.render(f"Step: {self.step_count}", True, self.WHITE)
                self.screen.blit(text, (10, 10))
            
            pygame.display.flip()
            self.clock.tick(10)  # Slower for environment steps
        
        pygame.quit()
        sys.exit()


def main():
    """Demo the basic renderer."""
    renderer = BasicPixelRenderer(width=800, height=600, grid_size=10)
    renderer.run()


if __name__ == "__main__":
    main()