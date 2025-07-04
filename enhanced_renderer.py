"""
Enhanced Pixel Life Renderer
A renderer with resizable window, fullscreen support, and zoom functionality.
"""

import pygame
import sys
import numpy as np
from typing import Optional, Tuple
import math
import os
import json
import csv
import datetime


class EnhancedPixelRenderer:
    def __init__(self, width: int = 1200, height: int = 800, initial_zoom: Optional[float] = None):
        """Initialize the enhanced pixel renderer.
        
        Args:
            width: Initial window width in pixels
            height: Initial window height in pixels  
            initial_zoom: Initial zoom level (None = auto-fit to window)
        """
        pygame.init()
        
        self.width = width
        self.height = height
        self.initial_zoom = initial_zoom
        
        # Zoom and viewport settings
        self.zoom = 1.0  # Start with reasonable zoom
        self.min_zoom = 0.1  # 10x smaller (was 0.001)
        self.max_zoom = 20.0  # 20x larger (was 10.0)
        self.zoom_speed = 1.2  # Zoom factor per scroll
        
        # Camera/viewport position
        self.camera_x = 0
        self.camera_y = 0
        self.drag_start = None
        self.is_dragging = False
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (32, 32, 32)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        # Create resizable display
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Enhanced Pixel Life Renderer")
        
        # Environment grid (will be set by environment)
        self.env_grid = None
        self.env_width = 0
        self.env_height = 0
        self.env_loaded = False  # Track if environment has been loaded
        
        # Control state
        self.running = True
        self.paused = False
        self.fullscreen = False
        
        # Clock for frame rate control
        self.clock = pygame.time.Clock()
        
        # Font for UI
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.debug_menu = False
        self.last_export_path = None
        self.debug_info = {}
        self.export_message = None
        
        print("Enhanced Pixel Renderer Controls:")
        print("SPACE - Pause/Unpause")
        print("F - Toggle fullscreen")
        print("ESC or Q - Quit")
        print("Mouse wheel - Zoom in/out")
        print("Middle mouse drag - Pan camera")
        print("Click - Toggle pixels")
        print("R - Reset zoom and camera")
        print("C - Center camera on live pixels")
        print("+/- - Zoom in/out")
        print("D - Toggle debug menu")
        print("E - Export debug info (when debug menu open)")

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
                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_r:
                    self.reset_view()
                elif event.key == pygame.K_c:
                    self.center_on_pixels()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.zoom_in()
                elif event.key == pygame.K_MINUS:
                    self.zoom_out()
                elif event.key == pygame.K_d:
                    self.debug_menu = not self.debug_menu
                    print(f"Debug menu {'shown' if self.debug_menu else 'hidden'}")
                elif event.key == pygame.K_e:
                    self.export_debug_info()
            
            elif event.type == pygame.VIDEORESIZE:
                if not self.fullscreen:
                    self.width, self.height = event.size
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.toggle_pixel_at_mouse(event.pos)
                elif event.button == 2:  # Middle mouse
                    self.drag_start = event.pos
                    self.is_dragging = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:  # Middle mouse
                    self.is_dragging = False
                    self.drag_start = None
            
            elif event.type == pygame.MOUSEMOTION:
                if self.is_dragging and self.drag_start:
                    dx = event.pos[0] - self.drag_start[0]
                    dy = event.pos[1] - self.drag_start[1]
                    self.camera_x -= dx / self.zoom
                    self.camera_y -= dy / self.zoom
                    self.drag_start = event.pos
            
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.zoom_in_at_mouse(pygame.mouse.get_pos())
                else:
                    self.zoom_out_at_mouse(pygame.mouse.get_pos())
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.width, self.height = self.screen.get_size()
        else:
            self.screen = pygame.display.set_mode((1200, 800), pygame.RESIZABLE)
            self.width, self.height = 1200, 800
        print(f"{'Entered' if self.fullscreen else 'Exited'} fullscreen mode")
    
    def zoom_in(self):
        """Zoom in."""
        self.zoom = min(self.zoom * self.zoom_speed, self.max_zoom)
    
    def zoom_out(self):
        """Zoom out."""
        self.zoom = max(self.zoom / self.zoom_speed, self.min_zoom)
    
    def fit_grid_to_window(self):
        """Auto-calculate zoom to fit the grid to the window."""
        if self.env_width > 0 and self.env_height > 0:
            zoom_x = self.width / self.env_width
            zoom_y = self.height / self.env_height
            self.zoom = max(self.min_zoom, min(zoom_x, zoom_y, self.max_zoom))
            # Center the grid
            self.camera_x = (self.env_width * self.zoom - self.width) / 2
            self.camera_y = (self.env_height * self.zoom - self.height) / 2
            print(f"Auto-fitted grid to window: zoom={self.zoom:.3f}")

    def zoom_in_at_mouse(self, mouse_pos):
        """Zoom in centered on mouse position."""
        old_zoom = self.zoom
        self.zoom = min(self.zoom * self.zoom_speed, self.max_zoom)
        
        # Adjust camera to keep mouse position fixed
        if self.zoom != old_zoom:
            mx, my = mouse_pos
            wx, wy = self.screen_to_world(mx, my)
            self.camera_x = wx * self.zoom - mx
            self.camera_y = wy * self.zoom - my

    def zoom_out_at_mouse(self, mouse_pos):
        """Zoom out centered on mouse position."""
        old_zoom = self.zoom
        self.zoom = max(self.zoom / self.zoom_speed, self.min_zoom)
        
        # Adjust camera to keep mouse position fixed
        if self.zoom != old_zoom:
            mx, my = mouse_pos
            wx, wy = self.screen_to_world(mx, my)
            self.camera_x = wx * self.zoom - mx
            self.camera_y = wy * self.zoom - my
    
    def reset_view(self):
        """Reset zoom and camera position."""
        self.zoom = self.initial_zoom if self.initial_zoom is not None else 1.0
        self.camera_x = 0
        self.camera_y = 0
        print("Reset view")
    
    def center_on_pixels(self):
        """Center camera on live pixels."""
        if self.env_grid is not None and np.any(self.env_grid == 1):
            # Find center of live pixels
            live_coords = np.where(self.env_grid == 1)
            if len(live_coords[0]) > 0:
                center_y = np.mean(live_coords[0])
                center_x = np.mean(live_coords[1])
                
                # Convert to screen coordinates
                self.camera_x = center_x * self.zoom - self.width / 2
                self.camera_y = center_y * self.zoom - self.height / 2
                print(f"Centered on pixels at ({center_x:.1f}, {center_y:.1f})")
    
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int((world_x * self.zoom) - self.camera_x)
        screen_y = int((world_y * self.zoom) - self.camera_y)
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        world_x = float((screen_x + self.camera_x) / self.zoom)
        world_y = float((screen_y + self.camera_y) / self.zoom)
        return world_x, world_y
    
    def toggle_pixel_at_mouse(self, pos):
        """Toggle pixel at mouse position."""
        world_x, world_y = self.screen_to_world(pos[0], pos[1])
        grid_x = int(world_x)
        grid_y = int(world_y)
        
        if 0 <= grid_x < self.env_width and 0 <= grid_y < self.env_height:
            if self.env_grid is not None:
                self.env_grid[grid_y, grid_x] = 1 - self.env_grid[grid_y, grid_x]
                print(f"Toggled pixel at ({grid_x}, {grid_y})")
    
    def update_from_env(self, env):
        """Update display from the environment state."""
        if hasattr(env, 'grid') and env.grid is not None:
            self.env_grid = env.grid.copy()
            self.env_height, self.env_width = env.grid.shape
            
            # Auto-fit grid to window on first load
            if self.initial_zoom is None and not self.env_loaded:
                self.fit_grid_to_window()
                self.env_loaded = True # Set flag to True after first fit
            elif self.initial_zoom is not None:
                self.zoom = max(self.min_zoom, min(self.initial_zoom, self.max_zoom))
            
            # Collect debug info if available
            self.debug_info = {}
            if hasattr(env, 'tick_count'):
                self.debug_info['step'] = env.tick_count
            if hasattr(env, 'live_pixels'):
                self.debug_info['live_pixels'] = len(env.live_pixels)
            if hasattr(env, 'dead_cells'):
                self.debug_info['dead_pixels'] = len(env.dead_cells)
            if hasattr(env, 'pixel_energy'):
                energies = list(env.pixel_energy.values())
                if energies:
                    self.debug_info['avg_energy'] = float(np.mean(energies))
                    self.debug_info['min_energy'] = float(np.min(energies))
                    self.debug_info['max_energy'] = float(np.max(energies))
                else:
                    self.debug_info['avg_energy'] = 0.0
                    self.debug_info['min_energy'] = 0.0
                    self.debug_info['max_energy'] = 0.0
            if hasattr(env, 'pixel_ages'):
                ages = list(env.pixel_ages.values())
                if ages:
                    self.debug_info['avg_age'] = float(np.mean(ages))
                    self.debug_info['max_age'] = float(np.max(ages))
                else:
                    self.debug_info['avg_age'] = 0.0
                    self.debug_info['max_age'] = 0.0
            if hasattr(env, 'params'):
                self.debug_info['params'] = dict(env.params)

    def export_debug_info(self):
        """Export current environment stats to a CSV and JSON file."""
        if self.env_grid is None:
            print("No environment data to export.")
            return
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"debug_export_{now}"
        # Gather stats
        stats = self.get_debug_stats()
        # Export JSON
        json_path = f"{base}.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        # Export CSV (flattened)
        csv_path = f"{base}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for k, v in stats.items():
                writer.writerow([k, v])
        self.last_export_path = (json_path, csv_path)
        print(f"Exported debug info to {json_path} and {csv_path}")

    def get_debug_stats(self):
        """Return a dict of detailed environment stats."""
        stats = {}
        if self.env_grid is not None:
            stats["grid_shape"] = list(self.env_grid.shape)
            stats["live_pixels"] = int(np.sum(self.env_grid == 1))
            stats["dead_pixels"] = int(np.sum(self.env_grid == -1))
            stats["empty_pixels"] = int(np.sum(self.env_grid == 0))
        env = getattr(self, 'env', None)
        if env is not None:
            stats["tick_count"] = getattr(env, "tick_count", None)
            stats["H"] = getattr(env, "W", None)
            stats["W"] = getattr(env, "W", None)
            stats["params"] = getattr(env, "params", None)
            stats["pixel_energy_mean"] = float(np.mean(list(getattr(env, "pixel_energy", {}).values())) if getattr(env, "pixel_energy", None) else 0)
            stats["pixel_ages_mean"] = float(np.mean(list(getattr(env, "pixel_ages", {}).values())) if getattr(env, "pixel_ages", None) else 0)
            stats["live_pixel_count"] = len(getattr(env, "live_pixels", []))
            stats["dead_cell_count"] = len(getattr(env, "dead_cells", []))
        return stats
    
    def render(self):
        """Render the current grid state."""
        # Fill background
        self.screen.fill(self.BLACK)
        
        if self.env_grid is not None:
            # Calculate visible grid range
            start_x, start_y = self.screen_to_world(0, 0)
            end_x, end_y = self.screen_to_world(self.width, self.height)
            
            # Add some margin for smooth scrolling
            start_x = max(0, int(start_x - 1))
            start_y = max(0, int(start_y - 1))
            end_x = min(self.env_width, int(end_x + 2))
            end_y = min(self.env_height, int(end_y + 2))
            
            # Ensure we don't exceed grid bounds
            start_x = max(0, min(start_x, self.env_width - 1))
            start_y = max(0, min(start_y, self.env_height - 1))
            end_x = max(0, min(end_x, self.env_width))
            end_y = max(0, min(end_y, self.env_height))
            
            # Draw visible pixels
            for y in range(start_y, end_y):
                for x in range(start_x, end_x):
                    if self.env_grid[y, x] == 1:
                        # Convert to screen coordinates
                        screen_x, screen_y = self.world_to_screen(x, y)
                        
                        # Calculate pixel size based on zoom - ensure minimum visibility
                        pixel_size = max(1, int(round(self.zoom)))
                        
                        # Only draw if pixel is visible on screen
                        if (0 <= screen_x < self.width and 0 <= screen_y < self.height):
                            # Draw pixel with proper bounds checking
                            rect = pygame.Rect(screen_x, screen_y, pixel_size, pixel_size)
                            pygame.draw.rect(self.screen, self.WHITE, rect)
            
            # Draw grid lines if zoomed in enough
            if self.zoom > 0.5:
                self.draw_grid_lines(start_x, start_y, end_x, end_y)
        
        # Draw UI
        self.draw_ui()
        if self.debug_menu:
            self.draw_debug_menu()
        
        pygame.display.flip()
    
    def draw_grid_lines(self, start_x, start_y, end_x, end_y):
        """Draw grid lines."""
        # Ensure bounds are within grid dimensions
        start_x = max(0, min(start_x, self.env_width - 1))
        start_y = max(0, min(start_y, self.env_height - 1))
        end_x = max(0, min(end_x, self.env_width - 1))
        end_y = max(0, min(end_y, self.env_height - 1))
        
        # Vertical lines
        for x in range(start_x, end_x + 1):
            screen_x, _ = self.world_to_screen(x, 0)
            pygame.draw.line(self.screen, self.DARK_GRAY, 
                           (screen_x, 0), (screen_x, self.height))
        
        # Horizontal lines
        for y in range(start_y, end_y + 1):
            _, screen_y = self.world_to_screen(0, y)
            pygame.draw.line(self.screen, self.DARK_GRAY, 
                           (0, screen_y), (self.width, screen_y))
    
    def draw_ui(self):
        """Draw UI elements."""
        # Show pause state
        if self.paused:
            text = self.font.render("PAUSED", True, self.RED)
            text_rect = text.get_rect(center=(self.width // 2, 30))
            self.screen.blit(text, text_rect)
        
        # Show zoom level
        zoom_text = self.small_font.render(f"Zoom: {self.zoom:.3f}x", True, self.WHITE)
        self.screen.blit(zoom_text, (10, 10))
        
        # Show camera position
        cam_text = self.small_font.render(f"Camera: ({self.camera_x:.0f}, {self.camera_y:.0f})", True, self.WHITE)
        self.screen.blit(cam_text, (10, 30))
        
        # Show environment info
        if self.env_grid is not None:
            live_pixels = np.sum(self.env_grid == 1)
            total_pixels = self.env_grid.size
            info_text = self.small_font.render(f"Live: {live_pixels}/{total_pixels}", True, self.GREEN)
            self.screen.blit(info_text, (10, 50))
            
            # Show grid size
            size_text = self.small_font.render(f"Grid: {self.env_width}x{self.env_height}", True, self.WHITE)
            self.screen.blit(size_text, (10, 70))
        
        # Show controls hint
        controls_text = self.small_font.render("Mouse wheel: Zoom | Middle drag: Pan | F: Fullscreen | D: Debug menu", True, self.GRAY)
        self.screen.blit(controls_text, (10, self.height - 20))
        # Removed: if self.show_debug: self.draw_debug_menu()
        # Removed: if self.export_message: ...

    def draw_debug_menu(self):
        """Draw debug overlay with detailed stats and export option."""
        # Semi-transparent background
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        # Title
        title = self.font.render("DEBUG MENU (D to close, E to export)", True, self.GREEN)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 40))
        # Gather stats
        stats = self.get_debug_stats()
        y = 100
        for k, v in stats.items():
            stat_text = self.small_font.render(f"{k}: {v}", True, self.WHITE)
            self.screen.blit(stat_text, (80, y))
            y += 28
        if self.last_export_path:
            export_text = self.small_font.render(f"Last export: {self.last_export_path[0]}, {self.last_export_path[1]}", True, self.GRAY)
            self.screen.blit(export_text, (80, y + 10))


class EnhancedPixelLifeRenderer(EnhancedPixelRenderer):
    """Enhanced renderer that can interface with PixelLife environment."""
    
    def __init__(self, env=None, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.step_count = 0
        
        if env is not None:
            self.update_from_env(env)
    
    def run_with_env(self):
        """Run the renderer with environment integration."""
        while self.running:
            self.handle_events()
            
            # Update display from environment
            if self.env is not None:
                self.update_from_env(self.env)
            
            self.render()
            self.clock.tick(60)  # 60 FPS for smooth display
        
        pygame.quit()
        sys.exit()

    def run(self):
        self.run_with_env()


def main():
    """Demo the enhanced renderer."""
    # Create a sample environment grid
    grid = np.zeros((100, 100), dtype=int)
    
    # Add some live pixels
    grid[50, 50] = 1
    grid[51, 50] = 1
    grid[50, 51] = 1
    grid[49, 50] = 1
    grid[50, 49] = 1
    
    # Create a mock environment
    class MockEnv:
        def __init__(self, grid):
            self.grid = grid
            self.H, self.W = grid.shape
    
    mock_env = MockEnv(grid)
    
    renderer = EnhancedPixelLifeRenderer(env=mock_env)
    renderer.run()


if __name__ == "__main__":
    main() 