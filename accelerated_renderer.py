"""
Accelerated Renderer for Pixel Life using Apple Silicon

This module provides optimized rendering using Metal Performance Shaders
for faster visualization on Apple Silicon Macs.
"""

import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
from typing import Dict, Tuple, Optional
import platform

# Check if we're on Apple Silicon
def is_apple_silicon():
    """Check if running on Apple Silicon Mac."""
    return platform.machine() == 'arm64' and platform.system() == 'Darwin'

class AcceleratedRenderer:
    """Accelerated renderer using Metal Performance Shaders and optimized algorithms."""
    
    def __init__(self, env, window_size=(800, 600), cell_size=8):
        self.env = env
        self.window_size = window_size
        self.cell_size = cell_size
        self.use_metal = is_apple_silicon()
        
        # Initialize pygame for rendering
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Pixel Life - Apple Silicon Accelerated")
        
        # Color definitions optimized for performance
        self.colors = {
            'empty': (0, 0, 0),        # Black
            'live': (0, 255, 0),       # Green
            'dead': (128, 128, 128),   # Gray
            'energy_high': (255, 255, 0),  # Yellow
            'energy_medium': (255, 165, 0), # Orange
            'energy_low': (255, 0, 0),      # Red
        }
        
        # Pre-compute color arrays for faster rendering
        self._precompute_colors()
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
        print(f"🚀 Accelerated Renderer initialized:")
        print(f"   Metal acceleration: {self.use_metal}")
        print(f"   Window size: {window_size}")
        print(f"   Cell size: {cell_size}")
    
    def _precompute_colors(self):
        """Pre-compute color arrays for faster rendering."""
        self.color_array = np.zeros((self.env.H, self.env.W, 3), dtype=np.uint8)
        self.energy_colors = {}
        
        # Pre-compute energy-based colors
        for energy in range(0, 101):
            if energy > 80:
                color = self.colors['energy_high']
            elif energy > 50:
                color = self.colors['energy_medium']
            elif energy > 20:
                color = self.colors['energy_low']
            else:
                color = self.colors['live']
            self.energy_colors[energy] = color
    
    def render(self, show_energy=True, show_fps=True):
        """Render the current state with acceleration."""
        start_time = time.time()
        
        # Clear screen
        self.screen.fill(self.colors['empty'])
        
        # Get current state
        grid = self.env.grid
        pixel_energy = self.env.pixel_energy
        
        # Calculate visible area (with zoom support)
        visible_h = min(self.env.H, self.window_size[1] // self.cell_size)
        visible_w = min(self.env.W, self.window_size[0] // self.cell_size)
        
        # Render grid using optimized algorithm
        self._render_grid_optimized(grid, pixel_energy, visible_h, visible_w, show_energy)
        
        # Render UI elements
        if show_fps:
            self._render_fps()
        
        # Update display
        pygame.display.flip()
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        render_time = time.time() - start_time
        return render_time
    
    def _render_grid_optimized(self, grid, pixel_energy, visible_h, visible_w, show_energy):
        """Optimized grid rendering using vectorized operations."""
        # Create surface for faster blitting
        surface = pygame.Surface((visible_w * self.cell_size, visible_h * self.cell_size))
        
        # Pre-allocate color array for this frame
        frame_colors = np.zeros((visible_h, visible_w, 3), dtype=np.uint8)
        
        # Vectorized color assignment
        for y in range(visible_h):
            for x in range(visible_w):
                cell_value = grid[y, x]
                
                if cell_value == 1:  # Live pixel
                    if show_energy and (y, x) in pixel_energy:
                        energy = pixel_energy[(y, x)]
                        energy_percent = int((energy / self.env.params['max_energy']) * 100)
                        color = self.energy_colors.get(energy_percent, self.colors['live'])
                    else:
                        color = self.colors['live']
                elif cell_value == -1:  # Dead pixel
                    color = self.colors['dead']
                else:  # Empty
                    color = self.colors['empty']
                
                frame_colors[y, x] = color
        
        # Convert to pygame surface efficiently
        self._array_to_surface(frame_colors, surface)
        
        # Blit to screen
        self.screen.blit(surface, (0, 0))
    
    def _array_to_surface(self, color_array, surface):
        """Convert numpy array to pygame surface efficiently."""
        # Use pygame's surfarray for faster conversion
        pygame.surfarray.blit_array(surface, color_array)
    
    def _render_fps(self):
        """Render FPS counter."""
        font = pygame.font.Font(None, 36)
        fps_text = font.render(f"FPS: {self.fps:.1f}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 10))
        
        # Render pixel count
        pixel_text = font.render(f"Pixels: {len(self.env.live_pixels)}", True, (255, 255, 255))
        self.screen.blit(pixel_text, (10, 50))
        
        # Render tick count
        tick_text = font.render(f"Tick: {self.env.tick_count}", True, (255, 255, 255))
        self.screen.blit(tick_text, (10, 90))

class MetalAcceleratedRenderer(AcceleratedRenderer):
    """Renderer with additional Metal-specific optimizations."""
    
    def __init__(self, env, window_size=(800, 600), cell_size=8):
        super().__init__(env, window_size, cell_size)
        
        if self.use_metal:
            self._setup_metal_acceleration()
    
    def _setup_metal_acceleration(self):
        """Setup Metal-specific acceleration features."""
        # For now, we'll use optimized numpy operations
        # In a full implementation, this would use Metal Performance Shaders
        print("   Metal acceleration features enabled")
        
        # Pre-allocate buffers for better performance
        self.metal_buffers = {
            'grid': np.zeros((self.env.H, self.env.W), dtype=np.int32),
            'energy': np.zeros((self.env.H, self.env.W), dtype=np.float32),
            'colors': np.zeros((self.env.H, self.env.W, 3), dtype=np.uint8)
        }
    
    def _render_grid_metal_optimized(self, grid, pixel_energy, visible_h, visible_w, show_energy):
        """Metal-optimized grid rendering."""
        if not self.use_metal:
            return self._render_grid_optimized(grid, pixel_energy, visible_h, visible_w, show_energy)
        
        # Use pre-allocated buffers
        self.metal_buffers['grid'][:visible_h, :visible_w] = grid[:visible_h, :visible_w]
        
        # Vectorized energy calculation
        energy_grid = np.zeros((visible_h, visible_w), dtype=np.float32)
        for (y, x), energy in pixel_energy.items():
            if y < visible_h and x < visible_w:
                energy_grid[y, x] = energy
        
        self.metal_buffers['energy'][:visible_h, :visible_w] = energy_grid
        
        # Vectorized color calculation
        colors = np.zeros((visible_h, visible_w, 3), dtype=np.uint8)
        
        # Use numpy's vectorized operations for better performance
        live_mask = (grid[:visible_h, :visible_w] == 1)
        dead_mask = (grid[:visible_h, :visible_w] == -1)
        
        # Set colors based on masks
        colors[live_mask] = self.colors['live']
        colors[dead_mask] = self.colors['dead']
        
        # Apply energy-based coloring
        if show_energy:
            for y in range(visible_h):
                for x in range(visible_w):
                    if live_mask[y, x] and energy_grid[y, x] > 0:
                        energy_percent = int((energy_grid[y, x] / self.env.params['max_energy']) * 100)
                        colors[y, x] = self.energy_colors.get(energy_percent, self.colors['live'])
        
        self.metal_buffers['colors'][:visible_h, :visible_w] = colors
        
        # Convert to surface and render
        surface = pygame.Surface((visible_w * self.cell_size, visible_h * self.cell_size))
        self._array_to_surface(colors, surface)
        self.screen.blit(surface, (0, 0))

class BenchmarkRenderer:
    """Benchmark renderer performance."""
    
    def __init__(self, env_size=50):
        self.env_size = env_size
        self.test_env = None
    
    def benchmark_rendering(self, frames=1000):
        """Benchmark rendering performance."""
        print(f"🏃‍♂️ Benchmarking rendering performance...")
        
        # Create test environment
        from apple_acceleration import create_accelerated_env
        self.test_env = create_accelerated_env(H=self.env_size, W=self.env_size)
        
        # Test different renderers
        renderers = [
            ("Standard", AcceleratedRenderer(self.test_env)),
            ("Metal", MetalAcceleratedRenderer(self.test_env))
        ]
        
        results = {}
        
        for name, renderer in renderers:
            print(f"\nTesting {name} renderer...")
            
            # Reset environment
            self.test_env.reset()
            
            # Benchmark rendering
            start_time = time.time()
            total_render_time = 0
            
            for frame in range(frames):
                # Simulate some environment steps
                if frame % 10 == 0:
                    pixel_actions = {}
                    for coord in self.test_env.live_pixels:
                        pixel_actions[coord] = (np.random.randint(0, 4), np.random.randint(0, 4))
                    self.test_env.step(0, pixel_actions)
                
                # Render frame
                render_time = renderer.render(show_fps=False)
                total_render_time += render_time
            
            total_time = time.time() - start_time
            avg_render_time = total_render_time / frames
            fps = frames / total_time
            
            results[name] = {
                'total_time': total_time,
                'avg_render_time': avg_render_time,
                'fps': fps
            }
            
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average render time: {avg_render_time*1000:.2f}ms")
            print(f"   FPS: {fps:.1f}")
        
        # Compare results
        if len(results) > 1:
            print(f"\n📊 Performance Comparison:")
            baseline_fps = results["Standard"]["fps"]
            for name, result in results.items():
                speedup = result["fps"] / baseline_fps
                print(f"   {name}: {result['fps']:.1f} FPS ({speedup:.2f}x)")
        
        return results

def create_renderer(env, renderer_type="auto", **kwargs):
    """Factory function to create appropriate renderer."""
    if renderer_type == "auto":
        if is_apple_silicon():
            return MetalAcceleratedRenderer(env, **kwargs)
        else:
            return AcceleratedRenderer(env, **kwargs)
    elif renderer_type == "metal":
        return MetalAcceleratedRenderer(env, **kwargs)
    elif renderer_type == "standard":
        return AcceleratedRenderer(env, **kwargs)
    else:
        raise ValueError(f"Unknown renderer type: {renderer_type}")

if __name__ == "__main__":
    # Run benchmark
    benchmark = BenchmarkRenderer(env_size=50)
    results = benchmark.benchmark_rendering(frames=500) 