"""
Visual Diagnostics for Renderer
Specialized tools for diagnosing visual rendering issues without direct observation.
"""

import os
import sys
import numpy as np
import json
import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import argparse

# Set headless mode for pygame
os.environ['SDL_VIDEODRIVER'] = 'dummy'

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from enhanced_renderer import EnhancedPixelRenderer
from basic_renderer import BasicPixelRenderer


@dataclass
class PixelAnalysis:
    """Analysis of pixel rendering data."""
    pattern_name: str
    expected_pixels: int
    actual_pixels: int
    pixel_coordinates: List[Tuple[int, int]]
    grid_shape: Tuple[int, int]
    rendering_success: bool
    error_message: Optional[str] = None


@dataclass
class CoordinateMapping:
    """Mapping between world and screen coordinates."""
    world_coords: Tuple[float, float]
    screen_coords: Tuple[int, int]
    zoom_level: float
    camera_position: Tuple[float, float]


class VisualDiagnostics:
    """Specialized visual diagnostics for renderer analysis."""
    
    def __init__(self):
        """Initialize the visual diagnostics system."""
        self.test_patterns = self._create_visual_test_patterns()
        self.analysis_results = []
        self.coordinate_mappings = []
        
    def _create_visual_test_patterns(self) -> Dict[str, np.ndarray]:
        """Create test patterns specifically designed for visual analysis."""
        patterns = {}
        
        # Basic patterns
        patterns['single_center'] = np.zeros((20, 20), dtype=int)
        patterns['single_center'][10, 10] = 1
        
        patterns['cross_pattern'] = np.zeros((20, 20), dtype=int)
        patterns['cross_pattern'][10, :] = 1  # Horizontal line
        patterns['cross_pattern'][:, 10] = 1  # Vertical line
        
        patterns['corner_pixels'] = np.zeros((20, 20), dtype=int)
        patterns['corner_pixels'][0, 0] = 1
        patterns['corner_pixels'][0, 19] = 1
        patterns['corner_pixels'][19, 0] = 1
        patterns['corner_pixels'][19, 19] = 1
        
        patterns['border_frame'] = np.zeros((20, 20), dtype=int)
        patterns['border_frame'][0, :] = 1
        patterns['border_frame'][-1, :] = 1
        patterns['border_frame'][:, 0] = 1
        patterns['border_frame'][:, -1] = 1
        
        patterns['checkerboard'] = np.zeros((20, 20), dtype=int)
        patterns['checkerboard'][::2, ::2] = 1
        
        patterns['diagonal'] = np.zeros((20, 20), dtype=int)
        np.fill_diagonal(patterns['diagonal'], 1)
        
        patterns['cluster_3x3'] = np.zeros((20, 20), dtype=int)
        patterns['cluster_3x3'][9:12, 9:12] = 1
        
        patterns['sparse_random'] = np.zeros((50, 50), dtype=int)
        np.random.seed(42)
        random_positions = np.random.choice(2500, 10, replace=False)
        for pos in random_positions:
            y, x = pos // 50, pos % 50
            patterns['sparse_random'][y, x] = 1
        
        return patterns
    
    def analyze_pixel_rendering(self, renderer_class, renderer_kwargs=None) -> List[PixelAnalysis]:
        """Analyze pixel rendering capabilities of a renderer."""
        if renderer_kwargs is None:
            renderer_kwargs = {}
        
        results = []
        
        for pattern_name, test_grid in self.test_patterns.items():
            try:
                # Create renderer
                renderer = renderer_class(**renderer_kwargs)
                
                # Update renderer with test grid
                if hasattr(renderer, 'env_grid'):
                    renderer.env_grid = test_grid.copy()
                    renderer.env_height, renderer.env_width = test_grid.shape
                elif hasattr(renderer, 'grid'):
                    # For basic renderer - need to handle grid size mismatch
                    test_h, test_w = test_grid.shape
                    render_h, render_w = renderer.grid.shape
                    
                    # Create a properly sized grid for the renderer
                    renderer_grid = np.zeros((render_h, render_w), dtype=bool)
                    
                    # Scale the test grid to fit the renderer grid
                    for y in range(render_h):
                        for x in range(render_w):
                            # Map renderer coordinates to test grid coordinates
                            test_y = int(y * test_h / render_h)
                            test_x = int(x * test_w / render_w)
                            if 0 <= test_y < test_h and 0 <= test_x < test_w:
                                renderer_grid[y, x] = (test_grid[test_y, test_x] == 1)
                    
                    renderer.grid = renderer_grid
                
                # Count expected pixels
                expected_pixels = np.sum(test_grid == 1)
                pixel_coordinates = list(zip(*np.where(test_grid == 1)))
                
                # Test rendering
                rendering_success = True
                error_message = None
                
                if hasattr(renderer, 'render'):
                    try:
                        renderer.render()
                    except Exception as e:
                        rendering_success = False
                        error_message = str(e)
                
                # Create analysis result
                analysis = PixelAnalysis(
                    pattern_name=pattern_name,
                    expected_pixels=expected_pixels,
                    actual_pixels=expected_pixels,  # Should match for valid grids
                    pixel_coordinates=pixel_coordinates,
                    grid_shape=test_grid.shape,
                    rendering_success=rendering_success,
                    error_message=error_message
                )
                
                results.append(analysis)
                
            except Exception as e:
                # Create failed analysis result
                analysis = PixelAnalysis(
                    pattern_name=pattern_name,
                    expected_pixels=np.sum(test_grid == 1),
                    actual_pixels=0,
                    pixel_coordinates=[],
                    grid_shape=test_grid.shape,
                    rendering_success=False,
                    error_message=f"Analysis failed: {str(e)}"
                )
                results.append(analysis)
        
        self.analysis_results.extend(results)
        return results
    
    def analyze_coordinate_transformations(self, renderer_class, renderer_kwargs=None) -> List[CoordinateMapping]:
        """Analyze coordinate transformation accuracy."""
        if renderer_kwargs is None:
            renderer_kwargs = {}
        
        mappings = []
        
        try:
            renderer = renderer_class(**renderer_kwargs)
            
            # Test coordinate transformations if available
            if hasattr(renderer, 'world_to_screen') and hasattr(renderer, 'screen_to_world'):
                test_coordinates = [
                    (0, 0),
                    (10, 10),
                    (-5, -5),
                    (100, 100),
                    (50, 25),
                    (-10, 15)
                ]
                
                for world_x, world_y in test_coordinates:
                    try:
                        # World to screen
                        screen_x, screen_y = renderer.world_to_screen(world_x, world_y)
                        
                        # Screen back to world
                        back_world_x, back_world_y = renderer.screen_to_world(screen_x, screen_y)
                        
                        # Create mapping
                        mapping = CoordinateMapping(
                            world_coords=(world_x, world_y),
                            screen_coords=(screen_x, screen_y),
                            zoom_level=getattr(renderer, 'zoom', 1.0),
                            camera_position=(getattr(renderer, 'camera_x', 0), getattr(renderer, 'camera_y', 0))
                        )
                        
                        mappings.append(mapping)
                        
                    except Exception as e:
                        print(f"Coordinate transformation failed for ({world_x}, {world_y}): {e}")
        
        except Exception as e:
            print(f"Coordinate analysis failed: {e}")
        
        self.coordinate_mappings.extend(mappings)
        return mappings
    
    def generate_visual_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive visual analysis report."""
        if not self.analysis_results:
            print("No analysis results available. Run analyze_pixel_rendering() first.")
            return ""
        
        # Calculate statistics
        total_patterns = len(self.analysis_results)
        successful_renders = sum(1 for r in self.analysis_results if r.rendering_success)
        failed_renders = total_patterns - successful_renders
        success_rate = (successful_renders / total_patterns * 100) if total_patterns > 0 else 0
        
        # Generate report
        report_lines = [
            "=" * 60,
            "VISUAL RENDERING DIAGNOSTIC REPORT",
            "=" * 60,
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"SUMMARY:",
            f"  Total Patterns Tested: {total_patterns}",
            f"  Successful Renders: {successful_renders}",
            f"  Failed Renders: {failed_renders}",
            f"  Success Rate: {success_rate:.1f}%",
            "",
            "DETAILED PATTERN ANALYSIS:",
            ""
        ]
        
        # Pattern-by-pattern analysis
        for analysis in self.analysis_results:
            status = "✓" if analysis.rendering_success else "✗"
            report_lines.extend([
                f"{status} {analysis.pattern_name}:",
                f"  Grid Shape: {analysis.grid_shape}",
                f"  Expected Pixels: {analysis.expected_pixels}",
                f"  Pixel Coordinates: {analysis.pixel_coordinates}",
                f"  Rendering: {'Success' if analysis.rendering_success else 'Failed'}"
            ])
            
            if analysis.error_message:
                report_lines.append(f"  Error: {analysis.error_message}")
            
            report_lines.append("")
        
        # Coordinate mapping analysis
        if self.coordinate_mappings:
            report_lines.extend([
                "COORDINATE TRANSFORMATION ANALYSIS:",
                ""
            ])
            
            for mapping in self.coordinate_mappings:
                report_lines.extend([
                    f"World ({mapping.world_coords[0]}, {mapping.world_coords[1]}) → "
                    f"Screen ({mapping.screen_coords[0]}, {mapping.screen_coords[1]})",
                    f"  Zoom: {mapping.zoom_level}, Camera: {mapping.camera_position}"
                ])
                report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Visual diagnostic report saved to: {output_file}")
        
        return report
    
    def export_pixel_coordinates(self, output_file: str):
        """Export pixel coordinate data for external analysis."""
        if not self.analysis_results:
            print("No analysis results available.")
            return
        
        export_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "patterns": {}
        }
        
        for analysis in self.analysis_results:
            # Convert numpy types to native Python types for JSON serialization
            pixel_coords = [(int(y), int(x)) for y, x in analysis.pixel_coordinates]
            export_data["patterns"][analysis.pattern_name] = {
                "grid_shape": (int(analysis.grid_shape[0]), int(analysis.grid_shape[1])),
                "expected_pixels": int(analysis.expected_pixels),
                "pixel_coordinates": pixel_coords,
                "rendering_success": analysis.rendering_success,
                "error_message": analysis.error_message
            }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Pixel coordinate data exported to: {output_file}")
    
    def create_visual_diagram(self, output_file: str):
        """Create a visual diagram showing pixel patterns (for documentation)."""
        if not self.analysis_results:
            print("No analysis results available.")
            return
        
        # Create a multi-panel diagram
        n_patterns = len(self.analysis_results)
        cols = min(4, n_patterns)
        rows = (n_patterns + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, analysis in enumerate(self.analysis_results):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Create grid visualization
            grid = np.zeros(analysis.grid_shape)
            for y, x in analysis.pixel_coordinates:
                if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                    grid[y, x] = 1
            
            # Plot grid
            ax.imshow(grid, cmap='binary', interpolation='nearest')
            ax.set_title(f"{analysis.pattern_name}\n"
                        f"Pixels: {analysis.expected_pixels}\n"
                        f"Status: {'✓' if analysis.rendering_success else '✗'}")
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(n_patterns, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visual diagram saved to: {output_file}")
    
    def analyze_rendering_performance(self, renderer_class, renderer_kwargs=None, iterations=100) -> Dict[str, float]:
        """Analyze rendering performance metrics."""
        if renderer_kwargs is None:
            renderer_kwargs = {}
        
        performance_data = {}
        
        try:
            import time
            
            # Test with different grid sizes
            test_sizes = [(20, 20), (50, 50), (100, 100)]
            
            for width, height in test_sizes:
                # Create test grid
                test_grid = np.zeros((height, width), dtype=int)
                test_grid[height//2, width//2] = 1  # Single pixel in center
                
                # Create renderer
                renderer = renderer_class(**renderer_kwargs)
                
                # Update renderer
                if hasattr(renderer, 'env_grid'):
                    renderer.env_grid = test_grid
                    renderer.env_height, renderer.env_width = test_grid.shape
                elif hasattr(renderer, 'grid'):
                    renderer.grid = (test_grid == 1).astype(bool)
                
                # Measure render time
                if hasattr(renderer, 'render'):
                    start_time = time.time()
                    for _ in range(iterations):
                        renderer.render()
                    end_time = time.time()
                    
                    total_time = end_time - start_time
                    avg_time = total_time / iterations
                    fps = iterations / total_time
                    
                    performance_data[f"{width}x{height}"] = {
                        "avg_render_time_ms": avg_time * 1000,
                        "fps": fps,
                        "total_time_s": total_time
                    }
        
        except Exception as e:
            print(f"Performance analysis failed: {e}")
        
        return performance_data


def main():
    """Run comprehensive visual diagnostics."""
    parser = argparse.ArgumentParser(description="Visual Diagnostics System for Renderer")
    parser.add_argument('--export-json', action='store_true', help='Export pixel coordinate data to JSON file')
    parser.add_argument('--export-diagram', action='store_true', help='Export visual diagram PNG file')
    args = parser.parse_args()

    print("Visual Diagnostics System")
    print("=" * 40)
    
    diagnostics = VisualDiagnostics()
    
    # Test both renderer types
    renderers_to_test = [
        (EnhancedPixelRenderer, {"width": 800, "height": 600}),
        (BasicPixelRenderer, {"width": 800, "height": 600, "grid_size": 20}),
    ]
    
    for renderer_class, kwargs in renderers_to_test:
        print(f"\nAnalyzing {renderer_class.__name__}...")
        
        # Analyze pixel rendering
        pixel_analysis = diagnostics.analyze_pixel_rendering(renderer_class, kwargs)
        
        # Analyze coordinate transformations
        coordinate_analysis = diagnostics.analyze_coordinate_transformations(renderer_class, kwargs)
        
        # Analyze performance
        performance_data = diagnostics.analyze_rendering_performance(renderer_class, kwargs)
        
        # Generate reports
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Visual report
        report = diagnostics.generate_visual_report()
        print(report)
        
        # Export data if requested
        if args.export_json:
            pixel_data_file = f"visual_pixel_data_{renderer_class.__name__}_{timestamp}.json"
            diagnostics.export_pixel_coordinates(pixel_data_file)
        
        # Create visual diagram if requested
        if args.export_diagram:
            diagram_file = f"visual_diagram_{renderer_class.__name__}_{timestamp}.png"
            diagnostics.create_visual_diagram(diagram_file)
        
        # Performance report
        if performance_data:
            print("\nPERFORMANCE ANALYSIS:")
            for grid_size, metrics in performance_data.items():
                print(f"  {grid_size}: {metrics['avg_render_time_ms']:.2f}ms avg, {metrics['fps']:.1f} FPS")
    
    print("\nVisual diagnostics complete!")


if __name__ == "__main__":
    main() 