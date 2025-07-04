"""
Renderer Diagnostics System
A comprehensive testing framework for diagnosing visual rendering issues without direct observation.
"""

import os
import sys
import numpy as np
import json
import csv
import datetime
import tempfile
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import unittest
from collections import defaultdict

# Set headless mode for pygame tests
os.environ['SDL_VIDEODRIVER'] = 'dummy'

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from enhanced_renderer import EnhancedPixelRenderer, EnhancedPixelLifeRenderer
from basic_renderer import BasicPixelRenderer, PixelLifeRenderer


@dataclass
class PixelDiagnosticResult:
    """Result of a pixel diagnostic test."""
    test_name: str
    passed: bool
    details: Dict[str, Any]
    error_message: Optional[str] = None
    pixel_data: Optional[np.ndarray] = None
    coordinates: Optional[List[Tuple[int, int]]] = None


class RendererDiagnostics:
    """Comprehensive diagnostic system for renderer visual output."""
    
    def __init__(self, renderer_class=None, renderer_kwargs=None):
        """Initialize the diagnostic system.
        
        Args:
            renderer_class: Class of renderer to test
            renderer_kwargs: Arguments to pass to renderer constructor
        """
        self.renderer_class = renderer_class or EnhancedPixelRenderer
        self.renderer_kwargs = renderer_kwargs or {}
        self.results = []
        self.test_grids = {}
        
        # Create test patterns
        self._create_test_patterns()
    
    def _create_test_patterns(self):
        """Create various test patterns for diagnostic testing."""
        # Simple test patterns
        self.test_grids = {
            'empty': np.zeros((20, 20), dtype=int),
            'single_pixel': np.zeros((20, 20), dtype=int),
            'checkerboard': np.zeros((20, 20), dtype=int),
            'border': np.zeros((20, 20), dtype=int),
            'diagonal': np.zeros((20, 20), dtype=int),
            'cluster': np.zeros((20, 20), dtype=int),
            'random': np.zeros((20, 20), dtype=int),
            'large_grid': np.zeros((100, 100), dtype=int),
        }
        
        # Single pixel in center
        self.test_grids['single_pixel'][10, 10] = 1
        
        # Checkerboard pattern
        self.test_grids['checkerboard'][::2, ::2] = 1
        
        # Border pattern
        self.test_grids['border'][0, :] = 1
        self.test_grids['border'][-1, :] = 1
        self.test_grids['border'][:, 0] = 1
        self.test_grids['border'][:, -1] = 1
        
        # Diagonal pattern
        np.fill_diagonal(self.test_grids['diagonal'], 1)
        
        # Cluster pattern (3x3 in center)
        self.test_grids['cluster'][9:12, 9:12] = 1
        
        # Random pattern
        np.random.seed(42)  # For reproducible tests
        self.test_grids['random'] = (np.random.random((20, 20)) > 0.7).astype(int)
        
        # Large grid with sparse pixels
        self.test_grids['large_grid'][25, 25] = 1
        self.test_grids['large_grid'][75, 75] = 1
        self.test_grids['large_grid'][50, 50] = 1
    
    def run_all_diagnostics(self) -> List[PixelDiagnosticResult]:
        """Run all diagnostic tests and return results."""
        print("Running comprehensive renderer diagnostics...")
        
        # Basic functionality tests
        self.results.extend(self._test_basic_functionality())
        
        # Coordinate transformation tests
        self.results.extend(self._test_coordinate_transformations())
        
        # Pixel rendering tests
        self.results.extend(self._test_pixel_rendering())
        
        # Zoom and camera tests
        self.results.extend(self._test_zoom_and_camera())
        
        # Grid update tests
        self.results.extend(self._test_grid_updates())
        
        # Performance tests
        self.results.extend(self._test_performance())
        
        # Memory and resource tests
        self.results.extend(self._test_memory_usage())
        
        return self.results
    
    def _test_basic_functionality(self) -> List[PixelDiagnosticResult]:
        """Test basic renderer functionality."""
        results = []
        
        try:
            # Test renderer creation
            renderer = self.renderer_class(**self.renderer_kwargs)
            results.append(PixelDiagnosticResult(
                test_name="renderer_creation",
                passed=True,
                details={"renderer_type": type(renderer).__name__}
            ))
            
            # Test basic properties
            expected_props = ['width', 'height', 'screen']
            for prop in expected_props:
                if hasattr(renderer, prop):
                    results.append(PixelDiagnosticResult(
                        test_name=f"property_{prop}",
                        passed=True,
                        details={"property": prop, "value": getattr(renderer, prop)}
                    ))
                else:
                    results.append(PixelDiagnosticResult(
                        test_name=f"property_{prop}",
                        passed=False,
                        details={"property": prop},
                        error_message=f"Missing property: {prop}"
                    ))
            
            # Test color definitions
            expected_colors = ['BLACK', 'WHITE', 'GRAY']
            for color in expected_colors:
                if hasattr(renderer, color):
                    color_value = getattr(renderer, color)
                    if isinstance(color_value, tuple) and len(color_value) == 3:
                        results.append(PixelDiagnosticResult(
                            test_name=f"color_{color}",
                            passed=True,
                            details={"color": color, "value": color_value}
                        ))
                    else:
                        results.append(PixelDiagnosticResult(
                            test_name=f"color_{color}",
                            passed=False,
                            details={"color": color, "value": color_value},
                            error_message=f"Invalid color format: {color_value}"
                        ))
                else:
                    results.append(PixelDiagnosticResult(
                        test_name=f"color_{color}",
                        passed=False,
                        details={"color": color},
                        error_message=f"Missing color: {color}"
                    ))
            
        except Exception as e:
            results.append(PixelDiagnosticResult(
                test_name="basic_functionality",
                passed=False,
                details={},
                error_message=f"Basic functionality test failed: {str(e)}"
            ))
        
        return results
    
    def _test_coordinate_transformations(self) -> List[PixelDiagnosticResult]:
        """Test coordinate transformation methods."""
        results = []
        
        try:
            renderer = self.renderer_class(**self.renderer_kwargs)
            
            # Test world_to_screen transformation
            if hasattr(renderer, 'world_to_screen'):
                test_cases = [
                    (0, 0, 0, 0),
                    (10, 10, 10, 10),
                    (-5, -5, -5, -5),
                ]
                
                for world_x, world_y, expected_screen_x, expected_screen_y in test_cases:
                    try:
                        screen_x, screen_y = renderer.world_to_screen(world_x, world_y)
                        
                        # Allow for small floating point differences
                        tolerance = 0.001
                        x_match = abs(screen_x - expected_screen_x) < tolerance
                        y_match = abs(screen_y - expected_screen_y) < tolerance
                        
                        results.append(PixelDiagnosticResult(
                            test_name=f"world_to_screen_{world_x}_{world_y}",
                            passed=x_match and y_match,
                            details={
                                "world_coords": (world_x, world_y),
                                "screen_coords": (screen_x, screen_y),
                                "expected": (expected_screen_x, expected_screen_y)
                            },
                            error_message=None if (x_match and y_match) else "Coordinate mismatch"
                        ))
                    except Exception as e:
                        results.append(PixelDiagnosticResult(
                            test_name=f"world_to_screen_{world_x}_{world_y}",
                            passed=False,
                            details={"world_coords": (world_x, world_y)},
                            error_message=f"Transformation failed: {str(e)}"
                        ))
            
            # Test screen_to_world transformation
            if hasattr(renderer, 'screen_to_world'):
                test_cases = [
                    (0, 0, 0, 0),
                    (100, 100, 100, 100),
                    (-50, -50, -50, -50),
                ]
                
                for screen_x, screen_y, expected_world_x, expected_world_y in test_cases:
                    try:
                        world_x, world_y = renderer.screen_to_world(screen_x, screen_y)
                        
                        # Allow for small floating point differences
                        tolerance = 0.001
                        x_match = abs(world_x - expected_world_x) < tolerance
                        y_match = abs(world_y - expected_world_y) < tolerance
                        
                        results.append(PixelDiagnosticResult(
                            test_name=f"screen_to_world_{screen_x}_{screen_y}",
                            passed=x_match and y_match,
                            details={
                                "screen_coords": (screen_x, screen_y),
                                "world_coords": (world_x, world_y),
                                "expected": (expected_world_x, expected_world_y)
                            },
                            error_message=None if (x_match and y_match) else "Coordinate mismatch"
                        ))
                    except Exception as e:
                        results.append(PixelDiagnosticResult(
                            test_name=f"screen_to_world_{screen_x}_{screen_y}",
                            passed=False,
                            details={"screen_coords": (screen_x, screen_y)},
                            error_message=f"Transformation failed: {str(e)}"
                        ))
            
            # Test round-trip transformations
            if hasattr(renderer, 'world_to_screen') and hasattr(renderer, 'screen_to_world'):
                test_coords = [(0, 0), (10, 10), (-5, -5), (100, 100)]
                
                for world_x, world_y in test_coords:
                    try:
                        screen_x, screen_y = renderer.world_to_screen(world_x, world_y)
                        back_world_x, back_world_y = renderer.screen_to_world(screen_x, screen_y)
                        
                        # Check if round-trip preserves coordinates
                        tolerance = 0.001
                        x_preserved = abs(world_x - back_world_x) < tolerance
                        y_preserved = abs(world_y - back_world_y) < tolerance
                        
                        results.append(PixelDiagnosticResult(
                            test_name=f"round_trip_{world_x}_{world_y}",
                            passed=x_preserved and y_preserved,
                            details={
                                "original": (world_x, world_y),
                                "screen": (screen_x, screen_y),
                                "back": (back_world_x, back_world_y)
                            },
                            error_message=None if (x_preserved and y_preserved) else "Round-trip failed"
                        ))
                    except Exception as e:
                        results.append(PixelDiagnosticResult(
                            test_name=f"round_trip_{world_x}_{world_y}",
                            passed=False,
                            details={"original": (world_x, world_y)},
                            error_message=f"Round-trip failed: {str(e)}"
                        ))
            
        except Exception as e:
            results.append(PixelDiagnosticResult(
                test_name="coordinate_transformations",
                passed=False,
                details={},
                error_message=f"Coordinate transformation tests failed: {str(e)}"
            ))
        
        return results
    
    def _test_pixel_rendering(self) -> List[PixelDiagnosticResult]:
        """Test pixel rendering with various patterns."""
        results = []
        
        try:
            renderer = self.renderer_class(**self.renderer_kwargs)
            
            # Test each pattern
            for pattern_name, test_grid in self.test_grids.items():
                try:
                    # Update renderer with test grid
                    if hasattr(renderer, 'env_grid'):
                        renderer.env_grid = test_grid.copy()
                        renderer.env_height, renderer.env_width = test_grid.shape
                    
                    # Count expected pixels
                    expected_pixels = np.sum(test_grid == 1)
                    
                    # Get pixel coordinates
                    live_coords = list(zip(*np.where(test_grid == 1)))
                    
                    # Test if renderer can process the grid
                    render_success = True
                    render_error_msg = None
                    if hasattr(renderer, 'render'):
                        try:
                            renderer.render()
                        except Exception as render_error:
                            render_success = False
                            render_error_msg = str(render_error)
                    
                    results.append(PixelDiagnosticResult(
                        test_name=f"pattern_{pattern_name}",
                        passed=render_success,
                        details={
                            "pattern": pattern_name,
                            "grid_shape": test_grid.shape,
                            "expected_pixels": int(expected_pixels),
                            "pixel_coordinates": live_coords
                        },
                        error_message=render_error_msg,
                        pixel_data=test_grid.copy(),
                        coordinates=live_coords
                    ))
                    
                except Exception as e:
                    results.append(PixelDiagnosticResult(
                        test_name=f"pattern_{pattern_name}",
                        passed=False,
                        details={"pattern": pattern_name},
                        error_message=f"Pattern test failed: {str(e)}"
                    ))
            
        except Exception as e:
            results.append(PixelDiagnosticResult(
                test_name="pixel_rendering",
                passed=False,
                details={},
                error_message=f"Pixel rendering tests failed: {str(e)}"
            ))
        
        return results
    
    def _test_zoom_and_camera(self) -> List[PixelDiagnosticResult]:
        """Test zoom and camera functionality."""
        results = []
        
        try:
            renderer = self.renderer_class(**self.renderer_kwargs)
            
            # Test zoom functionality
            if hasattr(renderer, 'zoom'):
                initial_zoom = renderer.zoom
                
                # Test zoom in
                if hasattr(renderer, 'zoom_in'):
                    renderer.zoom_in()
                    zoom_increased = renderer.zoom > initial_zoom
                    results.append(PixelDiagnosticResult(
                        test_name="zoom_in",
                        passed=zoom_increased,
                        details={
                            "initial_zoom": initial_zoom,
                            "new_zoom": renderer.zoom
                        },
                        error_message=None if zoom_increased else "Zoom did not increase"
                    ))
                
                # Test zoom out
                if hasattr(renderer, 'zoom_out'):
                    current_zoom = renderer.zoom
                    renderer.zoom_out()
                    zoom_decreased = renderer.zoom < current_zoom
                    results.append(PixelDiagnosticResult(
                        test_name="zoom_out",
                        passed=zoom_decreased,
                        details={"previous_zoom": current_zoom, "current_zoom": renderer.zoom},
                        error_message=None if zoom_decreased else "Zoom did not decrease"
                    ))
                
                # Test zoom limits
                if hasattr(renderer, 'min_zoom') and hasattr(renderer, 'max_zoom'):
                    results.append(PixelDiagnosticResult(
                        test_name="zoom_limits",
                        passed=renderer.min_zoom <= renderer.zoom <= renderer.max_zoom,
                        details={
                            "min_zoom": renderer.min_zoom,
                            "current_zoom": renderer.zoom,
                            "max_zoom": renderer.max_zoom
                        },
                        error_message=None if (renderer.min_zoom <= renderer.zoom <= renderer.max_zoom) else "Zoom out of bounds"
                    ))
            
            # Test camera functionality
            if hasattr(renderer, 'camera_x') and hasattr(renderer, 'camera_y'):
                initial_camera_x = renderer.camera_x
                initial_camera_y = renderer.camera_y
                
                # Test camera reset
                if hasattr(renderer, 'reset_view'):
                    # Change camera position first
                    renderer.camera_x = 100
                    renderer.camera_y = 100
                    renderer.reset_view()
                    camera_reset = (renderer.camera_x != 100 or renderer.camera_y != 100)
                    results.append(PixelDiagnosticResult(
                        test_name="camera_reset",
                        passed=camera_reset,
                        details={
                            "initial_camera": (initial_camera_x, initial_camera_y),
                            "changed_camera": (100, 100),
                            "reset_camera": (renderer.camera_x, renderer.camera_y)
                        },
                        error_message=None if camera_reset else "Camera position unchanged"
                    ))
            
        except Exception as e:
            results.append(PixelDiagnosticResult(
                test_name="zoom_and_camera",
                passed=False,
                details={},
                error_message=f"Zoom and camera tests failed: {str(e)}"
            ))
        
        return results
    
    def _test_grid_updates(self) -> List[PixelDiagnosticResult]:
        """Test grid update functionality."""
        results = []
        
        try:
            renderer = self.renderer_class(**self.renderer_kwargs)
            
            # Test grid update methods
            if hasattr(renderer, 'update_from_env'):
                # Create mock environment
                class MockEnv:
                    def __init__(self, grid):
                        self.grid = grid
                        self.tick_count = 0
                        self.live_pixels = set(zip(*np.where(grid == 1)))
                        self.dead_cells = set()
                        self.pixel_energy = {(y, x): 1.0 for y, x in self.live_pixels}
                        self.pixel_ages = {(y, x): 0 for y, x in self.live_pixels}
                        self.params = {}
                
                # Test with different grid sizes
                test_grids = {
                    'small': np.zeros((10, 10), dtype=int),
                    'medium': np.zeros((50, 50), dtype=int),
                    'large': np.zeros((100, 100), dtype=int),
                }
                
                # Add some pixels to test grids
                test_grids['small'][5, 5] = 1
                test_grids['medium'][25, 25] = 1
                test_grids['large'][50, 50] = 1
                
                for size_name, test_grid in test_grids.items():
                    try:
                        mock_env = MockEnv(test_grid)
                        renderer.update_from_env(mock_env)
                        
                        # Check if grid was updated
                        if hasattr(renderer, 'env_grid') and renderer.env_grid is not None:
                            grid_updated = np.array_equal(renderer.env_grid, test_grid)
                            results.append(PixelDiagnosticResult(
                                test_name=f"grid_update_{size_name}",
                                passed=grid_updated,
                                details={
                                    "size": size_name,
                                    "grid_shape": test_grid.shape,
                                    "pixels": int(np.sum(test_grid == 1))
                                },
                                error_message=None if grid_updated else "Grid not updated correctly"
                            ))
                        else:
                            results.append(PixelDiagnosticResult(
                                test_name=f"grid_update_{size_name}",
                                passed=False,
                                details={"size": size_name},
                                error_message="No env_grid attribute found"
                            ))
                    
                    except Exception as e:
                        results.append(PixelDiagnosticResult(
                            test_name=f"grid_update_{size_name}",
                            passed=False,
                            details={"size": size_name},
                            error_message=f"Grid update failed: {str(e)}"
                        ))
            
        except Exception as e:
            results.append(PixelDiagnosticResult(
                test_name="grid_updates",
                passed=False,
                details={},
                error_message=f"Grid update tests failed: {str(e)}"
            ))
        
        return results
    
    def _test_performance(self) -> List[PixelDiagnosticResult]:
        """Test rendering performance."""
        results = []
        
        try:
            renderer = self.renderer_class(**self.renderer_kwargs)
            
            # Test rendering speed with different grid sizes
            if hasattr(renderer, 'render'):
                import time
                
                test_sizes = [(20, 20), (50, 50), (100, 100)]
                
                for width, height in test_sizes:
                    try:
                        # Create test grid
                        test_grid = np.zeros((height, width), dtype=int)
                        test_grid[height//2, width//2] = 1  # Single pixel in center
                        
                        # Update renderer
                        if hasattr(renderer, 'env_grid'):
                            renderer.env_grid = test_grid
                            renderer.env_height, renderer.env_width = test_grid.shape
                        
                        # Measure render time
                        start_time = time.time()
                        for _ in range(10):  # Render 10 times for average
                            renderer.render()
                        end_time = time.time()
                        
                        avg_render_time = (end_time - start_time) / 10
                        
                        # Performance threshold: should render in under 100ms
                        performance_ok = avg_render_time < 0.1
                        
                        results.append(PixelDiagnosticResult(
                            test_name=f"performance_{width}x{height}",
                            passed=performance_ok,
                            details={
                                "grid_size": (width, height),
                                "avg_render_time": avg_render_time,
                                "threshold": 0.1
                            },
                            error_message=None if performance_ok else f"Render time {avg_render_time:.3f}s exceeds threshold"
                        ))
                    
                    except Exception as e:
                        results.append(PixelDiagnosticResult(
                            test_name=f"performance_{width}x{height}",
                            passed=False,
                            details={"grid_size": (width, height)},
                            error_message=f"Performance test failed: {str(e)}"
                        ))
            
        except Exception as e:
            results.append(PixelDiagnosticResult(
                test_name="performance",
                passed=False,
                details={},
                error_message=f"Performance tests failed: {str(e)}"
            ))
        
        return results
    
    def _test_memory_usage(self) -> List[PixelDiagnosticResult]:
        """Test memory usage and resource management."""
        results = []
        
        try:
            import psutil
            import gc
            
            # Test memory usage during renderer creation
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            renderer = self.renderer_class(**self.renderer_kwargs)
            
            after_creation_memory = process.memory_info().rss / 1024 / 1024
            creation_memory_increase = after_creation_memory - initial_memory
            
            # Test with large grid
            large_grid = np.zeros((200, 200), dtype=int)
            large_grid[100, 100] = 1
            
            if hasattr(renderer, 'env_grid'):
                renderer.env_grid = large_grid
                renderer.env_height, renderer.env_width = large_grid.shape
            
            after_large_grid_memory = process.memory_info().rss / 1024 / 1024
            large_grid_memory_increase = after_large_grid_memory - after_creation_memory
            
            # Test cleanup
            del renderer
            gc.collect()
            
            after_cleanup_memory = process.memory_info().rss / 1024 / 1024
            cleanup_memory_decrease = after_large_grid_memory - after_cleanup_memory
            
            results.append(PixelDiagnosticResult(
                test_name="memory_usage",
                passed=creation_memory_increase < 100,  # Should use less than 100MB
                details={
                    "initial_memory_mb": initial_memory,
                    "creation_memory_increase_mb": creation_memory_increase,
                    "large_grid_memory_increase_mb": large_grid_memory_increase,
                    "cleanup_memory_decrease_mb": cleanup_memory_decrease
                },
                error_message=None if creation_memory_increase < 100 else f"Memory usage too high: {creation_memory_increase:.1f}MB"
            ))
            
        except ImportError:
            results.append(PixelDiagnosticResult(
                test_name="memory_usage",
                passed=True,
                details={"note": "psutil not available, skipping memory test"}
            ))
        except Exception as e:
            results.append(PixelDiagnosticResult(
                test_name="memory_usage",
                passed=False,
                details={},
                error_message=f"Memory usage test failed: {str(e)}"
            ))
        
        return results
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive diagnostic report."""
        if not self.results:
            self.run_all_diagnostics()
        
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Group results by test category
        categories = defaultdict(list)
        for result in self.results:
            category = result.test_name.split('_')[0]
            categories[category].append(result)
        
        # Generate report
        report_lines = [
            "=" * 60,
            "RENDERER DIAGNOSTIC REPORT",
            "=" * 60,
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Renderer: {self.renderer_class.__name__}",
            "",
            f"SUMMARY:",
            f"  Total Tests: {total_tests}",
            f"  Passed: {passed_tests}",
            f"  Failed: {failed_tests}",
            f"  Pass Rate: {pass_rate:.1f}%",
            "",
            "DETAILED RESULTS:",
            ""
        ]
        
        # Add category results
        for category, category_results in categories.items():
            category_passed = sum(1 for r in category_results if r.passed)
            category_total = len(category_results)
            category_rate = (category_passed / category_total * 100) if category_total > 0 else 0
            
            report_lines.extend([
                f"{category.upper()} TESTS:",
                f"  Passed: {category_passed}/{category_total} ({category_rate:.1f}%)",
                ""
            ])
            
            # Add failed tests details
            failed_in_category = [r for r in category_results if not r.passed]
            if failed_in_category:
                report_lines.append("  FAILED TESTS:")
                for result in failed_in_category:
                    report_lines.append(f"    - {result.test_name}: {result.error_message}")
                report_lines.append("")
        
        # Add pixel coordinate data for successful tests
        successful_pixel_tests = [r for r in self.results if r.passed and r.coordinates]
        if successful_pixel_tests:
            report_lines.extend([
                "PIXEL COORDINATE DATA:",
                ""
            ])
            
            for result in successful_pixel_tests:
                if result.coordinates:
                    report_lines.append(f"  {result.test_name}:")
                    report_lines.append(f"    Coordinates: {result.coordinates}")
                    if result.pixel_data is not None:
                        pixel_count = np.sum(result.pixel_data == 1)
                        report_lines.append(f"    Pixel count: {pixel_count}")
                    report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Diagnostic report saved to: {output_file}")
        
        return report
    
    def export_pixel_data(self, output_file: str):
        """Export pixel coordinate data to JSON file."""
        if not self.results:
            self.run_all_diagnostics()
        
        pixel_data = {}
        for result in self.results:
            if result.coordinates:
                pixel_data[result.test_name] = {
                    "coordinates": result.coordinates,
                    "passed": result.passed,
                    "details": result.details
                }
        
        with open(output_file, 'w') as f:
            json.dump(pixel_data, f, indent=2)
        
        print(f"Pixel coordinate data exported to: {output_file}")


class RendererTestSuite(unittest.TestCase):
    """Unit test suite for renderer diagnostics."""
    
    def setUp(self):
        """Set up test environment."""
        self.diagnostics = RendererDiagnostics()
    
    def test_basic_functionality(self):
        """Test basic renderer functionality."""
        results = self.diagnostics._test_basic_functionality()
        failed_tests = [r for r in results if not r.passed]
        self.assertEqual(len(failed_tests), 0, f"Basic functionality tests failed: {failed_tests}")
    
    def test_coordinate_transformations(self):
        """Test coordinate transformation methods."""
        results = self.diagnostics._test_coordinate_transformations()
        failed_tests = [r for r in results if not r.passed]
        self.assertEqual(len(failed_tests), 0, f"Coordinate transformation tests failed: {failed_tests}")
    
    def test_pixel_rendering(self):
        """Test pixel rendering with various patterns."""
        results = self.diagnostics._test_pixel_rendering()
        failed_tests = [r for r in results if not r.passed]
        self.assertEqual(len(failed_tests), 0, f"Pixel rendering tests failed: {failed_tests}")
    
    def test_zoom_and_camera(self):
        """Test zoom and camera functionality."""
        results = self.diagnostics._test_zoom_and_camera()
        failed_tests = [r for r in results if not r.passed]
        self.assertEqual(len(failed_tests), 0, f"Zoom and camera tests failed: {failed_tests}")
    
    def test_grid_updates(self):
        """Test grid update functionality."""
        results = self.diagnostics._test_grid_updates()
        failed_tests = [r for r in results if not r.passed]
        self.assertEqual(len(failed_tests), 0, f"Grid update tests failed: {failed_tests}")
    
    def test_performance(self):
        """Test rendering performance."""
        results = self.diagnostics._test_performance()
        failed_tests = [r for r in results if not r.passed]
        self.assertEqual(len(failed_tests), 0, f"Performance tests failed: {failed_tests}")


def main():
    """Run comprehensive renderer diagnostics."""
    print("Renderer Diagnostics System")
    print("=" * 40)
    
    # Test both renderer types
    renderers_to_test = [
        (EnhancedPixelRenderer, {"width": 800, "height": 600}),
        (BasicPixelRenderer, {"width": 800, "height": 600, "grid_size": 20}),
    ]
    
    all_results = []
    
    for renderer_class, kwargs in renderers_to_test:
        print(f"\nTesting {renderer_class.__name__}...")
        
        diagnostics = RendererDiagnostics(renderer_class, kwargs)
        results = diagnostics.run_all_diagnostics()
        all_results.extend(results)
        
        # Generate report
        report = diagnostics.generate_report()
        print(report)
        
        # Export pixel data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pixel_data_file = f"pixel_data_{renderer_class.__name__}_{timestamp}.json"
        diagnostics.export_pixel_data(pixel_data_file)
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\nDiagnostics complete!")


if __name__ == "__main__":
    main() 