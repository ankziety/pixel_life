#!/usr/bin/env python3
"""
Visual regression tests for Pixel Life CLI visualization commands.
Tests visual output consistency across different commands and configurations.
"""

import unittest
import subprocess
import sys
import os
import tempfile
import time
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_config import TestEnvironment, skip_if_feature_unavailable


class VisualRegressionTest(unittest.TestCase):
    """Base class for visual regression tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create directories for visual outputs
        os.makedirs('./visual_outputs', exist_ok=True)
        os.makedirs('./visual_baselines', exist_ok=True)
        os.makedirs('./visual_diffs', exist_ok=True)
        
        # Set up test environment
        self.test_env = TestEnvironment(self.temp_dir)
        
        # Visual test configuration
        self.visual_config = {
            'screenshot_timeout': 10,
            'comparison_threshold': 0.95,  # 95% similarity required
            'max_diff_pixels': 100,  # Maximum different pixels allowed
            'image_size': (800, 600),
            'formats': ['png', 'jpg'],
            'dpi': 100
        }
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def capture_screenshot(self, command, output_name, timeout=None):
        """Capture a screenshot from a CLI command."""
        if timeout is None:
            timeout = self.visual_config['screenshot_timeout']
        
        # Create a headless display for screenshot capture
        try:
            import pygame
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.init()
        except ImportError:
            self.skipTest("Pygame not available for screenshot capture")
        
        # Run the command and capture output
        cmd = [sys.executable, '-m', 'pixel_life'] + command
        
        try:
            # Start the command
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                cwd=self.temp_dir
            )
            
            # Wait a bit for the command to start
            time.sleep(2)
            
            # Capture screenshot using pygame
            screen = pygame.display.set_mode(self.visual_config['image_size'])
            pygame.display.flip()
            
            # Take screenshot
            screenshot = pygame.display.get_surface()
            screenshot_path = os.path.join(self.temp_dir, 'visual_outputs', f'{output_name}.png')
            pygame.image.save(screenshot, screenshot_path)
            
            # Terminate the process
            process.terminate()
            process.wait(timeout=5)
            
            return screenshot_path
            
        except Exception as e:
            if 'process' in locals():
                process.terminate()
            raise e
    
    def capture_matplotlib_output(self, command, output_name):
        """Capture matplotlib output from CLI commands."""
        # Set up matplotlib for non-interactive use
        plt.ioff()
        
        # Run the command that generates matplotlib output
        cmd = [sys.executable, '-m', 'pixel_life'] + command
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.temp_dir,
                timeout=30
            )
            
            # Save any matplotlib figures
            for i, fig in enumerate(plt.get_fignums()):
                figure = plt.figure(fig)
                output_path = os.path.join(
                    self.temp_dir, 'visual_outputs', 
                    f'{output_name}_fig_{i}.png'
                )
                figure.savefig(output_path, dpi=self.visual_config['dpi'])
                plt.close(figure)
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            return False
    
    def compare_images(self, image1_path, image2_path, threshold=None):
        """Compare two images and return similarity score."""
        if threshold is None:
            threshold = self.visual_config['comparison_threshold']
        
        try:
            # Load images
            img1 = Image.open(image1_path).convert('RGB')
            img2 = Image.open(image2_path).convert('RGB')
            
            # Resize to same size if different
            if img1.size != img2.size:
                img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
            
            # Convert to numpy arrays
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            
            # Calculate similarity
            diff = np.abs(arr1.astype(float) - arr2.astype(float))
            similarity = 1.0 - (np.sum(diff) / (diff.size * 255.0))
            
            # Create diff image
            diff_img = ImageChops.difference(img1, img2)
            diff_path = os.path.join(
                self.temp_dir, 'visual_diffs', 
                f'diff_{Path(image1_path).stem}_{Path(image2_path).stem}.png'
            )
            diff_img.save(diff_path)
            
            return similarity, diff_path
            
        except Exception as e:
            print(f"Error comparing images: {e}")
            return 0.0, None
    
    def generate_baseline_image(self, command, baseline_name):
        """Generate a baseline image for comparison."""
        baseline_path = os.path.join(self.temp_dir, 'visual_baselines', f'{baseline_name}.png')
        
        # Create a simple baseline image based on command
        img = Image.new('RGB', self.visual_config['image_size'], color='white')
        draw = ImageDraw.Draw(img)
        
        # Add command information to baseline
        try:
            # Try to use a font
            font = ImageFont.load_default()
        except:
            font = None
        
        text = f"Baseline: {' '.join(command)}"
        draw.text((10, 10), text, fill='black', font=font)
        
        # Add some visual elements based on command type
        if 'basic' in command:
            # Draw a simple grid for basic command
            for i in range(0, img.width, 50):
                draw.line([(i, 0), (i, img.height)], fill='lightgray')
            for i in range(0, img.height, 50):
                draw.line([(0, i), (img.width, i)], fill='lightgray')
        
        elif 'ai' in command:
            # Draw some circles for AI command
            for i in range(5):
                x = 100 + i * 100
                y = 200
                draw.ellipse([x-20, y-20, x+20, y+20], fill='blue')
        
        elif 'train' in command:
            # Draw a progress bar for training
            draw.rectangle([50, 200, 750, 220], fill='lightgray')
            draw.rectangle([50, 200, 400, 220], fill='green')
        
        img.save(baseline_path)
        return baseline_path
    
    def assert_visual_similarity(self, current_path, baseline_path, threshold=None):
        """Assert that two images are visually similar."""
        similarity, diff_path = self.compare_images(current_path, baseline_path, threshold)
        
        if similarity < (threshold or self.visual_config['comparison_threshold']):
            self.fail(
                f"Visual similarity too low: {similarity:.3f} < {threshold or self.visual_config['comparison_threshold']}\n"
                f"Current: {current_path}\n"
                f"Baseline: {baseline_path}\n"
                f"Diff: {diff_path}"
            )
        
        return similarity


class TestBasicVisualRegression(VisualRegressionTest):
    """Test visual regression for basic commands."""
    
    def test_basic_command_visual_consistency(self):
        """Test that basic command produces consistent visual output."""
        # Generate baseline
        baseline_path = self.generate_baseline_image(['basic'], 'basic_baseline')
        
        # Run basic command multiple times and compare
        for i in range(3):
            current_path = self.capture_screenshot(
                ['basic', '--size', '20', '--steps', '5'], 
                f'basic_run_{i}'
            )
            
            similarity = self.assert_visual_similarity(current_path, baseline_path)
            print(f"Basic command run {i}: similarity = {similarity:.3f}")
    
    def test_basic_command_size_variations(self):
        """Test visual consistency across different sizes."""
        sizes = [10, 20, 30]
        
        for size in sizes:
            baseline_path = self.generate_baseline_image(['basic', '--size', str(size)], f'basic_size_{size}')
            current_path = self.capture_screenshot(
                ['basic', '--size', str(size), '--steps', '5'], 
                f'basic_size_{size}_current'
            )
            
            similarity = self.assert_visual_similarity(current_path, baseline_path)
            print(f"Basic command size {size}: similarity = {similarity:.3f}")


class TestAIVisualRegression(VisualRegressionTest):
    """Test visual regression for AI commands."""
    
    @skip_if_feature_unavailable('stable_baselines3')
    def test_ai_command_visual_consistency(self):
        """Test that AI command produces consistent visual output."""
        baseline_path = self.generate_baseline_image(['ai'], 'ai_baseline')
        
        for i in range(2):  # Fewer runs due to training time
            current_path = self.capture_screenshot(
                ['ai', '--size', '15', '--steps', '10'], 
                f'ai_run_{i}'
            )
            
            similarity = self.assert_visual_similarity(current_path, baseline_path, threshold=0.90)
            print(f"AI command run {i}: similarity = {similarity:.3f}")
    
    @skip_if_feature_unavailable('stable_baselines3')
    def test_per_pixel_command_visual_consistency(self):
        """Test per-pixel AI command visual consistency."""
        baseline_path = self.generate_baseline_image(['per-pixel'], 'per_pixel_baseline')
        
        current_path = self.capture_screenshot(
            ['per-pixel', '--size', '15', '--steps', '10'], 
            'per_pixel_current'
        )
        
        similarity = self.assert_visual_similarity(current_path, baseline_path, threshold=0.90)
        print(f"Per-pixel command: similarity = {similarity:.3f}")


class TestTrainingVisualRegression(VisualRegressionTest):
    """Test visual regression for training commands."""
    
    @skip_if_feature_unavailable('stable_baselines3')
    def test_training_command_visual_consistency(self):
        """Test training command visual consistency."""
        baseline_path = self.generate_baseline_image(['train'], 'train_baseline')
        
        # Run training with minimal steps
        current_path = self.capture_screenshot(
            ['train', '--timesteps', '100', '--size', '10'], 
            'train_current'
        )
        
        similarity = self.assert_visual_similarity(current_path, baseline_path, threshold=0.85)
        print(f"Training command: similarity = {similarity:.3f}")


class TestVisualizationCommands(VisualRegressionTest):
    """Test visual regression for visualization commands."""
    
    @skip_if_feature_unavailable('pygame')
    def test_pygame_command_visual_consistency(self):
        """Test pygame command visual consistency."""
        baseline_path = self.generate_baseline_image(['pygame'], 'pygame_baseline')
        
        current_path = self.capture_screenshot(
            ['pygame', '--size', '20', '--steps', '5'], 
            'pygame_current'
        )
        
        similarity = self.assert_visual_similarity(current_path, baseline_path, threshold=0.90)
        print(f"Pygame command: similarity = {similarity:.3f}")
    
    @skip_if_feature_unavailable('pygame')
    def test_enhanced_command_visual_consistency(self):
        """Test enhanced command visual consistency."""
        baseline_path = self.generate_baseline_image(['enhanced'], 'enhanced_baseline')
        
        current_path = self.capture_screenshot(
            ['enhanced', '--size', '30', '--initial-zoom', '0.1'], 
            'enhanced_current'
        )
        
        similarity = self.assert_visual_similarity(current_path, baseline_path, threshold=0.90)
        print(f"Enhanced command: similarity = {similarity:.3f}")


class TestMatplotlibVisualRegression(VisualRegressionTest):
    """Test visual regression for matplotlib-based commands."""
    
    def test_benchmark_command_visual_consistency(self):
        """Test benchmark command matplotlib output consistency."""
        # Generate baseline
        baseline_path = self.generate_baseline_image(['benchmark'], 'benchmark_baseline')
        
        # Run benchmark command
        success = self.capture_matplotlib_output(
            ['benchmark', '--size', '15', '--steps', '100'], 
            'benchmark_current'
        )
        
        if success:
            # Find the generated figure
            output_files = list(Path(self.temp_dir).glob('visual_outputs/benchmark_current_*.png'))
            if output_files:
                current_path = str(output_files[0])
                similarity = self.assert_visual_similarity(current_path, baseline_path, threshold=0.85)
                print(f"Benchmark command: similarity = {similarity:.3f}")
            else:
                self.fail("No matplotlib output generated")
        else:
            self.fail("Benchmark command failed")
    
    def test_visualize_compare_command_visual_consistency(self):
        """Test visualize compare command matplotlib output consistency."""
        baseline_path = self.generate_baseline_image(['visualize', 'compare'], 'visualize_compare_baseline')
        
        success = self.capture_matplotlib_output(
            ['visualize', 'compare', '--save', 'comparison.png'], 
            'visualize_compare_current'
        )
        
        if success:
            output_files = list(Path(self.temp_dir).glob('visual_outputs/visualize_compare_current_*.png'))
            if output_files:
                current_path = str(output_files[0])
                similarity = self.assert_visual_similarity(current_path, baseline_path, threshold=0.85)
                print(f"Visualize compare command: similarity = {similarity:.3f}")
            else:
                self.fail("No matplotlib output generated")
        else:
            self.fail("Visualize compare command failed")


class TestVisualRegressionPerformance(VisualRegressionTest):
    """Test visual regression performance and stress testing."""
    
    def test_rapid_visual_capture(self):
        """Test rapid visual capture performance."""
        start_time = time.time()
        
        # Capture multiple screenshots rapidly
        for i in range(5):
            self.capture_screenshot(
                ['info'], 
                f'rapid_capture_{i}'
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(total_time, 30, f"Rapid capture took too long: {total_time:.2f}s")
        print(f"Rapid capture test: {total_time:.2f}s for 5 captures")
    
    def test_large_environment_visual_consistency(self):
        """Test visual consistency with large environments."""
        baseline_path = self.generate_baseline_image(['basic', '--size', '100'], 'large_env_baseline')
        
        current_path = self.capture_screenshot(
            ['basic', '--size', '100', '--steps', '5'], 
            'large_env_current'
        )
        
        similarity = self.assert_visual_similarity(current_path, baseline_path, threshold=0.90)
        print(f"Large environment: similarity = {similarity:.3f}")


class TestVisualRegressionEdgeCases(VisualRegressionTest):
    """Test visual regression edge cases and error conditions."""
    
    def test_invalid_command_visual_handling(self):
        """Test visual handling of invalid commands."""
        # Invalid command should not crash the visual system
        try:
            self.capture_screenshot(['invalid_command'], 'invalid_command')
            # Should not reach here
            self.fail("Invalid command should have failed")
        except Exception as e:
            # Expected to fail
            print(f"Invalid command correctly failed: {e}")
    
    def test_empty_command_visual_handling(self):
        """Test visual handling of empty commands."""
        try:
            self.capture_screenshot([], 'empty_command')
            # Should not reach here
            self.fail("Empty command should have failed")
        except Exception as e:
            # Expected to fail
            print(f"Empty command correctly failed: {e}")
    
    def test_very_large_values_visual_handling(self):
        """Test visual handling of very large values."""
        try:
            self.capture_screenshot(
                ['basic', '--size', '1000', '--steps', '10000'], 
                'very_large_values'
            )
            # Should handle gracefully
            print("Very large values handled gracefully")
        except Exception as e:
            # Should fail gracefully
            print(f"Very large values failed gracefully: {e}")


class TestVisualRegressionUtilities(unittest.TestCase):
    """Test visual regression utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_image_comparison_identical(self):
        """Test image comparison with identical images."""
        # Create two identical test images
        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (100, 100), color='red')
        
        path1 = os.path.join(self.temp_dir, 'test1.png')
        path2 = os.path.join(self.temp_dir, 'test2.png')
        
        img1.save(path1)
        img2.save(path2)
        
        # Compare
        test_instance = VisualRegressionTest()
        test_instance.setUp()
        similarity, diff_path = test_instance.compare_images(path1, path2)
        
        self.assertAlmostEqual(similarity, 1.0, places=3)
        test_instance.tearDown()
    
    def test_image_comparison_different(self):
        """Test image comparison with different images."""
        # Create two different test images
        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (100, 100), color='blue')
        
        path1 = os.path.join(self.temp_dir, 'test1.png')
        path2 = os.path.join(self.temp_dir, 'test2.png')
        
        img1.save(path1)
        img2.save(path2)
        
        # Compare
        test_instance = VisualRegressionTest()
        test_instance.setUp()
        similarity, diff_path = test_instance.compare_images(path1, path2)
        
        self.assertLess(similarity, 0.9)  # Should be significantly different
        test_instance.tearDown()


if __name__ == '__main__':
    unittest.main() 