#!/usr/bin/env python3
"""
Tests for the pixel_life CLI tool.
"""

import unittest
import subprocess
import sys
import os
import tempfile
import json
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pixel_life import main
from env import PixelLifeEnv


class TestPixelLifeCLI(unittest.TestCase):
    """Test the pixel_life CLI tool."""
    
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
    
    def test_basic_command(self):
        """Test the basic command."""
        # Test with minimal arguments
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'basic', 
            '--size', '10', '--steps', '5'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Running Basic Environment Demo", result.stdout)
    
    def test_info_command(self):
        """Test the info command."""
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'info', 
            '--size', '10'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Pixel Life System Information", result.stdout)
        self.assertIn("Action spaces:", result.stdout)
        self.assertIn("Main agent:", result.stdout)
        self.assertIn("Spice agent:", result.stdout)
    
    def test_config_generate(self):
        """Test config generation."""
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'config', 
            '--generate'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Generated pixel_life_config.json", result.stdout)
        
        # Check that config file was created
        config_path = os.path.join(self.temp_dir, "pixel_life_config.json")
        self.assertTrue(os.path.exists(config_path))
        
        # Check config content
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.assertIn("environment", config)
        self.assertIn("training", config)
        self.assertIn("rendering", config)
    
    def test_config_load(self):
        """Test config loading."""
        # First generate a config
        subprocess.run([
            sys.executable, '-m', 'pixel_life', 'config', 
            '--generate'
        ], capture_output=True, cwd=self.temp_dir)
        
        # Then load it
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'config', 
            '--load', 'pixel_life_config.json'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Loaded configuration from pixel_life_config.json", result.stdout)
    
    def test_benchmark_command(self):
        """Test the benchmark command."""
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'benchmark', 
            '--size', '10', '--steps', '100'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Running Performance Benchmark", result.stdout)
        self.assertIn("Benchmark Results:", result.stdout)
        self.assertIn("Steps per second:", result.stdout)
    
    def test_help_command(self):
        """Test help command."""
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', '--help'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Pixel Life: A 2D Artificial Life Environment", result.stdout)
        self.assertIn("Available modes", result.stdout)
    
    def test_invalid_command(self):
        """Test invalid command handling."""
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'invalid_command'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error:", result.stderr.lower())
    
    def test_evaluate_command_missing_model(self):
        """Test evaluate command with missing model."""
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'evaluate', 
            '--model-path', 'nonexistent_model.zip'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        # Should fail gracefully
        self.assertNotEqual(result.returncode, 0)
    
    def test_pygame_command(self):
        """Test pygame command (should work if pygame is available)."""
        try:
            import pygame
            result = subprocess.run([
                sys.executable, '-m', 'pixel_life', 'pygame', 
                '--size', '10', '--steps', '5'
            ], capture_output=True, text=True, cwd=self.temp_dir, timeout=10)
            
            # Should either succeed or fail gracefully
            self.assertIn("Running Pygame Demo", result.stdout)
        except ImportError:
            # Skip test if pygame not available
            self.skipTest("Pygame not available")
    
    def test_enhanced_command(self):
        """Test enhanced command (should work if pygame is available)."""
        try:
            import pygame
            result = subprocess.run([
                sys.executable, '-m', 'pixel_life', 'enhanced', 
                '--size', '20', '--initial-zoom', '0.1'
            ], capture_output=True, text=True, cwd=self.temp_dir, timeout=10)
            
            # Should either succeed or fail gracefully
            self.assertIn("Running Enhanced Pygame Demo", result.stdout)
        except ImportError:
            # Skip test if pygame not available
            self.skipTest("Pygame not available")


class TestEnvironmentCompatibility(unittest.TestCase):
    """Test environment compatibility with gym interfaces."""
    
    def test_action_space_exists(self):
        """Test that action_space attribute exists."""
        env = PixelLifeEnv(H=10, W=10)
        self.assertTrue(hasattr(env, 'action_space'))
        self.assertTrue(hasattr(env, 'spice_action_space'))
        self.assertTrue(hasattr(env, 'pixel_action_space'))
        self.assertTrue(hasattr(env, 'observation_space'))
    
    def test_step_returns_five_values(self):
        """Test that step method returns 5 values (gymnasium standard)."""
        env = PixelLifeEnv(H=10, W=10)
        obs, info = env.reset()
        
        # Create some actions
        spice_action = env.spice_action_space.sample()
        pixel_actions = {}
        for coord in env.live_pixels:
            pixel_actions[coord] = (0, 0)  # no-op action
        
        result = env.step(spice_action, pixel_actions)
        
        # Should return 5 values: obs, rewards, terminated, truncated, info
        self.assertEqual(len(result), 5)
        observations, rewards, terminated, truncated, info = result
        
        # Check types
        self.assertIsInstance(observations, tuple)
        self.assertIsInstance(rewards, tuple)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
    
    def test_reset_returns_two_values(self):
        """Test that reset method returns 2 values (gymnasium standard)."""
        env = PixelLifeEnv(H=10, W=10)
        result = env.reset()
        
        # Should return 2 values: obs, info
        self.assertEqual(len(result), 2)
        obs, info = result
        
        # Check types
        self.assertIsInstance(obs, tuple)
        self.assertIsInstance(info, dict)


class TestContinualLearningCompatibility(unittest.TestCase):
    """Test continual learning compatibility."""
    
    def test_continual_learning_imports(self):
        """Test that continual learning can be imported."""
        try:
            from continual_learning import ContinualLearningSystem
            self.assertTrue(True)  # Import successful
        except ImportError as e:
            self.fail(f"Failed to import ContinualLearningSystem: {e}")
    
    def test_continual_learning_initialization(self):
        """Test continual learning system initialization."""
        try:
            from continual_learning import ContinualLearningSystem
            
            # Should initialize without errors
            cl_system = ContinualLearningSystem(
                env_kwargs={'H': 10, 'W': 10},
                log_dir='./test_logs'
            )
            
            self.assertIsNotNone(cl_system)
            self.assertTrue(hasattr(cl_system, 'initialize_models'))
            
        except Exception as e:
            self.fail(f"Failed to initialize ContinualLearningSystem: {e}")


if __name__ == '__main__':
    unittest.main() 