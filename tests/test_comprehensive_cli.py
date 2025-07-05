#!/usr/bin/env python3
"""
Comprehensive test suite for the pixel_life CLI tool.
Tests all commands, subcommands, and options systematically.
"""

import unittest
import subprocess
import sys
import os
import tempfile
import json
import time
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pixel_life import main, run_basic_demo, run_info, run_config, run_benchmark
from pixel_life import run_logs_overview, run_models_list, run_experiments_list
from pixel_life import run_monitor_status, run_visualize_compare, run_help_tutorial


class TestComprehensiveCLI(unittest.TestCase):
    """Comprehensive test suite for all CLI commands and options."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create necessary directories
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./models', exist_ok=True)
        
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def run_cli_command(self, args, expect_success=True, timeout=30):
        """Helper method to run CLI commands and check results."""
        cmd = [sys.executable, '-m', 'pixel_life'] + args
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=self.temp_dir,
            timeout=timeout
        )
        
        if expect_success:
            self.assertEqual(result.returncode, 0, 
                           f"Command failed: {' '.join(cmd)}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        else:
            self.assertNotEqual(result.returncode, 0, 
                              f"Command should have failed: {' '.join(cmd)}")
        
        return result
    
    def test_main_help(self):
        """Test main help command."""
        result = self.run_cli_command(['--help'])
        self.assertIn("Pixel Life: A 2D Artificial Life Environment", result.stdout)
        self.assertIn("Available modes", result.stdout)
        self.assertIn("basic", result.stdout)
        self.assertIn("train", result.stdout)
        self.assertIn("info", result.stdout)
    
    def test_no_mode_specified(self):
        """Test behavior when no mode is specified."""
        result = self.run_cli_command([], expect_success=False)
        # Should show help when no mode specified
    
    # ============================================================================
    # BASIC COMMAND TESTS
    # ============================================================================
    
    def test_basic_command_minimal(self):
        """Test basic command with minimal arguments."""
        result = self.run_cli_command(['basic'])
        self.assertIn("Running Basic Environment Demo", result.stdout)
        self.assertIn("Final state:", result.stdout)
    
    def test_basic_command_all_options(self):
        """Test basic command with all options."""
        result = self.run_cli_command([
            'basic', '--size', '15', '--steps', '10', '--render'
        ])
        self.assertIn("Running Basic Environment Demo", result.stdout)
        self.assertIn("Final state:", result.stdout)
    
    def test_basic_command_large_size(self):
        """Test basic command with large environment size."""
        result = self.run_cli_command([
            'basic', '--size', '50', '--steps', '5'
        ])
        self.assertIn("Running Basic Environment Demo", result.stdout)
    
    def test_basic_command_many_steps(self):
        """Test basic command with many steps."""
        result = self.run_cli_command([
            'basic', '--size', '10', '--steps', '100'
        ])
        self.assertIn("Running Basic Environment Demo", result.stdout)
    
    def test_basic_command_accelerated(self):
        """Test basic command with Apple Silicon acceleration (if available)."""
        try:
            from apple_acceleration import create_accelerated_env
            result = self.run_cli_command([
                'basic', '--size', '20', '--steps', '10', '--accelerated'
            ])
            self.assertIn("Running Basic Environment Demo", result.stdout)
        except ImportError:
            # Skip if acceleration not available
            pass
    
    # ============================================================================
    # AI COMMAND TESTS
    # ============================================================================
    
    def test_ai_command_minimal(self):
        """Test AI command with minimal arguments."""
        result = self.run_cli_command(['ai'])
        self.assertIn("Running AI Agent Demo", result.stdout)
    
    def test_ai_command_all_options(self):
        """Test AI command with all options."""
        result = self.run_cli_command([
            'ai', '--size', '15', '--steps', '10', '--render'
        ])
        self.assertIn("Running AI Agent Demo", result.stdout)
    
    def test_ai_command_accelerated(self):
        """Test AI command with Apple Silicon acceleration."""
        try:
            from apple_acceleration import create_accelerated_env
            result = self.run_cli_command([
                'ai', '--size', '20', '--steps', '10', '--accelerated'
            ])
            self.assertIn("Running AI Agent Demo", result.stdout)
        except ImportError:
            pass
    
    # ============================================================================
    # PER-PIXEL COMMAND TESTS
    # ============================================================================
    
    def test_per_pixel_command_minimal(self):
        """Test per-pixel command with minimal arguments."""
        result = self.run_cli_command(['per-pixel'])
        self.assertIn("Running Per-Pixel AI Demo", result.stdout)
    
    def test_per_pixel_command_all_options(self):
        """Test per-pixel command with all options."""
        result = self.run_cli_command([
            'per-pixel', '--size', '15', '--steps', '10', '--render', 
            '--train', '--generations', '2'
        ])
        self.assertIn("Running Per-Pixel AI Demo", result.stdout)
    
    def test_per_pixel_command_training_only(self):
        """Test per-pixel command with training only."""
        result = self.run_cli_command([
            'per-pixel', '--size', '10', '--steps', '5', '--train', 
            '--generations', '1'
        ])
        self.assertIn("Running Per-Pixel AI Demo", result.stdout)
    
    # ============================================================================
    # CONTINUAL LEARNING COMMAND TESTS
    # ============================================================================
    
    def test_continual_command_minimal(self):
        """Test continual learning command with minimal arguments."""
        result = self.run_cli_command(['continual'])
        self.assertIn("Running Continual Learning Demo", result.stdout)
    
    def test_continual_command_all_options(self):
        """Test continual learning command with all options."""
        result = self.run_cli_command([
            'continual', '--size', '15', '--steps', '10', '--render',
            '--train', '--episodes', '2'
        ])
        self.assertIn("Running Continual Learning Demo", result.stdout)
    
    # ============================================================================
    # PYGAME COMMAND TESTS
    # ============================================================================
    
    def test_pygame_command_minimal(self):
        """Test pygame command with minimal arguments."""
        try:
            import pygame
            result = self.run_cli_command(['pygame'], timeout=5)
            self.assertIn("Running Pygame Demo", result.stdout)
        except ImportError:
            self.skipTest("Pygame not available")
    
    def test_pygame_command_all_options(self):
        """Test pygame command with all options."""
        try:
            import pygame
            result = self.run_cli_command([
                'pygame', '--size', '15', '--steps', '5'
            ], timeout=5)
            self.assertIn("Running Pygame Demo", result.stdout)
        except ImportError:
            self.skipTest("Pygame not available")
    
    # ============================================================================
    # ENHANCED COMMAND TESTS
    # ============================================================================
    
    def test_enhanced_command_minimal(self):
        """Test enhanced command with minimal arguments."""
        try:
            import pygame
            result = self.run_cli_command(['enhanced'], timeout=5)
            self.assertIn("Running Enhanced Pygame Demo", result.stdout)
        except ImportError:
            self.skipTest("Pygame not available")
    
    def test_enhanced_command_all_options(self):
        """Test enhanced command with all options."""
        try:
            import pygame
            result = self.run_cli_command([
                'enhanced', '--size', '50', '--initial-zoom', '0.05'
            ], timeout=5)
            self.assertIn("Running Enhanced Pygame Demo", result.stdout)
        except ImportError:
            self.skipTest("Pygame not available")
    
    # ============================================================================
    # ACCELERATED COMMAND TESTS
    # ============================================================================
    
    def test_accelerated_command_minimal(self):
        """Test accelerated command with minimal arguments."""
        try:
            from apple_acceleration import create_accelerated_env
            result = self.run_cli_command(['accelerated'], timeout=5)
            self.assertIn("Running Apple Silicon Accelerated Demo", result.stdout)
        except ImportError:
            self.skipTest("Apple acceleration not available")
    
    def test_accelerated_command_all_options(self):
        """Test accelerated command with all options."""
        try:
            from apple_acceleration import create_accelerated_env
            result = self.run_cli_command([
                'accelerated', '--size', '30', '--steps', '10'
            ], timeout=5)
            self.assertIn("Running Apple Silicon Accelerated Demo", result.stdout)
        except ImportError:
            self.skipTest("Apple acceleration not available")
    
    def test_benchmark_demo_command(self):
        """Test benchmark demo command."""
        try:
            from apple_acceleration import create_accelerated_env
            result = self.run_cli_command(['benchmark-demo'])
            self.assertIn("Running Performance Benchmark Demo", result.stdout)
        except ImportError:
            self.skipTest("Apple acceleration not available")
    
    # ============================================================================
    # TRAINING COMMAND TESTS
    # ============================================================================
    
    def test_train_command_minimal(self):
        """Test training command with minimal arguments."""
        result = self.run_cli_command([
            'train', '--timesteps', '1000'
        ])
        self.assertIn("Running Full Training Session", result.stdout)
    
    def test_train_command_all_options(self):
        """Test training command with all options."""
        result = self.run_cli_command([
            'train', '--size', '20', '--timesteps', '1000', '--n-envs', '2',
            '--learning-rate', '1e-4', '--n-steps', '1024', '--batch-size', '32',
            '--n-epochs', '5', '--gamma', '0.95', '--device', 'cpu',
            '--no-tensorboard'
        ])
        self.assertIn("Running Full Training Session", result.stdout)
    
    def test_train_command_accelerated(self):
        """Test training command with Apple Silicon acceleration."""
        try:
            from apple_acceleration import create_accelerated_env
            result = self.run_cli_command([
                'train', '--timesteps', '1000', '--accelerated'
            ])
            self.assertIn("Running Full Training Session", result.stdout)
        except ImportError:
            pass
    
    # ============================================================================
    # EVALUATION COMMAND TESTS
    # ============================================================================
    
    def test_evaluate_command_missing_model(self):
        """Test evaluate command with missing model."""
        result = self.run_cli_command([
            'evaluate', '--model-path', 'nonexistent.zip'
        ], expect_success=False)
        self.assertIn("Error loading model", result.stdout)
    
    def test_evaluate_command_all_options(self):
        """Test evaluate command with all options."""
        result = self.run_cli_command([
            'evaluate', '--model-path', 'nonexistent.zip', '--size', '20',
            '--episodes', '3', '--steps', '50', '--render'
        ], expect_success=False)
        self.assertIn("Error loading model", result.stdout)
    
    # ============================================================================
    # INFO COMMAND TESTS
    # ============================================================================
    
    def test_info_command_minimal(self):
        """Test info command with minimal arguments."""
        result = self.run_cli_command(['info'])
        self.assertIn("Pixel Life System Information", result.stdout)
        self.assertIn("Action spaces:", result.stdout)
    
    def test_info_command_with_size(self):
        """Test info command with size option."""
        result = self.run_cli_command(['info', '--size', '25'])
        self.assertIn("Pixel Life System Information", result.stdout)
    
    # ============================================================================
    # BENCHMARK COMMAND TESTS
    # ============================================================================
    
    def test_benchmark_command_minimal(self):
        """Test benchmark command with minimal arguments."""
        result = self.run_cli_command(['benchmark'])
        self.assertIn("Running Performance Benchmark", result.stdout)
        self.assertIn("Benchmark Results:", result.stdout)
    
    def test_benchmark_command_all_options(self):
        """Test benchmark command with all options."""
        result = self.run_cli_command([
            'benchmark', '--size', '25', '--steps', '500'
        ])
        self.assertIn("Running Performance Benchmark", result.stdout)
        self.assertIn("Benchmark Results:", result.stdout)
    
    # ============================================================================
    # CONFIG COMMAND TESTS
    # ============================================================================
    
    def test_config_generate(self):
        """Test config generation."""
        result = self.run_cli_command(['config', '--generate'])
        self.assertIn("Generated pixel_life_config.json", result.stdout)
        
        # Check config file exists and is valid JSON
        config_path = os.path.join(self.temp_dir, "pixel_life_config.json")
        self.assertTrue(os.path.exists(config_path))
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.assertIn("environment", config)
        self.assertIn("training", config)
        self.assertIn("rendering", config)
    
    def test_config_load_existing(self):
        """Test config loading with existing file."""
        # First generate a config
        self.run_cli_command(['config', '--generate'])
        
        # Then load it
        result = self.run_cli_command(['config', '--load', 'pixel_life_config.json'])
        self.assertIn("Loaded configuration from pixel_life_config.json", result.stdout)
    
    def test_config_load_nonexistent(self):
        """Test config loading with nonexistent file."""
        result = self.run_cli_command([
            'config', '--load', 'nonexistent.json'
        ])
        self.assertIn("Use --generate to create a config file", result.stdout)
    
    def test_config_no_options(self):
        """Test config command with no options."""
        result = self.run_cli_command(['config'])
        self.assertIn("Use --generate to create a config file", result.stdout)
    
    # ============================================================================
    # LOGS COMMAND TESTS
    # ============================================================================
    
    def test_logs_overview(self):
        """Test logs overview command."""
        result = self.run_cli_command(['logs', 'overview'])
        # Should not fail even if no logs exist
        self.assertIn("Log Overview", result.stdout)
    
    def test_logs_search(self):
        """Test logs search command."""
        result = self.run_cli_command([
            'logs', 'search', 'test_query'
        ])
        # Should not fail even if no logs exist
        self.assertIn("Search Results", result.stdout)
    
    def test_logs_search_with_options(self):
        """Test logs search command with all options."""
        result = self.run_cli_command([
            'logs', 'search', 'test_query', '--level', 'INFO', '--limit', '5'
        ])
        self.assertIn("Search Results", result.stdout)
    
    def test_logs_performance(self):
        """Test logs performance command."""
        result = self.run_cli_command(['logs', 'performance'])
        self.assertIn("Performance Metrics", result.stdout)
    
    def test_logs_performance_with_limit(self):
        """Test logs performance command with limit."""
        result = self.run_cli_command([
            'logs', 'performance', '--limit', '3'
        ])
        self.assertIn("Performance Metrics", result.stdout)
    
    def test_logs_errors(self):
        """Test logs errors command."""
        result = self.run_cli_command(['logs', 'errors'])
        self.assertIn("Error Logs", result.stdout)
    
    def test_logs_errors_with_limit(self):
        """Test logs errors command with limit."""
        result = self.run_cli_command([
            'logs', 'errors', '--limit', '5'
        ])
        self.assertIn("Error Logs", result.stdout)
    
    def test_logs_cleanup_dry_run(self):
        """Test logs cleanup command with dry run."""
        result = self.run_cli_command([
            'logs', 'cleanup', '--days', '7'
        ])
        self.assertIn("Cleanup Summary", result.stdout)
    
    def test_logs_cleanup_execute(self):
        """Test logs cleanup command with execute."""
        result = self.run_cli_command([
            'logs', 'cleanup', '--days', '30', '--execute'
        ])
        self.assertIn("Cleanup Summary", result.stdout)
    
    # ============================================================================
    # MODELS COMMAND TESTS
    # ============================================================================
    
    def test_models_list(self):
        """Test models list command."""
        result = self.run_cli_command(['models', 'list'])
        self.assertIn("Registered Models", result.stdout)
    
    def test_models_list_with_tags(self):
        """Test models list command with tags filter."""
        result = self.run_cli_command([
            'models', 'list', '--tags', 'test,training'
        ])
        self.assertIn("Registered Models", result.stdout)
    
    def test_models_register_missing_file(self):
        """Test models register command with missing file."""
        result = self.run_cli_command([
            'models', 'register', '--model-path', 'nonexistent.zip',
            '--name', 'test_model'
        ], expect_success=False)
        self.assertIn("Failed to register model", result.stdout)
    
    def test_models_info_nonexistent(self):
        """Test models info command with nonexistent model."""
        result = self.run_cli_command([
            'models', 'info', 'nonexistent_id'
        ])
        self.assertIn("not found", result.stdout)
    
    def test_models_delete_nonexistent(self):
        """Test models delete command with nonexistent model."""
        result = self.run_cli_command([
            'models', 'delete', 'nonexistent_id'
        ], expect_success=False)
        self.assertIn("Failed to delete model", result.stdout)
    
    # ============================================================================
    # EXPERIMENTS COMMAND TESTS
    # ============================================================================
    
    def test_experiments_list(self):
        """Test experiments list command."""
        result = self.run_cli_command(['experiments', 'list'])
        self.assertIn("Experiments", result.stdout)
    
    def test_experiments_list_with_filters(self):
        """Test experiments list command with filters."""
        result = self.run_cli_command([
            'experiments', 'list', '--status', 'completed', '--tags', 'test'
        ])
        self.assertIn("Experiments", result.stdout)
    
    def test_experiments_create_minimal(self):
        """Test experiments create command with minimal arguments."""
        result = self.run_cli_command([
            'experiments', 'create', '--name', 'test_experiment'
        ])
        self.assertIn("Experiment created successfully", result.stdout)
    
    def test_experiments_create_all_options(self):
        """Test experiments create command with all options."""
        result = self.run_cli_command([
            'experiments', 'create', '--name', 'test_experiment_2',
            '--description', 'Test experiment description',
            '--hyperparams', 'learning_rate=0.001', 'batch_size=64',
            '--tags', 'test,training', '--parent', 'parent_id'
        ])
        self.assertIn("Experiment created successfully", result.stdout)
    
    def test_experiments_info_nonexistent(self):
        """Test experiments info command with nonexistent experiment."""
        result = self.run_cli_command([
            'experiments', 'info', 'nonexistent_id'
        ])
        self.assertIn("not found", result.stdout)
    
    # ============================================================================
    # MONITOR COMMAND TESTS
    # ============================================================================
    
    def test_monitor_status(self):
        """Test monitor status command."""
        result = self.run_cli_command(['monitor', 'status'])
        self.assertIn("System Status", result.stdout)
        self.assertIn("CPU Usage:", result.stdout)
        self.assertIn("Memory Usage:", result.stdout)
    
    def test_monitor_start_timeout(self):
        """Test monitor start command with timeout."""
        try:
            result = self.run_cli_command([
                'monitor', 'start', '--interval', '1'
            ], timeout=3)
            self.assertIn("System monitoring started", result.stdout)
        except subprocess.TimeoutExpired:
            # Expected to timeout as monitoring runs continuously
            pass
    
    # ============================================================================
    # VISUALIZE COMMAND TESTS
    # ============================================================================
    
    def test_visualize_compare(self):
        """Test visualize compare command."""
        result = self.run_cli_command(['visualize', 'compare'])
        self.assertIn("No data available for comparison", result.stdout)
    
    def test_visualize_compare_with_options(self):
        """Test visualize compare command with options."""
        result = self.run_cli_command([
            'visualize', 'compare', '--show', '--save', 'comparison.png'
        ])
        self.assertIn("No data available for comparison", result.stdout)
    
    # ============================================================================
    # WORKFLOW COMMAND TESTS
    # ============================================================================
    
    def test_workflow_batch_train_multiple(self):
        """Test workflow batch train-multiple command."""
        result = self.run_cli_command([
            'workflow', 'batch', '--workflow', 'train-multiple'
        ])
        self.assertIn("Starting batch processing", result.stdout)
    
    def test_workflow_batch_evaluate_all(self):
        """Test workflow batch evaluate-all command."""
        result = self.run_cli_command([
            'workflow', 'batch', '--workflow', 'evaluate-all'
        ])
        self.assertIn("Starting batch processing", result.stdout)
    
    def test_workflow_batch_invalid(self):
        """Test workflow batch command with invalid workflow."""
        result = self.run_cli_command([
            'workflow', 'batch', '--workflow', 'invalid_workflow'
        ])
        self.assertIn("Unknown workflow", result.stdout)
    
    # ============================================================================
    # DEBUG COMMAND TESTS
    # ============================================================================
    
    def test_debug_profile_basic(self):
        """Test debug profile basic command."""
        result = self.run_cli_command([
            'debug', 'profile', '--command', 'basic', '--size', '10', '--steps', '50'
        ])
        self.assertIn("Profiling Results", result.stdout)
    
    def test_debug_profile_training(self):
        """Test debug profile training command."""
        result = self.run_cli_command([
            'debug', 'profile', '--command', 'training', '--size', '10', '--steps', '1000'
        ])
        self.assertIn("Profiling Results", result.stdout)
    
    def test_debug_profile_with_save(self):
        """Test debug profile command with save option."""
        result = self.run_cli_command([
            'debug', 'profile', '--command', 'basic', '--size', '10', '--steps', '50',
            '--save', 'profile.stats'
        ])
        self.assertIn("Profiling Results", result.stdout)
    
    def test_debug_profile_invalid_command(self):
        """Test debug profile command with invalid command."""
        result = self.run_cli_command([
            'debug', 'profile', '--command', 'invalid_command'
        ])
        self.assertIn("Unknown debug command", result.stdout)
    
    # ============================================================================
    # HELP COMMAND TESTS
    # ============================================================================
    
    def test_help_basic(self):
        """Test help basic command."""
        result = self.run_cli_command(['help', 'basic'])
        self.assertIn("Basic Tutorial", result.stdout)
        self.assertIn("Getting Started", result.stdout)
    
    def test_help_training(self):
        """Test help training command."""
        result = self.run_cli_command(['help', 'training'])
        self.assertIn("Training Tutorial", result.stdout)
        self.assertIn("Training AI Models", result.stdout)
    
    def test_help_experiments(self):
        """Test help experiments command."""
        result = self.run_cli_command(['help', 'experiments'])
        self.assertIn("Experiments Tutorial", result.stdout)
        self.assertIn("Managing Experiments", result.stdout)
    
    def test_help_monitoring(self):
        """Test help monitoring command."""
        result = self.run_cli_command(['help', 'monitoring'])
        self.assertIn("Monitoring Tutorial", result.stdout)
        self.assertIn("System Monitoring", result.stdout)
    
    def test_help_interactive(self):
        """Test help command with interactive option."""
        result = self.run_cli_command([
            'help', 'basic', '--interactive'
        ])
        self.assertIn("Basic Tutorial", result.stdout)
    
    # ============================================================================
    # EDGE CASES AND ERROR HANDLING TESTS
    # ============================================================================
    
    def test_invalid_command(self):
        """Test invalid command handling."""
        result = self.run_cli_command(['invalid_command'], expect_success=False)
        self.assertIn("error:", result.stderr.lower())
    
    def test_invalid_subcommand(self):
        """Test invalid subcommand handling."""
        result = self.run_cli_command(['logs', 'invalid_subcommand'], expect_success=False)
        self.assertIn("error:", result.stderr.lower())
    
    def test_missing_required_argument(self):
        """Test missing required argument handling."""
        result = self.run_cli_command(['evaluate'], expect_success=False)
        self.assertIn("error:", result.stderr.lower())
    
    def test_invalid_argument_type(self):
        """Test invalid argument type handling."""
        result = self.run_cli_command([
            'basic', '--size', 'invalid_size'
        ], expect_success=False)
        self.assertIn("error:", result.stderr.lower())
    
    def test_negative_values(self):
        """Test negative value handling."""
        result = self.run_cli_command([
            'basic', '--size', '-10', '--steps', '-5'
        ], expect_success=False)
        self.assertIn("error:", result.stderr.lower())
    
    def test_zero_values(self):
        """Test zero value handling."""
        result = self.run_cli_command([
            'basic', '--size', '0', '--steps', '0'
        ], expect_success=False)
        self.assertIn("error:", result.stderr.lower())
    
    def test_very_large_values(self):
        """Test very large value handling."""
        result = self.run_cli_command([
            'basic', '--size', '1000', '--steps', '1000000'
        ])
        # Should handle large values gracefully
        self.assertIn("Running Basic Environment Demo", result.stdout)
    
    # ============================================================================
    # COMBINATION TESTS
    # ============================================================================
    
    def test_multiple_commands_sequence(self):
        """Test running multiple commands in sequence."""
        # Run basic demo
        result1 = self.run_cli_command(['basic', '--size', '10', '--steps', '5'])
        self.assertIn("Running Basic Environment Demo", result1.stdout)
        
        # Generate config
        result2 = self.run_cli_command(['config', '--generate'])
        self.assertIn("Generated pixel_life_config.json", result2.stdout)
        
        # Check info
        result3 = self.run_cli_command(['info', '--size', '10'])
        self.assertIn("Pixel Life System Information", result3.stdout)
        
        # Run benchmark
        result4 = self.run_cli_command(['benchmark', '--size', '10', '--steps', '100'])
        self.assertIn("Running Performance Benchmark", result4.stdout)
    
    def test_all_log_commands_sequence(self):
        """Test running all log commands in sequence."""
        commands = [
            ['logs', 'overview'],
            ['logs', 'search', 'test'],
            ['logs', 'performance'],
            ['logs', 'errors'],
            ['logs', 'cleanup', '--days', '30']
        ]
        
        for cmd in commands:
            result = self.run_cli_command(cmd)
            # Each command should complete without error
    
    def test_all_model_commands_sequence(self):
        """Test running all model commands in sequence."""
        commands = [
            ['models', 'list'],
            ['models', 'list', '--tags', 'test'],
            ['models', 'info', 'nonexistent'],
            ['models', 'delete', 'nonexistent']
        ]
        
        for cmd in commands:
            result = self.run_cli_command(cmd)
            # Each command should complete without error
    
    def test_all_experiment_commands_sequence(self):
        """Test running all experiment commands in sequence."""
        # Create an experiment first
        result1 = self.run_cli_command([
            'experiments', 'create', '--name', 'test_sequence'
        ])
        self.assertIn("Experiment created successfully", result1.stdout)
        
        # Then run other commands
        commands = [
            ['experiments', 'list'],
            ['experiments', 'list', '--status', 'running'],
            ['experiments', 'info', 'nonexistent']
        ]
        
        for cmd in commands:
            result = self.run_cli_command(cmd)
            # Each command should complete without error
    
    # ============================================================================
    # PERFORMANCE AND STRESS TESTS
    # ============================================================================
    
    def test_rapid_command_execution(self):
        """Test rapid execution of multiple commands."""
        commands = [
            ['info'],
            ['config', '--generate'],
            ['benchmark', '--size', '10', '--steps', '50'],
            ['logs', 'overview'],
            ['monitor', 'status']
        ]
        
        start_time = time.time()
        for cmd in commands:
            result = self.run_cli_command(cmd)
        end_time = time.time()
        
        # Should complete all commands in reasonable time
        execution_time = end_time - start_time
        self.assertLess(execution_time, 30, f"Commands took too long: {execution_time}s")
    
    def test_memory_usage_under_load(self):
        """Test memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run multiple commands
        for i in range(10):
            self.run_cli_command(['info'])
            self.run_cli_command(['benchmark', '--size', '10', '--steps', '10'])
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024, 
                       f"Memory usage increased too much: {memory_increase / (1024*1024):.1f}MB")


class TestCLIFunctionCalls(unittest.TestCase):
    """Test CLI functions directly without subprocess."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        os.makedirs('./logs', exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_run_basic_demo_function(self):
        """Test run_basic_demo function directly."""
        args = MagicMock()
        args.size = 10
        args.steps = 5
        args.render = False
        args.accelerated = False
        
        # Should not raise any exceptions
        run_basic_demo(args)
    
    def test_run_info_function(self):
        """Test run_info function directly."""
        args = MagicMock()
        args.size = 10
        
        # Should not raise any exceptions
        run_info(args)
    
    def test_run_config_generate_function(self):
        """Test run_config function with generate."""
        args = MagicMock()
        args.generate = True
        args.load = None
        
        run_config(args)
        
        # Check config file was created
        config_path = os.path.join(self.temp_dir, "pixel_life_config.json")
        self.assertTrue(os.path.exists(config_path))
    
    def test_run_benchmark_function(self):
        """Test run_benchmark function directly."""
        args = MagicMock()
        args.size = 10
        args.steps = 100
        
        # Should not raise any exceptions
        run_benchmark(args)
    
    def test_run_logs_overview_function(self):
        """Test run_logs_overview function directly."""
        args = MagicMock()
        
        # Should not raise any exceptions
        run_logs_overview(args)
    
    def test_run_models_list_function(self):
        """Test run_models_list function directly."""
        args = MagicMock()
        args.tags = None
        
        # Should not raise any exceptions
        run_models_list(args)
    
    def test_run_experiments_list_function(self):
        """Test run_experiments_list function directly."""
        args = MagicMock()
        args.status = None
        args.tags = None
        
        # Should not raise any exceptions
        run_experiments_list(args)
    
    def test_run_monitor_status_function(self):
        """Test run_monitor_status function directly."""
        args = MagicMock()
        
        # Should not raise any exceptions
        run_monitor_status(args)
    
    def test_run_visualize_compare_function(self):
        """Test run_visualize_compare function directly."""
        args = MagicMock()
        args.save = None
        args.show = False
        
        # Should not raise any exceptions
        run_visualize_compare(args)
    
    def test_run_help_tutorial_function(self):
        """Test run_help_tutorial function directly."""
        args = MagicMock()
        args.topic = 'basic'
        args.interactive = False
        
        # Should not raise any exceptions
        run_help_tutorial(args)


if __name__ == '__main__':
    unittest.main() 