#!/usr/bin/env python3
"""
Test configuration and utilities for comprehensive CLI testing.
"""

import os
import tempfile
import shutil
from pathlib import Path

# Test configuration
TEST_CONFIG = {
    # Environment sizes for testing
    'SIZES': [5, 10, 15, 20, 30, 50],
    
    # Step counts for testing
    'STEPS': [5, 10, 50, 100, 500],
    
    # Training parameters for testing
    'TRAINING_TIMESTEPS': [100, 1000, 5000],
    'TRAINING_N_ENVS': [1, 2, 4],
    'TRAINING_LEARNING_RATES': [1e-4, 3e-4, 1e-3],
    
    # Timeouts for different command types
    'TIMEOUTS': {
        'basic': 30,
        'ai': 60,
        'training': 300,
        'pygame': 10,
        'enhanced': 10,
        'accelerated': 10,
        'monitor': 5,
        'default': 30
    },
    
    # Test categories and their priorities
    'CATEGORIES': {
        'core': ['basic', 'info', 'benchmark', 'config'],
        'ai': ['ai', 'per-pixel', 'continual', 'train', 'evaluate'],
        'visualization': ['pygame', 'enhanced', 'accelerated'],
        'management': ['logs', 'models', 'experiments'],
        'monitoring': ['monitor', 'visualize'],
        'automation': ['workflow', 'debug'],
        'help': ['help']
    },
    
    # Expected output patterns for validation
    'EXPECTED_OUTPUTS': {
        'basic': ['Running Basic Environment Demo', 'Final state:'],
        'ai': ['Running AI Agent Demo'],
        'per-pixel': ['Running Per-Pixel AI Demo'],
        'continual': ['Running Continual Learning Demo'],
        'pygame': ['Running Pygame Demo'],
        'enhanced': ['Running Enhanced Pygame Demo'],
        'accelerated': ['Running Apple Silicon Accelerated Demo'],
        'train': ['Running Full Training Session'],
        'info': ['Pixel Life System Information', 'Action spaces:'],
        'benchmark': ['Running Performance Benchmark', 'Benchmark Results:'],
        'config': ['Generated pixel_life_config.json'],
        'logs': ['Log Overview', 'Search Results', 'Performance Metrics'],
        'models': ['Registered Models'],
        'experiments': ['Experiments'],
        'monitor': ['System Status', 'CPU Usage:'],
        'visualize': ['No data available for comparison'],
        'workflow': ['Starting batch processing'],
        'debug': ['Profiling Results'],
        'help': ['Tutorial', 'Getting Started']
    },
    
    # Error patterns to check for
    'ERROR_PATTERNS': {
        'missing_model': ['Error loading model', 'Failed to register model'],
        'invalid_command': ['error:', 'invalid'],
        'missing_argument': ['error:', 'required'],
        'timeout': ['TimeoutExpired', 'timeout'],
        'import_error': ['ImportError', 'ModuleNotFoundError']
    }
}

class TestEnvironment:
    """Test environment setup and teardown utilities."""
    
    def __init__(self, temp_dir=None):
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for testing."""
        directories = [
            './logs',
            './models', 
            './logs/training_runs',
            './logs/experiments',
            './logs/performance'
        ]
        
        for directory in directories:
            os.makedirs(os.path.join(self.temp_dir, directory), exist_ok=True)
    
    def cleanup(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_file(self, filename, content=""):
        """Create a test file in the temp directory."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath
    
    def create_test_log(self, logname, entries=None):
        """Create a test log file."""
        if entries is None:
            entries = [
                "2024-01-01 10:00:00 INFO Test log entry 1",
                "2024-01-01 10:01:00 WARNING Test warning",
                "2024-01-01 10:02:00 ERROR Test error"
            ]
        
        logpath = os.path.join(self.temp_dir, 'logs', f'{logname}.log')
        with open(logpath, 'w') as f:
            for entry in entries:
                f.write(entry + '\n')
        return logpath
    
    def create_test_model(self, model_id, name="test_model"):
        """Create a test model file."""
        modelpath = os.path.join(self.temp_dir, 'models', f'{model_id}_{name}.zip')
        # Create a dummy zip file
        import zipfile
        with zipfile.ZipFile(modelpath, 'w') as zf:
            zf.writestr('model.pkl', 'dummy model data')
        return modelpath
    
    def create_test_experiment(self, exp_id, name="test_experiment"):
        """Create a test experiment file."""
        exp_data = {
            "experiment_id": exp_id,
            "name": name,
            "description": "Test experiment",
            "status": "completed",
            "created_at": "2024-01-01T10:00:00",
            "hyperparameters": {"learning_rate": 0.001, "batch_size": 64},
            "results": {"final_reward": 100.0, "training_steps": 1000}
        }
        
        exppath = os.path.join(self.temp_dir, 'logs', 'experiments', f'{exp_id}.json')
        import json
        with open(exppath, 'w') as f:
            json.dump(exp_data, f, indent=2)
        return exppath

class TestDataGenerator:
    """Generate test data for comprehensive testing."""
    
    @staticmethod
    def generate_command_combinations():
        """Generate all possible command combinations for testing."""
        combinations = []
        
        # Basic command combinations
        basic_commands = [
            ['basic'],
            ['basic', '--size', '10'],
            ['basic', '--steps', '50'],
            ['basic', '--render'],
            ['basic', '--size', '20', '--steps', '100', '--render']
        ]
        combinations.extend(basic_commands)
        
        # AI command combinations
        ai_commands = [
            ['ai'],
            ['ai', '--size', '15'],
            ['ai', '--steps', '75'],
            ['ai', '--render'],
            ['ai', '--size', '25', '--steps', '150', '--render']
        ]
        combinations.extend(ai_commands)
        
        # Training command combinations
        training_commands = [
            ['train', '--timesteps', '1000'],
            ['train', '--size', '20', '--timesteps', '2000'],
            ['train', '--n-envs', '2', '--timesteps', '1000'],
            ['train', '--learning-rate', '1e-4', '--timesteps', '1000'],
            ['train', '--size', '15', '--n-envs', '1', '--timesteps', '500', '--no-tensorboard']
        ]
        combinations.extend(training_commands)
        
        # Logs command combinations
        logs_commands = [
            ['logs', 'overview'],
            ['logs', 'search', 'test'],
            ['logs', 'search', 'error', '--level', 'ERROR'],
            ['logs', 'performance', '--limit', '5'],
            ['logs', 'errors', '--limit', '10'],
            ['logs', 'cleanup', '--days', '30'],
            ['logs', 'cleanup', '--days', '7', '--execute']
        ]
        combinations.extend(logs_commands)
        
        # Models command combinations
        models_commands = [
            ['models', 'list'],
            ['models', 'list', '--tags', 'test'],
            ['models', 'info', 'test_id'],
            ['models', 'delete', 'test_id']
        ]
        combinations.extend(models_commands)
        
        # Experiments command combinations
        experiments_commands = [
            ['experiments', 'list'],
            ['experiments', 'list', '--status', 'completed'],
            ['experiments', 'create', '--name', 'test_exp'],
            ['experiments', 'create', '--name', 'test_exp2', '--description', 'Test experiment'],
            ['experiments', 'info', 'test_id']
        ]
        combinations.extend(experiments_commands)
        
        return combinations
    
    @staticmethod
    def generate_edge_cases():
        """Generate edge case command combinations."""
        edge_cases = [
            # Invalid commands
            ['invalid_command'],
            ['logs', 'invalid_subcommand'],
            
            # Missing required arguments
            ['evaluate'],
            ['models', 'register'],
            ['experiments', 'create'],
            
            # Invalid argument types
            ['basic', '--size', 'invalid'],
            ['basic', '--steps', 'not_a_number'],
            ['train', '--timesteps', 'invalid'],
            
            # Negative values
            ['basic', '--size', '-10'],
            ['basic', '--steps', '-5'],
            ['train', '--timesteps', '-1000'],
            
            # Zero values
            ['basic', '--size', '0'],
            ['basic', '--steps', '0'],
            ['train', '--timesteps', '0'],
            
            # Very large values
            ['basic', '--size', '10000'],
            ['basic', '--steps', '1000000'],
            ['train', '--timesteps', '10000000'],
            
            # Empty strings
            ['experiments', 'create', '--name', ''],
            ['models', 'register', '--name', ''],
            
            # Special characters
            ['experiments', 'create', '--name', 'test@#$%'],
            ['models', 'register', '--name', 'test with spaces'],
        ]
        
        return edge_cases
    
    @staticmethod
    def generate_performance_tests():
        """Generate performance test combinations."""
        performance_tests = [
            # Rapid command execution
            ['info'],
            ['benchmark', '--size', '10', '--steps', '50'],
            ['config', '--generate'],
            ['logs', 'overview'],
            ['monitor', 'status'],
            
            # Large environment tests
            ['basic', '--size', '100', '--steps', '10'],
            ['benchmark', '--size', '100', '--steps', '100'],
            
            # Many steps tests
            ['basic', '--size', '10', '--steps', '1000'],
            ['benchmark', '--size', '10', '--steps', '5000'],
        ]
        
        return performance_tests

def validate_output(output, expected_patterns, error_patterns=None):
    """Validate command output against expected patterns."""
    if error_patterns is None:
        error_patterns = []
    
    # Check for expected patterns
    for pattern in expected_patterns:
        if pattern not in output:
            return False, f"Expected pattern '{pattern}' not found in output"
    
    # Check for error patterns
    for pattern in error_patterns:
        if pattern.lower() in output.lower():
            return False, f"Error pattern '{pattern}' found in output"
    
    return True, "Output validation passed"

def get_timeout_for_command(command):
    """Get appropriate timeout for a command."""
    if not command:
        return TEST_CONFIG['TIMEOUTS']['default']
    
    cmd = command[0] if isinstance(command, list) else command
    
    for category, timeout in TEST_CONFIG['TIMEOUTS'].items():
        if category in cmd:
            return timeout
    
    return TEST_CONFIG['TIMEOUTS']['default']

def is_optional_feature_available(feature):
    """Check if an optional feature is available."""
    optional_features = {
        'pygame': 'pygame',
        'apple_acceleration': 'apple_acceleration',
        'tensorboard': 'tensorboard',
        'stable_baselines3': 'stable_baselines3'
    }
    
    if feature not in optional_features:
        return True  # Assume available if not in optional list
    
    try:
        __import__(optional_features[feature])
        return True
    except ImportError:
        return False

def skip_if_feature_unavailable(feature):
    """Decorator to skip tests if feature is unavailable."""
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            if not is_optional_feature_available(feature):
                import unittest
                raise unittest.SkipTest(f"{feature} not available")
            return test_func(*args, **kwargs)
        return wrapper
    return decorator 