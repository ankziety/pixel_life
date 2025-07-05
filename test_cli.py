#!/usr/bin/env python3
"""
Comprehensive tests for the enhanced Pixel Life CLI.
Tests all new features including model management, experiment tracking, monitoring, etc.
"""

import unittest
import subprocess
import sys
import os
import tempfile
import json
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the classes directly from the module
import pixel_life
ModelManager = pixel_life.ModelManager
ExperimentManager = pixel_life.ExperimentManager
SystemMonitor = pixel_life.SystemMonitor


class TestEnhancedCLI(unittest.TestCase):
    """Test the enhanced CLI features."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create test directories
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./experiments', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_models_management(self):
        """Test model management features."""
        print("\n🔧 Testing Model Management")
        print("=" * 50)
        
        # Test model manager initialization
        model_manager = ModelManager('./models')
        self.assertIsNotNone(model_manager)
        
        # Test model registration (with a dummy model file)
        dummy_model_path = './test_model.zip'
        with open(dummy_model_path, 'w') as f:
            f.write('dummy model content')
        
        try:
            model_id = model_manager.register_model(
                model_path=dummy_model_path,
                name="test-model",
                description="Test model for CLI testing",
                author="test-user",
                tags=["test", "cli"]
            )
            self.assertIsNotNone(model_id)
            print(f"✅ Model registered with ID: {model_id}")
            
            # Test listing models
            models = model_manager.list_models()
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0].name, "test-model")
            print(f"✅ Found {len(models)} registered models")
            
            # Test getting model info
            model = model_manager.get_model(model_id)
            self.assertIsNotNone(model)
            self.assertEqual(model.name, "test-model")
            print(f"✅ Retrieved model info for {model.name}")
            
            # Test updating model
            model_manager.update_model(model_id, description="Updated description")
            updated_model = model_manager.get_model(model_id)
            self.assertEqual(updated_model.description, "Updated description")
            print("✅ Model updated successfully")
            
            # Test deleting model
            model_manager.delete_model(model_id)
            models_after_delete = model_manager.list_models()
            self.assertEqual(len(models_after_delete), 0)
            print("✅ Model deleted successfully")
            
        finally:
            if os.path.exists(dummy_model_path):
                os.remove(dummy_model_path)
    
    def test_experiments_management(self):
        """Test experiment management features."""
        print("\n🧪 Testing Experiment Management")
        print("=" * 50)
        
        # Test experiment manager initialization
        exp_manager = ExperimentManager('./experiments')
        self.assertIsNotNone(exp_manager)
        
        # Test experiment creation
        experiment_id = exp_manager.create_experiment(
            name="test-experiment",
            description="Test experiment for CLI testing",
            hyperparameters={"learning_rate": 0.001, "batch_size": 64},
            tags=["test", "experiment"]
        )
        self.assertIsNotNone(experiment_id)
        print(f"✅ Experiment created with ID: {experiment_id}")
        
        # Test listing experiments
        experiments = exp_manager.list_experiments()
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0].name, "test-experiment")
        print(f"✅ Found {len(experiments)} experiments")
        
        # Test getting experiment info
        exp = exp_manager.get_experiment(experiment_id)
        self.assertIsNotNone(exp)
        self.assertEqual(exp.name, "test-experiment")
        print(f"✅ Retrieved experiment info for {exp.name}")
        
        # Test updating experiment
        exp_manager.update_experiment(experiment_id, status="completed", results={"reward": 85.5})
        updated_exp = exp_manager.get_experiment(experiment_id)
        self.assertEqual(updated_exp.status, "completed")
        self.assertEqual(updated_exp.results["reward"], 85.5)
        print("✅ Experiment updated successfully")
        
        # Test filtering experiments
        completed_exps = exp_manager.list_experiments(status="completed")
        self.assertEqual(len(completed_exps), 1)
        print("✅ Experiment filtering works")
    
    def test_system_monitoring(self):
        """Test system monitoring features."""
        print("\n🖥️  Testing System Monitoring")
        print("=" * 50)
        
        # Test system monitor initialization
        monitor = SystemMonitor()
        self.assertIsNotNone(monitor)
        
        # Test current metrics collection
        metrics = monitor.get_current_metrics()
        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(metrics.cpu_percent, 0)
        self.assertLessEqual(metrics.cpu_percent, 100)
        self.assertGreaterEqual(metrics.memory_percent, 0)
        self.assertLessEqual(metrics.memory_percent, 100)
        print(f"✅ Current metrics: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%")
        
        # Test monitoring start/stop
        monitor.start_monitoring(interval=0.1)
        time.sleep(0.3)  # Let it collect some metrics
        monitor.stop_monitoring()
        
        recent_metrics = monitor.get_recent_metrics()
        self.assertGreater(len(recent_metrics), 0)
        print(f"✅ Collected {len(recent_metrics)} metrics during monitoring")
    
    def test_cli_commands(self):
        """Test CLI commands via subprocess."""
        print("\n💻 Testing CLI Commands")
        print("=" * 50)
        
        # Test help command
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', '--help'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Pixel Life: A 2D Artificial Life Environment", result.stdout)
        self.assertIn("models", result.stdout)
        self.assertIn("experiments", result.stdout)
        self.assertIn("monitor", result.stdout)
        print("✅ Help command shows new features")
        
        # Test models list command
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'models', 'list'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("No models registered", result.stdout)
        print("✅ Models list command works")
        
        # Test experiments list command
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'experiments', 'list'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("No experiments found", result.stdout)
        print("✅ Experiments list command works")
        
        # Test monitor status command
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'monitor', 'status'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("System Status", result.stdout)
        self.assertIn("CPU Usage", result.stdout)
        self.assertIn("Memory Usage", result.stdout)
        print("✅ Monitor status command works")
        
        # Test help tutorial command
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'help', 'basic'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Basic Tutorial", result.stdout)
        print("✅ Help tutorial command works")
    
    def test_workflow_commands(self):
        """Test workflow and batch processing commands."""
        print("\n🔄 Testing Workflow Commands")
        print("=" * 50)
        
        # Test workflow batch command (should show available workflows)
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'workflow', 'batch', '--help'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("train-multiple", result.stdout)
        self.assertIn("evaluate-all", result.stdout)
        print("✅ Workflow batch command help works")
        
        # Test debug profile command help
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'debug', 'profile', '--help'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Profile performance", result.stdout)
        print("✅ Debug profile command help works")
    
    def test_visualization_commands(self):
        """Test visualization commands."""
        print("\n📊 Testing Visualization Commands")
        print("=" * 50)
        
        # Test visualize compare command help
        result = subprocess.run([
            sys.executable, '-m', 'pixel_life', 'visualize', 'compare', '--help'
        ], capture_output=True, text=True, cwd=self.temp_dir)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Compare multiple models", result.stdout)
        print("✅ Visualize compare command help works")
    
    def test_integration(self):
        """Test integration between different features."""
        print("\n🔗 Testing Feature Integration")
        print("=" * 50)
        
        # Create a model and experiment, then test their interaction
        model_manager = ModelManager('./models')
        exp_manager = ExperimentManager('./experiments')
        
        # Create dummy model
        dummy_model_path = './test_integration_model.zip'
        with open(dummy_model_path, 'w') as f:
            f.write('integration test model')
        
        try:
            # Register model
            model_id = model_manager.register_model(
                model_path=dummy_model_path,
                name="integration-model",
                description="Model for integration testing",
                tags=["integration", "test"]
            )
            
            # Create experiment
            exp_id = exp_manager.create_experiment(
                name="integration-experiment",
                description="Experiment for integration testing",
                hyperparameters={"model_id": model_id},
                tags=["integration", "test"]
            )
            
            # Update experiment with model results
            exp_manager.update_experiment(exp_id, status="completed", results={
                "model_id": model_id,
                "final_reward": 95.5,
                "training_steps": 10000
            })
            
            # Update model with performance metrics
            model_manager.update_model(model_id, performance_metrics={
                "experiment_id": exp_id,
                "final_reward": 95.5
            })
            
            # Verify integration
            model = model_manager.get_model(model_id)
            exp = exp_manager.get_experiment(exp_id)
            
            self.assertEqual(model.performance_metrics["experiment_id"], exp_id)
            self.assertEqual(exp.results["model_id"], model_id)
            self.assertEqual(model.performance_metrics["final_reward"], exp.results["final_reward"])
            
            print("✅ Model and experiment integration works")
            
        finally:
            if os.path.exists(dummy_model_path):
                os.remove(dummy_model_path)


def run_comprehensive_test():
    """Run comprehensive CLI test suite."""
    print("🚀 Starting Comprehensive Pixel Life CLI Test Suite")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedCLI)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed! Enhanced CLI features are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 