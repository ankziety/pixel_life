#!/usr/bin/env python3
"""
Test script for the new logging system.
Demonstrates the logging manager functionality and CLI tools.
"""

import os
import sys
import time
from datetime import datetime

def test_logging_system():
    """Test the new logging system."""
    print("🧪 Testing Pixel Life Logging System")
    print("=" * 50)
    
    # Test 1: Initialize logging manager
    print("\n1. Testing LogManager initialization...")
    try:
        from log_manager import LogManager, ModelPerformance
        
        log_manager = LogManager("./test_logs")
        request_id = log_manager.start_request()
        print(f"✅ LogManager initialized successfully")
        print(f"   Request ID: {request_id}")
        print(f"   Database: {log_manager.db_path}")
        
    except ImportError as e:
        print(f"❌ Failed to import LogManager: {e}")
        return
    
    # Test 2: Basic logging
    print("\n2. Testing basic logging...")
    try:
        log_manager.log("INFO", "Test info message", module="test", function="test_logging_system")
        log_manager.log("WARNING", "Test warning message", module="test", function="test_logging_system")
        log_manager.log("ERROR", "Test error message", module="test", function="test_logging_system")
        print("✅ Basic logging successful")
        
    except Exception as e:
        print(f"❌ Basic logging failed: {e}")
    
    # Test 3: Model performance logging
    print("\n3. Testing model performance logging...")
    try:
        performance = ModelPerformance(
            model_path="./test_models/test_model.zip",
            timestamp=datetime.now().isoformat(),
            generation=1,
            episode_reward=85.5,
            survival_rate=0.92,
            training_steps=100000,
            model_size_mb=15.2,
            training_time_seconds=120.5,
            environment_size=30,
            hyperparameters={
                'learning_rate': 3e-4,
                'batch_size': 64,
                'n_epochs': 10
            },
            device='cpu',
            framework='stable-baselines3'
        )
        
        log_manager.log_model_performance(performance)
        print("✅ Model performance logging successful")
        
    except Exception as e:
        print(f"❌ Model performance logging failed: {e}")
    
    # Test 4: Log statistics
    print("\n4. Testing log statistics...")
    try:
        stats = log_manager.get_log_stats()
        print(f"✅ Log statistics retrieved:")
        print(f"   Total log files: {stats.total_log_files}")
        print(f"   Total size: {stats.total_size_mb:.2f} MB")
        print(f"   Performance models: {stats.performance_models}")
        
    except Exception as e:
        print(f"❌ Log statistics failed: {e}")
    
    # Test 5: Best model retrieval
    print("\n5. Testing best model retrieval...")
    try:
        best_model = log_manager.get_best_model()
        if best_model:
            print(f"✅ Best model found:")
            print(f"   Path: {best_model.model_path}")
            print(f"   Reward: {best_model.episode_reward:.2f}")
            print(f"   Generation: {best_model.generation}")
        else:
            print("ℹ️  No best model found (expected for test)")
            
    except Exception as e:
        print(f"❌ Best model retrieval failed: {e}")
    
    print("\n🎉 Logging system test completed!")


def test_cli_tools():
    """Test the CLI tools."""
    print("\n🔧 Testing CLI Tools")
    print("=" * 50)
    
    try:
        from pixel_logs import PixelLogManager
        
        log_manager = PixelLogManager("./test_logs")
        
        print("\n1. Testing overview...")
        log_manager.show_overview()
        
        print("\n2. Testing performance display...")
        log_manager.show_performance(limit=5)
        
        print("\n3. Testing error display...")
        log_manager.show_errors(limit=5)
        
        print("\n4. Testing cleanup (dry run)...")
        log_manager.cleanup_logs(days=1, dry_run=True)
        
        print("\n✅ CLI tools test completed!")
        
    except Exception as e:
        print(f"❌ CLI tools test failed: {e}")


def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    import shutil
    
    test_dirs = ["./test_logs", "./test_models"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"   Removed {test_dir}")


if __name__ == "__main__":
    print("Pixel Life Logging System Test")
    print("=" * 60)
    
    # Run tests
    test_logging_system()
    test_cli_tools()
    
    # Ask if user wants to clean up
    response = input("\nDo you want to clean up test files? (y/N): ")
    if response.lower() in ['y', 'yes']:
        cleanup_test_files()
    else:
        print("Test files preserved for inspection.")
    
    print("\n✨ Test completed!") 