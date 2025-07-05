#!/usr/bin/env python3
"""
Test runner for comprehensive Pixel Life CLI tests.
"""

import unittest
import sys
import os
import time
import argparse
import multiprocessing
import concurrent.futures
from pathlib import Path
from functools import partial

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_test_module(module_name, verbose=False):
    """Run a single test module and return results."""
    try:
        # Import the module
        module = __import__(module_name, fromlist=['*'])
        
        # Create test suite for the module
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=2 if verbose else 1,
            stream=open(os.devnull, 'w') if not verbose else sys.stdout
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        return {
            'module': module_name,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'time': end_time - start_time,
            'success': result.wasSuccessful(),
            'failures_list': result.failures,
            'errors_list': result.errors
        }
    except Exception as e:
        return {
            'module': module_name,
            'tests_run': 0,
            'failures': 0,
            'errors': 1,
            'skipped': 0,
            'time': 0,
            'success': False,
            'failures_list': [],
            'errors_list': [(f"ImportError: {e}", str(e))]
        }


def run_all_tests(verbose=False, pattern=None, timeout=None, parallel=False, max_workers=None):
    """Run all tests with optional filtering and parallel execution."""
    # Discover and load all test modules
    loader = unittest.TestLoader()
    
    if pattern:
        loader.testNamePatterns = [pattern]
    
    # Find all test files
    test_dir = Path(__file__).parent / 'tests'
    test_files = list(test_dir.glob('test_*.py'))
    
    if not test_files:
        print("No test files found")
        return False
    
    # Prepare module names
    module_names = [f"tests.{test_file.stem}" for test_file in test_files]
    
    if parallel:
        return run_tests_parallel(module_names, verbose, max_workers)
    else:
        return run_tests_sequential(module_names, verbose)


def run_tests_sequential(module_names, verbose=False):
    """Run tests sequentially."""
    # Create test suite
    suite = unittest.TestSuite()
    
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=['*'])
            tests = unittest.TestLoader().loadTestsFromModule(module)
            suite.addTests(tests)
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        stream=sys.stdout
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    print_summary(result, end_time - start_time)
    return result.wasSuccessful()


def run_tests_parallel(module_names, verbose=False, max_workers=None):
    """Run tests in parallel."""
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(module_names))
    
    print(f"🚀 Running tests in parallel with {max_workers} workers...")
    
    # Run tests in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all test modules
        future_to_module = {
            executor.submit(run_test_module, module_name, verbose): module_name 
            for module_name in module_names
        }
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(future_to_module):
            module_name = future_to_module[future]
            try:
                result = future.result()
                results.append(result)
                
                if verbose:
                    status = "✅" if result['success'] else "❌"
                    print(f"{status} {module_name}: {result['tests_run']} tests, "
                          f"{result['failures']} failures, {result['errors']} errors "
                          f"({result['time']:.2f}s)")
                
            except Exception as e:
                print(f"❌ {module_name}: Exception occurred - {e}")
                results.append({
                    'module': module_name,
                    'tests_run': 0,
                    'failures': 0,
                    'errors': 1,
                    'skipped': 0,
                    'time': 0,
                    'success': False,
                    'failures_list': [],
                    'errors_list': [(f"Exception: {e}", str(e))]
                })
    
    # Aggregate results
    total_tests = sum(r['tests_run'] for r in results)
    total_failures = sum(r['failures'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    total_skipped = sum(r['skipped'] for r in results)
    total_time = sum(r['time'] for r in results)
    all_successful = all(r['success'] for r in results)
    
    # Print summary
    print("\n" + "="*60)
    print("PARALLEL TEST SUMMARY")
    print("="*60)
    print(f"Modules tested: {len(results)}")
    print(f"Tests run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Skipped: {total_skipped}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Parallel speedup: {total_time / max(1, len(module_names)):.1f}x")
    
    # Print failures and errors
    if total_failures > 0 or total_errors > 0:
        print("\nFAILURES AND ERRORS:")
        for result in results:
            if result['failures'] > 0 or result['errors'] > 0:
                print(f"\n{result['module']}:")
                for failure in result['failures_list']:
                    print(f"  FAILURE: {failure[0]}")
                    print(f"    {failure[1]}")
                for error in result['errors_list']:
                    print(f"  ERROR: {error[0]}")
                    print(f"    {error[1]}")
    
    return all_successful


def print_summary(result, elapsed_time):
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)


def run_specific_test_category(category, verbose=False, parallel=False, max_workers=None):
    """Run tests for a specific category."""
    categories = {
        'basic': 'test_basic_command*',
        'ai': 'test_ai_command*',
        'training': 'test_train_command*',
        'logs': 'test_logs_*',
        'models': 'test_models_*',
        'experiments': 'test_experiments_*',
        'monitor': 'test_monitor_*',
        'visualize': 'test_visualize_*',
        'workflow': 'test_workflow_*',
        'debug': 'test_debug_*',
        'help': 'test_help_*',
        'edge': 'test_invalid_*',
        'performance': 'test_rapid_*',
        'comprehensive': 'test_comprehensive_cli*',
        'visual': 'test_visual_*'
    }
    
    if category not in categories:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(categories.keys())}")
        return False
    
    return run_all_tests(verbose=verbose, pattern=categories[category], 
                        parallel=parallel, max_workers=max_workers)


def main():
    parser = argparse.ArgumentParser(description='Run Pixel Life CLI tests')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--category', '-c', type=str,
                       help='Run tests for specific category')
    parser.add_argument('--pattern', '-p', type=str,
                       help='Run tests matching pattern')
    parser.add_argument('--timeout', '-t', type=int,
                       help='Timeout for individual tests (seconds)')
    parser.add_argument('--list-categories', action='store_true',
                       help='List available test categories')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--max-workers', type=int,
                       help='Maximum number of parallel workers')
    parser.add_argument('--sequential', action='store_true',
                       help='Force sequential execution (overrides --parallel)')
    
    args = parser.parse_args()
    
    if args.list_categories:
        categories = [
            'basic', 'ai', 'training', 'logs', 'models', 'experiments',
            'monitor', 'visualize', 'workflow', 'debug', 'help', 'edge',
            'performance', 'comprehensive', 'visual'
        ]
        print("Available test categories:")
        for cat in categories:
            print(f"  {cat}")
        return
    
    # Determine if we should run in parallel
    parallel = args.parallel and not args.sequential
    
    if args.category:
        success = run_specific_test_category(
            args.category, args.verbose, parallel, args.max_workers
        )
    else:
        success = run_all_tests(
            verbose=args.verbose, pattern=args.pattern, 
            timeout=args.timeout, parallel=parallel, max_workers=args.max_workers
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 