#!/usr/bin/env python3
"""
Visual regression test runner for Pixel Life CLI.
Specialized runner for visual regression testing with screenshot comparison.
"""

import unittest
import sys
import os
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_visual_tests(verbose=False, category=None, generate_baselines=False, update_baselines=False):
    """Run visual regression tests."""
    # Import visual test module
    try:
        from tests.test_visual_regression import VisualRegressionTest
    except ImportError as e:
        print(f"Error importing visual regression tests: {e}")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    
    if category:
        # Run specific category
        if category == 'basic':
            from tests.test_visual_regression import TestBasicVisualRegression
            suite = loader.loadTestsFromTestCase(TestBasicVisualRegression)
        elif category == 'ai':
            from tests.test_visual_regression import TestAIVisualRegression
            suite = loader.loadTestsFromTestCase(TestAIVisualRegression)
        elif category == 'training':
            from tests.test_visual_regression import TestTrainingVisualRegression
            suite = loader.loadTestsFromTestCase(TestTrainingVisualRegression)
        elif category == 'visualization':
            from tests.test_visual_regression import TestVisualizationCommands
            suite = loader.loadTestsFromTestCase(TestVisualizationCommands)
        elif category == 'matplotlib':
            from tests.test_visual_regression import TestMatplotlibVisualRegression
            suite = loader.loadTestsFromTestCase(TestMatplotlibVisualRegression)
        elif category == 'performance':
            from tests.test_visual_regression import TestVisualRegressionPerformance
            suite = loader.loadTestsFromTestCase(TestVisualRegressionPerformance)
        elif category == 'edge':
            from tests.test_visual_regression import TestVisualRegressionEdgeCases
            suite = loader.loadTestsFromTestCase(TestVisualRegressionEdgeCases)
        elif category == 'utilities':
            from tests.test_visual_regression import TestVisualRegressionUtilities
            suite = loader.loadTestsFromTestCase(TestVisualRegressionUtilities)
        else:
            print(f"Unknown visual test category: {category}")
            return False
    else:
        # Run all visual tests
        suite = loader.loadTestsFromName('tests.test_visual_regression')
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        stream=sys.stdout
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print visual test summary
    print("\n" + "="*60)
    print("VISUAL REGRESSION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    if result.failures:
        print("\nVISUAL FAILURES:")
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)
    
    if result.errors:
        print("\nVISUAL ERRORS:")
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)
    
    return result.wasSuccessful()


def generate_visual_baselines():
    """Generate baseline images for all visual tests."""
    print("🎨 Generating visual baselines...")
    
    # Create baseline directory
    baseline_dir = Path("tests/visual_baselines")
    baseline_dir.mkdir(exist_ok=True)
    
    # Define baseline configurations
    baseline_configs = [
        {'command': ['basic'], 'name': 'basic_baseline'},
        {'command': ['basic', '--size', '10'], 'name': 'basic_size_10'},
        {'command': ['basic', '--size', '20'], 'name': 'basic_size_20'},
        {'command': ['basic', '--size', '30'], 'name': 'basic_size_30'},
        {'command': ['ai'], 'name': 'ai_baseline'},
        {'command': ['per-pixel'], 'name': 'per_pixel_baseline'},
        {'command': ['train'], 'name': 'train_baseline'},
        {'command': ['pygame'], 'name': 'pygame_baseline'},
        {'command': ['enhanced'], 'name': 'enhanced_baseline'},
        {'command': ['benchmark'], 'name': 'benchmark_baseline'},
        {'command': ['visualize', 'compare'], 'name': 'visualize_compare_baseline'},
        {'command': ['basic', '--size', '100'], 'name': 'large_env_baseline'},
    ]
    
    # Import test class to use its baseline generation
    from tests.test_visual_regression import VisualRegressionTest
    
    test_instance = VisualRegressionTest()
    test_instance.setUp()
    
    generated_baselines = []
    
    for config in baseline_configs:
        try:
            baseline_path = test_instance.generate_baseline_image(
                config['command'], config['name']
            )
            generated_baselines.append({
                'command': config['command'],
                'name': config['name'],
                'path': baseline_path,
                'status': 'success'
            })
            print(f"✅ Generated baseline: {config['name']}")
        except Exception as e:
            generated_baselines.append({
                'command': config['command'],
                'name': config['name'],
                'path': None,
                'status': 'failed',
                'error': str(e)
            })
            print(f"❌ Failed to generate baseline: {config['name']} - {e}")
    
    test_instance.tearDown()
    
    # Save baseline metadata
    baseline_metadata = {
        'generated_at': datetime.now().isoformat(),
        'baselines': generated_baselines
    }
    
    with open(baseline_dir / 'baseline_metadata.json', 'w') as f:
        json.dump(baseline_metadata, f, indent=2)
    
    print(f"\n🎨 Generated {len([b for b in generated_baselines if b['status'] == 'success'])} baselines")
    return generated_baselines


def update_visual_baselines():
    """Update existing visual baselines."""
    print("🔄 Updating visual baselines...")
    
    # First generate new baselines
    new_baselines = generate_visual_baselines()
    
    # Backup old baselines
    baseline_dir = Path("tests/visual_baselines")
    backup_dir = Path("tests/visual_baselines_backup")
    
    if baseline_dir.exists():
        import shutil
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(baseline_dir, backup_dir)
        print(f"📦 Backed up old baselines to {backup_dir}")
    
    print("✅ Visual baselines updated")


def analyze_visual_coverage():
    """Analyze visual test coverage."""
    print("📊 Analyzing visual test coverage...")
    
    # Define all visual test categories
    visual_categories = [
        'basic', 'ai', 'training', 'visualization', 
        'matplotlib', 'performance', 'edge', 'utilities'
    ]
    
    coverage_data = {
        'total_categories': len(visual_categories),
        'tested_categories': [],
        'untested_categories': [],
        'coverage_percentage': 0
    }
    
    for category in visual_categories:
        try:
            # Try to run a test from this category
            success = run_visual_tests(category=category, verbose=False)
            if success:
                coverage_data['tested_categories'].append(category)
            else:
                coverage_data['untested_categories'].append(category)
        except Exception as e:
            coverage_data['untested_categories'].append(category)
    
    coverage_data['coverage_percentage'] = (
        len(coverage_data['tested_categories']) / len(visual_categories) * 100
    )
    
    # Print coverage report
    print("\n" + "="*60)
    print("VISUAL TEST COVERAGE ANALYSIS")
    print("="*60)
    print(f"Total categories: {coverage_data['total_categories']}")
    print(f"Tested categories: {len(coverage_data['tested_categories'])}")
    print(f"Untested categories: {len(coverage_data['untested_categories'])}")
    print(f"Coverage: {coverage_data['coverage_percentage']:.1f}%")
    
    if coverage_data['tested_categories']:
        print(f"\n✅ Tested categories: {', '.join(coverage_data['tested_categories'])}")
    
    if coverage_data['untested_categories']:
        print(f"\n❌ Untested categories: {', '.join(coverage_data['untested_categories'])}")
    
    return coverage_data


def main():
    parser = argparse.ArgumentParser(description='Run Pixel Life visual regression tests')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--category', '-c', type=str,
                       help='Run tests for specific visual category')
    parser.add_argument('--generate-baselines', action='store_true',
                       help='Generate baseline images')
    parser.add_argument('--update-baselines', action='store_true',
                       help='Update existing baseline images')
    parser.add_argument('--analyze-coverage', action='store_true',
                       help='Analyze visual test coverage')
    parser.add_argument('--list-categories', action='store_true',
                       help='List available visual test categories')
    
    args = parser.parse_args()
    
    if args.list_categories:
        categories = [
            'basic', 'ai', 'training', 'visualization', 
            'matplotlib', 'performance', 'edge', 'utilities'
        ]
        print("Available visual test categories:")
        for cat in categories:
            print(f"  {cat}")
        return
    
    if args.generate_baselines:
        generate_visual_baselines()
        return
    
    if args.update_baselines:
        update_visual_baselines()
        return
    
    if args.analyze_coverage:
        analyze_visual_coverage()
        return
    
    # Run visual tests
    success = run_visual_tests(
        verbose=args.verbose,
        category=args.category,
        generate_baselines=args.generate_baselines,
        update_baselines=args.update_baselines
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 