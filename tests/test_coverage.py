#!/usr/bin/env python3
"""
Test coverage analysis for Pixel Life CLI commands.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CLICoverageAnalyzer:
    """Analyze test coverage for CLI commands and options."""
    
    def __init__(self):
        self.coverage_data = {
            'commands': defaultdict(dict),
            'options': defaultdict(set),
            'subcommands': defaultdict(set),
            'test_files': [],
            'total_tests': 0,
            'covered_commands': set(),
            'uncovered_commands': set()
        }
        
        # Define all CLI commands and their options
        self.cli_structure = {
            'basic': {
                'options': ['--size', '--steps', '--render', '--accelerated'],
                'subcommands': []
            },
            'ai': {
                'options': ['--size', '--steps', '--render', '--accelerated'],
                'subcommands': []
            },
            'per-pixel': {
                'options': ['--size', '--steps', '--render', '--train', '--generations'],
                'subcommands': []
            },
            'continual': {
                'options': ['--size', '--steps', '--render', '--train', '--episodes'],
                'subcommands': []
            },
            'pygame': {
                'options': ['--size', '--steps'],
                'subcommands': []
            },
            'enhanced': {
                'options': ['--size', '--initial-zoom'],
                'subcommands': []
            },
            'accelerated': {
                'options': ['--size', '--steps'],
                'subcommands': []
            },
            'benchmark-demo': {
                'options': ['--size'],
                'subcommands': []
            },
            'train': {
                'options': ['--size', '--timesteps', '--n-envs', '--learning-rate', 
                           '--n-steps', '--batch-size', '--n-epochs', '--gamma', 
                           '--device', '--no-tensorboard', '--accelerated'],
                'subcommands': []
            },
            'evaluate': {
                'options': ['--model-path', '--size', '--episodes', '--steps', '--render'],
                'subcommands': []
            },
            'info': {
                'options': ['--size'],
                'subcommands': []
            },
            'benchmark': {
                'options': ['--size', '--steps'],
                'subcommands': []
            },
            'config': {
                'options': ['--generate', '--load'],
                'subcommands': []
            },
            'logs': {
                'options': [],
                'subcommands': ['overview', 'search', 'performance', 'errors', 'cleanup']
            },
            'models': {
                'options': [],
                'subcommands': ['list', 'register', 'info', 'delete']
            },
            'experiments': {
                'options': [],
                'subcommands': ['list', 'create', 'info']
            },
            'monitor': {
                'options': [],
                'subcommands': ['start', 'status']
            },
            'visualize': {
                'options': [],
                'subcommands': ['compare']
            },
            'workflow': {
                'options': [],
                'subcommands': ['batch']
            },
            'debug': {
                'options': [],
                'subcommands': ['profile']
            },
            'help': {
                'options': [],
                'subcommands': ['basic', 'training', 'experiments', 'monitoring']
            }
        }
        
        # Define subcommand options
        self.subcommand_options = {
            'logs': {
                'overview': [],
                'search': ['--level', '--limit'],
                'performance': ['--limit'],
                'errors': ['--limit'],
                'cleanup': ['--days', '--execute']
            },
            'models': {
                'list': ['--tags'],
                'register': ['--model-path', '--name', '--description', '--author', '--tags'],
                'info': [],
                'delete': []
            },
            'experiments': {
                'list': ['--status', '--tags'],
                'create': ['--name', '--description', '--hyperparams', '--parent', '--tags'],
                'info': []
            },
            'monitor': {
                'start': ['--interval'],
                'status': []
            },
            'visualize': {
                'compare': ['--show', '--save']
            },
            'workflow': {
                'batch': ['--workflow']
            },
            'debug': {
                'profile': ['--command', '--size', '--steps', '--save']
            },
            'help': {
                'basic': ['--interactive'],
                'training': ['--interactive'],
                'experiments': ['--interactive'],
                'monitoring': ['--interactive']
            }
        }
    
    def analyze_test_file(self, test_file_path):
        """Analyze a test file for CLI command coverage."""
        with open(test_file_path, 'r') as f:
            content = f.read()
        
        # Extract test methods
        import re
        test_methods = re.findall(r'def (test_.*?)\(', content)
        
        # Analyze each test method
        for method in test_methods:
            self.analyze_test_method(content, method)
    
    def analyze_test_method(self, content, method_name):
        """Analyze a specific test method for CLI coverage."""
        # Find the test method content
        import re
        method_pattern = rf'def {method_name}\(.*?\):(.*?)(?=def |$)'
        match = re.search(method_pattern, content, re.DOTALL)
        
        if not match:
            return
        
        method_content = match.group(1)
        
        # Look for CLI command patterns
        self.extract_cli_commands(method_content, method_name)
    
    def extract_cli_commands(self, content, test_name):
        """Extract CLI commands from test content."""
        # Look for command patterns like ['command', '--option', 'value']
        import re
        
        # Pattern for command arrays
        command_pattern = r'\[([^\]]*?)\]'
        matches = re.findall(command_pattern, content)
        
        for match in matches:
            # Parse the command array
            try:
                # Simple parsing - split by comma and strip quotes
                parts = [part.strip().strip("'\"") for part in match.split(',')]
                parts = [part for part in parts if part]
                
                if not parts:
                    continue
                
                command = parts[0]
                options = parts[1:]
                
                self.record_command_coverage(command, options, test_name)
                
            except Exception as e:
                print(f"Error parsing command in {test_name}: {e}")
    
    def record_command_coverage(self, command, options, test_name):
        """Record coverage for a specific command and options."""
        self.coverage_data['covered_commands'].add(command)
        
        if command not in self.coverage_data['commands']:
            self.coverage_data['commands'][command] = {
                'tests': set(),
                'options_tested': set(),
                'subcommands_tested': set()
            }
        
        self.coverage_data['commands'][command]['tests'].add(test_name)
        
        # Record options
        for i, option in enumerate(options):
            if option.startswith('--'):
                self.coverage_data['commands'][command]['options_tested'].add(option)
                self.coverage_data['options'][command].add(option)
            elif i > 0 and options[i-1].startswith('--'):
                # This is a value for the previous option
                continue
            else:
                # This might be a subcommand
                self.coverage_data['commands'][command]['subcommands_tested'].add(option)
                self.coverage_data['subcommands'][command].add(option)
    
    def generate_coverage_report(self):
        """Generate a comprehensive coverage report."""
        # Calculate coverage statistics
        total_commands = len(self.cli_structure)
        covered_commands = len(self.coverage_data['covered_commands'])
        uncovered_commands = total_commands - covered_commands
        
        # Find uncovered commands
        self.coverage_data['uncovered_commands'] = set(self.cli_structure.keys()) - self.coverage_data['covered_commands']
        
        # Calculate option coverage
        total_options = 0
        covered_options = 0
        
        for command, structure in self.cli_structure.items():
            total_options += len(structure['options'])
            if command in self.coverage_data['commands']:
                covered_options += len(self.coverage_data['commands'][command]['options_tested'])
        
        # Calculate subcommand coverage
        total_subcommands = 0
        covered_subcommands = 0
        
        for command, structure in self.cli_structure.items():
            total_subcommands += len(structure['subcommands'])
            if command in self.cli_structure and command in self.coverage_data['subcommands']:
                covered_subcommands += len(self.coverage_data['subcommands'][command])
        
        # Generate report
        report = {
            'summary': {
                'total_commands': total_commands,
                'covered_commands': covered_commands,
                'uncovered_commands': uncovered_commands,
                'command_coverage_percentage': (covered_commands / total_commands * 100) if total_commands > 0 else 0,
                'total_options': total_options,
                'covered_options': covered_options,
                'option_coverage_percentage': (covered_options / total_options * 100) if total_options > 0 else 0,
                'total_subcommands': total_subcommands,
                'covered_subcommands': covered_subcommands,
                'subcommand_coverage_percentage': (covered_subcommands / total_subcommands * 100) if total_subcommands > 0 else 0
            },
            'command_details': {},
            'uncovered_commands': list(self.coverage_data['uncovered_commands']),
            'recommendations': []
        }
        
        # Generate command details
        for command, structure in self.cli_structure.items():
            command_data = {
                'tested': command in self.coverage_data['covered_commands'],
                'tests': list(self.coverage_data['commands'].get(command, {}).get('tests', [])),
                'options': {
                    'total': structure['options'],
                    'tested': list(self.coverage_data['commands'].get(command, {}).get('options_tested', [])),
                    'untested': [opt for opt in structure['options'] 
                               if opt not in self.coverage_data['commands'].get(command, {}).get('options_tested', [])]
                },
                'subcommands': {
                    'total': structure['subcommands'],
                    'tested': list(self.coverage_data['commands'].get(command, {}).get('subcommands_tested', [])),
                    'untested': [sub for sub in structure['subcommands'] 
                               if sub not in self.coverage_data['commands'].get(command, {}).get('subcommands_tested', [])]
                }
            }
            report['command_details'][command] = command_data
        
        # Generate recommendations
        recommendations = []
        
        if uncovered_commands > 0:
            recommendations.append(f"Add tests for {uncovered_commands} uncovered commands: {', '.join(self.coverage_data['uncovered_commands'])}")
        
        for command, structure in self.cli_structure.items():
            if command in self.coverage_data['commands']:
                untested_options = [opt for opt in structure['options'] 
                                  if opt not in self.coverage_data['commands'][command]['options_tested']]
                if untested_options:
                    recommendations.append(f"Add tests for untested options in '{command}': {', '.join(untested_options)}")
                
                untested_subcommands = [sub for sub in structure['subcommands'] 
                                      if sub not in self.coverage_data['commands'][command]['subcommands_tested']]
                if untested_subcommands:
                    recommendations.append(f"Add tests for untested subcommands in '{command}': {', '.join(untested_subcommands)}")
        
        report['recommendations'] = recommendations
        
        return report
    
    def analyze_all_tests(self, test_dir='tests'):
        """Analyze all test files in the test directory."""
        test_dir_path = Path(test_dir)
        
        if not test_dir_path.exists():
            print(f"Test directory {test_dir} not found")
            return
        
        test_files = list(test_dir_path.glob('test_*.py'))
        
        for test_file in test_files:
            print(f"Analyzing {test_file}")
            self.analyze_test_file(test_file)
            self.coverage_data['test_files'].append(str(test_file))
    
    def save_coverage_report(self, report, output_file='coverage_report.json'):
        """Save coverage report to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Coverage report saved to {output_file}")
    
    def print_coverage_summary(self, report):
        """Print a human-readable coverage summary."""
        summary = report['summary']
        
        print("\n" + "="*60)
        print("CLI TEST COVERAGE SUMMARY")
        print("="*60)
        print(f"Commands: {summary['covered_commands']}/{summary['total_commands']} ({summary['command_coverage_percentage']:.1f}%)")
        print(f"Options: {summary['covered_options']}/{summary['total_options']} ({summary['option_coverage_percentage']:.1f}%)")
        print(f"Subcommands: {summary['covered_subcommands']}/{summary['total_subcommands']} ({summary['subcommand_coverage_percentage']:.1f}%)")
        
        if summary['uncovered_commands'] > 0:
            print(f"\nUncovered Commands ({summary['uncovered_commands']}):")
            for cmd in report['uncovered_commands']:
                print(f"  - {cmd}")
        
        if report['recommendations']:
            print(f"\nRecommendations ({len(report['recommendations'])}):")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")


def main():
    """Main function to run coverage analysis."""
    analyzer = CLICoverageAnalyzer()
    
    # Analyze all test files
    analyzer.analyze_all_tests()
    
    # Generate report
    report = analyzer.generate_coverage_report()
    
    # Print summary
    analyzer.print_coverage_summary(report)
    
    # Save detailed report
    analyzer.save_coverage_report(report)
    
    return report['summary']['command_coverage_percentage'] >= 80  # Return success if coverage >= 80%


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 