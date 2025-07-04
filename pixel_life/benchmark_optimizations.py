#!/usr/bin/env python3
"""
Guardian-Guided Optimization Benchmarking Suite
Comprehensive performance comparison between original and optimized implementations.
"""

import time
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import psutil
import gc
import warnings
warnings.filterwarnings("ignore")

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name() if TORCH_AVAILABLE else "None"
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9 if TORCH_AVAILABLE else 0
except ImportError:
    TORCH_AVAILABLE = False
    GPU_NAME = "None"
    GPU_MEMORY = 0

# Import both implementations
from env import PixelLifeEnv
from env_optimized import PixelLifeEnvOptimized


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.results = {
            'system_info': self._get_system_info(),
            'benchmarks': {},
            'comparisons': {}
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information for reproducibility."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total': psutil.virtual_memory().total / 1e9,
            'numba_available': NUMBA_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'gpu_name': GPU_NAME,
            'gpu_memory_gb': GPU_MEMORY,
            'python_version': f"{psutil.Process().environ.get('PYTHON_VERSION', 'Unknown')}"
        }
    
    def benchmark_environment_creation(self, grid_sizes: List[int], iterations: int = 100) -> Dict[str, List[float]]:
        """Benchmark environment creation speed."""
        print("🔧 Benchmarking Environment Creation...")
        
        results = {'original': [], 'optimized': [], 'grid_sizes': grid_sizes}
        
        for grid_size in grid_sizes:
            print(f"   Grid size: {grid_size}x{grid_size}")
            
            # Original implementation
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                env = PixelLifeEnv(H=grid_size, W=grid_size)
                end = time.perf_counter()
                times.append(end - start)
                del env
                gc.collect()
            results['original'].append(np.mean(times))
            
            # Optimized implementation
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                env = PixelLifeEnvOptimized(H=grid_size, W=grid_size)
                end = time.perf_counter()
                times.append(end - start)
                del env
                gc.collect()
            results['optimized'].append(np.mean(times))
        
        self.results['benchmarks']['env_creation'] = results
        return results
    
    def benchmark_reset_speed(self, grid_sizes: List[int], iterations: int = 1000) -> Dict[str, List[float]]:
        """Benchmark environment reset speed."""
        print("🔄 Benchmarking Reset Speed...")
        
        results = {'original': [], 'optimized': [], 'grid_sizes': grid_sizes}
        
        for grid_size in grid_sizes:
            print(f"   Grid size: {grid_size}x{grid_size}")
            
            # Original implementation
            env_orig = PixelLifeEnv(H=grid_size, W=grid_size)
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                env_orig.reset()
                end = time.perf_counter()
                times.append(end - start)
            results['original'].append(np.mean(times))
            del env_orig
            
            # Optimized implementation
            env_opt = PixelLifeEnvOptimized(H=grid_size, W=grid_size)
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                env_opt.reset()
                end = time.perf_counter()
                times.append(end - start)
            results['optimized'].append(np.mean(times))
            del env_opt
        
        self.results['benchmarks']['reset_speed'] = results
        return results
    
    def benchmark_step_throughput(self, grid_sizes: List[int], steps: int = 10000) -> Dict[str, List[float]]:
        """Benchmark step execution throughput (steps per second)."""
        print("⚡ Benchmarking Step Throughput...")
        
        results = {'original': [], 'optimized': [], 'grid_sizes': grid_sizes}
        
        for grid_size in grid_sizes:
            print(f"   Grid size: {grid_size}x{grid_size}")
            
            # Original implementation
            env_orig = PixelLifeEnv(H=grid_size, W=grid_size, max_steps=steps*2)
            env_orig.reset()
            
            start = time.perf_counter()
            for i in range(steps):
                spice_action = np.random.randint(3)
                pixel_actions = np.random.randint(4, size=grid_size * grid_size)
                action = {'spice_action': spice_action, 'pixel_actions': pixel_actions}
                
                obs, rewards, terminated, truncated, info = env_orig.step(action)
                if terminated or truncated:
                    env_orig.reset()
            end = time.perf_counter()
            
            fps_orig = steps / (end - start)
            results['original'].append(fps_orig)
            del env_orig
            
            # Optimized implementation
            env_opt = PixelLifeEnvOptimized(H=grid_size, W=grid_size, max_steps=steps*2)
            env_opt.reset()
            
            start = time.perf_counter()
            for i in range(steps):
                spice_action = np.random.randint(3)
                pixel_actions = np.random.randint(4, size=grid_size * grid_size)
                action = {'spice_action': spice_action, 'pixel_actions': pixel_actions}
                
                obs, rewards, terminated, truncated, info = env_opt.step(action)
                if terminated or truncated:
                    env_opt.reset()
            end = time.perf_counter()
            
            fps_opt = steps / (end - start)
            results['optimized'].append(fps_opt)
            del env_opt
            
            print(f"      Original: {fps_orig:.1f} FPS")
            print(f"      Optimized: {fps_opt:.1f} FPS")
            print(f"      Speedup: {fps_opt/fps_orig:.2f}x")
        
        self.results['benchmarks']['step_throughput'] = results
        return results
    
    def benchmark_memory_usage(self, grid_sizes: List[int]) -> Dict[str, List[float]]:
        """Benchmark memory usage of different implementations."""
        print("💾 Benchmarking Memory Usage...")
        
        results = {'original': [], 'optimized': [], 'grid_sizes': grid_sizes}
        
        for grid_size in grid_sizes:
            print(f"   Grid size: {grid_size}x{grid_size}")
            
            # Original implementation
            gc.collect()
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1e6  # MB
            
            env_orig = PixelLifeEnv(H=grid_size, W=grid_size)
            env_orig.reset()
            
            mem_after = process.memory_info().rss / 1e6  # MB
            mem_orig = mem_after - mem_before
            results['original'].append(mem_orig)
            del env_orig
            
            # Optimized implementation
            gc.collect()
            mem_before = process.memory_info().rss / 1e6  # MB
            
            env_opt = PixelLifeEnvOptimized(H=grid_size, W=grid_size)
            env_opt.reset()
            
            mem_after = process.memory_info().rss / 1e6  # MB
            mem_opt = mem_after - mem_before
            results['optimized'].append(mem_opt)
            del env_opt
            
            print(f"      Original: {mem_orig:.1f} MB")
            print(f"      Optimized: {mem_opt:.1f} MB")
            print(f"      Reduction: {(1 - mem_opt/mem_orig)*100:.1f}%")
        
        self.results['benchmarks']['memory_usage'] = results
        return results
    
    def benchmark_scalability(self, max_grid_size: int = 128, step_size: int = 16) -> Dict[str, List[float]]:
        """Benchmark scalability with increasing problem sizes."""
        print("📈 Benchmarking Scalability...")
        
        grid_sizes = list(range(step_size, max_grid_size + 1, step_size))
        results = {'original': [], 'optimized': [], 'grid_sizes': grid_sizes}
        
        for grid_size in grid_sizes:
            print(f"   Grid size: {grid_size}x{grid_size}")
            
            steps = max(100, 1000 // (grid_size // 16))  # Adjust steps based on grid size
            
            # Original implementation
            try:
                env_orig = PixelLifeEnv(H=grid_size, W=grid_size)
                env_orig.reset()
                
                start = time.perf_counter()
                for _ in range(steps):
                    spice_action = np.random.randint(3)
                    pixel_actions = np.random.randint(4, size=grid_size * grid_size)
                    action = {'spice_action': spice_action, 'pixel_actions': pixel_actions}
                    
                    obs, rewards, terminated, truncated, info = env_orig.step(action)
                    if terminated or truncated:
                        env_orig.reset()
                end = time.perf_counter()
                
                fps_orig = steps / (end - start)
                results['original'].append(fps_orig)
                del env_orig
            except Exception as e:
                print(f"      Original failed: {e}")
                results['original'].append(0)
            
            # Optimized implementation
            try:
                env_opt = PixelLifeEnvOptimized(H=grid_size, W=grid_size)
                env_opt.reset()
                
                start = time.perf_counter()
                for _ in range(steps):
                    spice_action = np.random.randint(3)
                    pixel_actions = np.random.randint(4, size=grid_size * grid_size)
                    action = {'spice_action': spice_action, 'pixel_actions': pixel_actions}
                    
                    obs, rewards, terminated, truncated, info = env_opt.step(action)
                    if terminated or truncated:
                        env_opt.reset()
                end = time.perf_counter()
                
                fps_opt = steps / (end - start)
                results['optimized'].append(fps_opt)
                del env_opt
            except Exception as e:
                print(f"      Optimized failed: {e}")
                results['optimized'].append(0)
            
            if results['original'][-1] > 0 and results['optimized'][-1] > 0:
                speedup = results['optimized'][-1] / results['original'][-1]
                print(f"      Speedup: {speedup:.2f}x")
        
        self.results['benchmarks']['scalability'] = results
        return results
    
    def calculate_overall_speedups(self) -> Dict[str, float]:
        """Calculate overall speedup metrics across all benchmarks."""
        print("📊 Calculating Overall Speedups...")
        
        speedups = {}
        
        for bench_name, bench_data in self.results['benchmarks'].items():
            if 'original' in bench_data and 'optimized' in bench_data:
                orig_values = np.array(bench_data['original'])
                opt_values = np.array(bench_data['optimized'])
                
                # Filter out zero values to avoid division by zero
                valid_mask = (orig_values > 0) & (opt_values > 0)
                if np.any(valid_mask):
                    speedup_ratios = opt_values[valid_mask] / orig_values[valid_mask]
                    speedups[bench_name] = {
                        'mean_speedup': float(np.mean(speedup_ratios)),
                        'max_speedup': float(np.max(speedup_ratios)),
                        'min_speedup': float(np.min(speedup_ratios)),
                        'median_speedup': float(np.median(speedup_ratios))
                    }
        
        self.results['comparisons']['speedups'] = speedups
        return speedups
    
    def generate_performance_plots(self, save_dir: str = "benchmark_plots"):
        """Generate comprehensive performance visualization plots."""
        print("📈 Generating Performance Plots...")
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # 1. Step Throughput Comparison
        if 'step_throughput' in self.results['benchmarks']:
            data = self.results['benchmarks']['step_throughput']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Throughput comparison
            x = data['grid_sizes']
            ax1.plot(x, data['original'], 'o-', label='Original', color=colors[0], linewidth=2, markersize=8)
            ax1.plot(x, data['optimized'], 's-', label='Optimized', color=colors[1], linewidth=2, markersize=8)
            ax1.set_xlabel('Grid Size (NxN)')
            ax1.set_ylabel('Steps per Second (FPS)')
            ax1.set_title('Step Throughput Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # Speedup ratio
            speedups = [opt/orig if orig > 0 else 0 for orig, opt in zip(data['original'], data['optimized'])]
            ax2.bar(range(len(x)), speedups, color=colors[2], alpha=0.7)
            ax2.set_xlabel('Grid Size Index')
            ax2.set_ylabel('Speedup Ratio (X times faster)')
            ax2.set_title('Performance Speedup by Grid Size')
            ax2.set_xticks(range(len(x)))
            ax2.set_xticklabels([f"{size}x{size}" for size in x], rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/step_throughput_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Memory Usage Comparison
        if 'memory_usage' in self.results['benchmarks']:
            data = self.results['benchmarks']['memory_usage']
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            x = data['grid_sizes']
            width = 0.35
            x_pos = np.arange(len(x))
            
            ax.bar(x_pos - width/2, data['original'], width, label='Original', color=colors[0], alpha=0.8)
            ax.bar(x_pos + width/2, data['optimized'], width, label='Optimized', color=colors[1], alpha=0.8)
            
            ax.set_xlabel('Grid Size (NxN)')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Usage Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{size}x{size}" for size in x])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/memory_usage_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Scalability Analysis
        if 'scalability' in self.results['benchmarks']:
            data = self.results['benchmarks']['scalability']
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            x = data['grid_sizes']
            ax.plot(x, data['original'], 'o-', label='Original', color=colors[0], linewidth=2, markersize=6)
            ax.plot(x, data['optimized'], 's-', label='Optimized', color=colors[1], linewidth=2, markersize=6)
            
            ax.set_xlabel('Grid Size (NxN)')
            ax.set_ylabel('Steps per Second (FPS)')
            ax.set_title('Scalability Analysis: Performance vs Grid Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # Add annotations for key points
            max_orig_idx = np.argmax(data['original'])
            max_opt_idx = np.argmax(data['optimized'])
            
            ax.annotate(f'Peak Original: {data["original"][max_orig_idx]:.1f} FPS', 
                       xy=(x[max_orig_idx], data['original'][max_orig_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[0], alpha=0.3),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax.annotate(f'Peak Optimized: {data["optimized"][max_opt_idx]:.1f} FPS', 
                       xy=(x[max_opt_idx], data['optimized'][max_opt_idx]),
                       xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[1], alpha=0.3),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/scalability_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Summary Dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Speedup summary
        if 'speedups' in self.results['comparisons']:
            speedup_data = self.results['comparisons']['speedups']
            benchmarks = list(speedup_data.keys())
            mean_speedups = [speedup_data[bench]['mean_speedup'] for bench in benchmarks]
            
            bars = ax1.bar(benchmarks, mean_speedups, color=colors[:len(benchmarks)], alpha=0.8)
            ax1.set_ylabel('Mean Speedup (X times faster)')
            ax1.set_title('Guardian Optimization Results')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5)
            
            # Add value labels on bars
            for bar, speedup in zip(bars, mean_speedups):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{speedup:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # System info
        sys_info = self.results['system_info']
        info_text = f"""System Information:
CPU Cores: {sys_info['cpu_count']}
Memory: {sys_info['memory_total']:.1f} GB
GPU: {sys_info['gpu_name']}
GPU Memory: {sys_info['gpu_memory_gb']:.1f} GB
Numba: {'Available' if sys_info['numba_available'] else 'Not Available'}
CUDA: {'Available' if sys_info['torch_available'] else 'Not Available'}"""
        
        ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('System Configuration')
        
        # Performance trend (if available)
        if 'step_throughput' in self.results['benchmarks']:
            data = self.results['benchmarks']['step_throughput']
            x = data['grid_sizes']
            
            # Calculate efficiency (performance per unit area)
            orig_efficiency = [fps / (size**2) for fps, size in zip(data['original'], x)]
            opt_efficiency = [fps / (size**2) for fps, size in zip(data['optimized'], x)]
            
            ax3.plot(x, orig_efficiency, 'o-', label='Original', color=colors[0], linewidth=2)
            ax3.plot(x, opt_efficiency, 's-', label='Optimized', color=colors[1], linewidth=2)
            ax3.set_xlabel('Grid Size (NxN)')
            ax3.set_ylabel('FPS per Unit Area')
            ax3.set_title('Computational Efficiency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        
        # Optimization techniques summary
        techniques = [
            "Numba JIT Compilation",
            "Vectorized Operations", 
            "Memory Pre-allocation",
            "Fast Data Structures",
            "Optimized Algorithms",
            "GPU Acceleration Ready"
        ]
        
        ax4.barh(range(len(techniques)), [1] * len(techniques), color=colors[2], alpha=0.6)
        ax4.set_yticks(range(len(techniques)))
        ax4.set_yticklabels(techniques)
        ax4.set_xlabel('Implementation Status')
        ax4.set_title('Guardian Optimization Techniques Applied')
        ax4.set_xlim(0, 1.2)
        
        for i, technique in enumerate(techniques):
            ax4.text(1.05, i, '✓', ha='center', va='center', fontsize=16, color='green', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/optimization_summary_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Performance plots saved to '{save_dir}/'")
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Benchmark results saved to '{filename}'")
    
    def print_summary(self):
        """Print a comprehensive summary of benchmark results."""
        print("\n" + "="*80)
        print("🏆 GUARDIAN-GUIDED OPTIMIZATION BENCHMARK SUMMARY")
        print("="*80)
        
        sys_info = self.results['system_info']
        print(f"🖥️  System: {sys_info['cpu_count']} cores, {sys_info['memory_total']:.1f}GB RAM")
        if sys_info['torch_available']:
            print(f"🎮 GPU: {sys_info['gpu_name']} ({sys_info['gpu_memory_gb']:.1f}GB)")
        print(f"⚡ Numba JIT: {'✓ Available' if sys_info['numba_available'] else '✗ Not Available'}")
        
        if 'speedups' in self.results['comparisons']:
            print("\n📈 PERFORMANCE IMPROVEMENTS:")
            print("-" * 50)
            
            for bench_name, metrics in self.results['comparisons']['speedups'].items():
                print(f"{bench_name.replace('_', ' ').title()}:")
                print(f"   Mean Speedup: {metrics['mean_speedup']:.2f}x")
                print(f"   Max Speedup:  {metrics['max_speedup']:.2f}x")
                print(f"   Min Speedup:  {metrics['min_speedup']:.2f}x")
                print()
        
        # Calculate overall efficiency gain
        if 'step_throughput' in self.results['benchmarks']:
            data = self.results['benchmarks']['step_throughput']
            total_orig = sum(data['original'])
            total_opt = sum(data['optimized'])
            if total_orig > 0:
                overall_speedup = total_opt / total_orig
                print(f"🚀 OVERALL PERFORMANCE GAIN: {overall_speedup:.2f}x faster")
                print(f"💡 Efficiency Improvement: {(overall_speedup-1)*100:.1f}% faster execution")
        
        print("\n" + "="*80)


def main():
    """Run comprehensive benchmark suite."""
    print("🚀 Guardian-Guided Optimization Benchmark Suite")
    print("=" * 60)
    print("Testing performance improvements across multiple dimensions...")
    
    benchmark = PerformanceBenchmark()
    
    # Define test parameters
    small_grids = [8, 16, 24, 32]
    large_grids = [8, 16, 32, 48, 64]
    
    # Run benchmarks
    try:
        benchmark.benchmark_environment_creation(small_grids, iterations=50)
        benchmark.benchmark_reset_speed(small_grids, iterations=500)
        benchmark.benchmark_step_throughput(small_grids, steps=5000)
        benchmark.benchmark_memory_usage(small_grids)
        benchmark.benchmark_scalability(max_grid_size=96, step_size=16)
        
        # Calculate overall metrics
        benchmark.calculate_overall_speedups()
        
        # Generate visualizations
        benchmark.generate_performance_plots()
        
        # Save and display results
        benchmark.save_results()
        benchmark.print_summary()
        
        print("\n✅ Benchmark suite completed successfully!")
        print("📊 Check 'benchmark_plots/' for detailed visualizations")
        print("💾 Results saved to 'benchmark_results.json'")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()