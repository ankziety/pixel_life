"""
System Monitor for Pixel Life CLI
Handles system resource monitoring and metrics collection.
"""

import psutil
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemMetrics:
    """System metrics structure."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    gpu_usage_percent: float
    active_processes: int
    timestamp: float


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history = []
        self.max_history = 1000
    
    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring."""
        self.monitoring = True
        self.interval = interval
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Count active processes (simplified)
            active_processes = len([p for p in psutil.process_iter(['pid', 'status']) 
                                  if p.info['status'] == 'running'])
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                gpu_usage_percent=0.0,  # Placeholder for GPU monitoring
                active_processes=active_processes,
                timestamp=time.time()
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            # Return default metrics if monitoring fails
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                gpu_usage_percent=0.0,
                active_processes=0,
                timestamp=time.time()
            )
    
    def get_metrics_history(self, duration_seconds: Optional[float] = None) -> list[SystemMetrics]:
        """Get metrics history, optionally filtered by duration."""
        if duration_seconds is None:
            return self.metrics_history.copy()
        
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_average_metrics(self, duration_seconds: Optional[float] = None) -> SystemMetrics:
        """Get average metrics over a time period."""
        metrics = self.get_metrics_history(duration_seconds)
        
        if not metrics:
            return SystemMetrics(0.0, 0.0, 0.0, 0.0, 0, time.time())
        
        avg_cpu = sum(m.cpu_percent for m in metrics) / len(metrics)
        avg_memory = sum(m.memory_percent for m in metrics) / len(metrics)
        avg_disk = sum(m.disk_usage_percent for m in metrics) / len(metrics)
        avg_processes = sum(m.active_processes for m in metrics) / len(metrics)
        
        return SystemMetrics(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            disk_usage_percent=avg_disk,
            gpu_usage_percent=0.0,
            active_processes=int(avg_processes),
            timestamp=time.time()
        )
    
    def get_peak_metrics(self, duration_seconds: Optional[float] = None) -> SystemMetrics:
        """Get peak metrics over a time period."""
        metrics = self.get_metrics_history(duration_seconds)
        
        if not metrics:
            return SystemMetrics(0.0, 0.0, 0.0, 0.0, 0, time.time())
        
        peak_cpu = max(m.cpu_percent for m in metrics)
        peak_memory = max(m.memory_percent for m in metrics)
        peak_disk = max(m.disk_usage_percent for m in metrics)
        peak_processes = max(m.active_processes for m in metrics)
        
        return SystemMetrics(
            cpu_percent=peak_cpu,
            memory_percent=peak_memory,
            disk_usage_percent=peak_disk,
            gpu_usage_percent=0.0,
            active_processes=peak_processes,
            timestamp=time.time()
        ) 