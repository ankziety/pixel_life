# Pixel Life Logging System

A comprehensive logging management system that addresses the previous logging issues and provides powerful tools for monitoring, analyzing, and managing Pixel Life training sessions.

## 🎯 Problems Solved

The new logging system addresses all the previous issues:

- ✅ **Too many files**: Centralized log storage with organized structure
- ✅ **No tracking**: Comprehensive metadata tracking and statistics
- ✅ **No best model tracking**: Automatic performance tracking and best model identification
- ✅ **No performance analysis**: Built-in performance metrics and analysis tools
- ✅ **No CLI tools**: Human-readable CLI interface for log management

## 🏗️ Architecture

### Core Components

1. **LogManager** (`log_manager.py`): Central logging system with SQLite database
2. **PixelLogManager** (`pixel_logs.py`): CLI-friendly log analysis and management
3. **Integration**: Seamless integration with existing training scripts

### Directory Structure

```
logs/
├── log_metadata.db          # SQLite database with structured logs
├── structured/              # JSONL log files by date
├── models/                  # Model performance logs
├── performance/             # Performance metrics
└── archive/                 # Archived old logs
```

## 🚀 Quick Start

### 1. Basic Usage

The logging system is automatically integrated into training sessions:

```bash
# Run training with automatic logging
pixel_life train --timesteps 100000

# View log overview
python pixel_logs.py overview

# Search for specific logs
python pixel_logs.py search "training"
```

### 2. CLI Commands

```bash
# Show comprehensive log overview
python pixel_logs.py overview

# Search logs with filters
python pixel_logs.py search "error" --level ERROR --limit 10

# View model performance history
python pixel_logs.py performance --limit 20

# Show recent errors and warnings
python pixel_logs.py errors --limit 15

# Clean up old logs (dry run)
python pixel_logs.py cleanup --days 30

# Actually clean up old logs
python pixel_logs.py cleanup --days 30 --execute
```

## 📊 Features

### 1. Structured Logging

All logs are stored in structured JSON format with metadata:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Training started",
  "module": "train",
  "function": "train_pixel_life",
  "request_id": "abc123",
  "duration_ms": 1500.5,
  "metadata": {
    "timesteps": 100000,
    "device": "cpu"
  }
}
```

### 2. Performance Tracking

Automatic tracking of model performance metrics:

- Episode rewards
- Survival rates
- Training steps
- Model sizes
- Training time
- Hyperparameters
- Device information

### 3. Best Model Identification

The system automatically identifies and tracks the best performing model:

```bash
python pixel_logs.py performance
```

Output:
```
📈 Model Performance History
====================================================================================================
Gen  Reward   Survival  Steps     Size(MB)  Time(s)  Device  Date
----------------------------------------------------------------------------------------------------
5    125.50   0.95      50000     12.3      180.5    cpu     2024-01-15
3    98.20    0.88      30000     11.8      120.2    cpu     2024-01-14
1    75.30    0.82      10000     10.5      60.1     cpu     2024-01-13

🏆 Best Model: Generation 5 (Reward: 125.50)
```

### 4. Log Statistics

Comprehensive statistics about your logs:

```bash
python pixel_logs.py overview
```

Output:
```
📊 Pixel Life Log Overview
============================================================
📁 Total Files: 1,247
💾 Total Size: 156.78 MB

📋 File Types:
  .jsonl: 45 files
  .zip: 23 files
  .json: 156 files

🕒 Recent Activity (Last 24h):
  structured/2024-01-15.jsonl (2.3 MB)
  performance/performance_20240115_103000.json (0.1 MB)

🏆 Top Performing Models:
  1. Gen 5: Reward 125.50, Survival 0.95
  2. Gen 3: Reward 98.20, Survival 0.88
  3. Gen 1: Reward 75.30, Survival 0.82

❌ Recent Errors:
  2024-01-15T09:15:00: CUDA out of memory
  2024-01-15T08:30:00: Environment reset failed
```

### 5. Advanced Search

Powerful search capabilities with filters:

```bash
# Search for training-related logs
python pixel_logs.py search "training" --level INFO

# Search for errors in the last 24 hours
python pixel_logs.py search "error" --level ERROR --start-date 2024-01-15

# Search for specific module
python pixel_logs.py search "env" --limit 50
```

### 6. Automatic Cleanup

Intelligent log cleanup to manage disk space:

```bash
# See what would be cleaned up (dry run)
python pixel_logs.py cleanup --days 30

# Actually clean up old files
python pixel_logs.py cleanup --days 30 --execute
```

## 🔧 Integration

### Training Integration

The logging system is automatically integrated into training sessions:

```python
# In your training script
from log_manager import LogManager

log_manager = LogManager('./logs')
request_id = log_manager.start_request()

# Log training events
log_manager.log("INFO", "Training started", module="train")

# Log model performance
from log_manager import ModelPerformance
performance = ModelPerformance(
    model_path="./models/model.zip",
    timestamp=datetime.now().isoformat(),
    generation=1,
    episode_reward=85.5,
    survival_rate=0.92,
    training_steps=100000,
    model_size_mb=15.2,
    training_time_seconds=120.5,
    environment_size=30,
    hyperparameters={'learning_rate': 3e-4},
    device='cpu',
    framework='stable-baselines3'
)
log_manager.log_model_performance(performance)
```

### Custom Logging

You can add custom logging to any part of your code:

```python
from log_manager import LogManager

log_manager = LogManager('./logs')
request_id = log_manager.start_request()

# Log with metadata
log_manager.log("INFO", "Custom event", 
                module="my_module", 
                function="my_function",
                duration_ms=1500.5,
                metadata={'custom_field': 'value'})
```

## 🧪 Testing

Test the logging system:

```bash
# Run comprehensive tests
python test_logging.py

# Test specific components
python -c "
from log_manager import LogManager
lm = LogManager('./test_logs')
lm.log('INFO', 'Test message')
print('✅ Logging test passed')
"
```

## 📈 Performance Benefits

### Before (Old System)
- ❌ Scattered log files across multiple directories
- ❌ No way to track log file counts or sizes
- ❌ No identification of best performing models
- ❌ No performance analysis tools
- ❌ No CLI interface for log management
- ❌ Manual cleanup required

### After (New System)
- ✅ Centralized log storage with organized structure
- ✅ Automatic tracking of file counts, sizes, and types
- ✅ Automatic best model identification and tracking
- ✅ Built-in performance analysis and visualization
- ✅ Human-readable CLI interface with color coding
- ✅ Automatic log cleanup and archiving
- ✅ Structured logging with metadata
- ✅ SQLite database for fast queries
- ✅ Request tracking for debugging

## 🔍 Troubleshooting

### Common Issues

1. **Database not found**: Run training first to create the database
2. **Import errors**: Ensure `log_manager.py` and `pixel_logs.py` are in your project
3. **Permission errors**: Check write permissions for the logs directory

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 API Reference

### LogManager Class

```python
class LogManager:
    def __init__(self, base_dir: str = "./logs")
    def start_request(self, request_id: str = None) -> str
    def log(self, level: str, message: str, **kwargs)
    def log_model_performance(self, performance: ModelPerformance)
    def get_best_model(self) -> Optional[ModelPerformance]
    def get_log_stats(self) -> LogStats
    def cleanup_old_logs(self, days_to_keep: int = 30) -> int
    def search_logs(self, query: str, **kwargs) -> List[LogEntry]
```

### PixelLogManager Class

```python
class PixelLogManager:
    def __init__(self, logs_dir: str = "./logs")
    def show_overview(self)
    def search_logs(self, query: str, level: str = None, limit: int = 20)
    def show_performance(self, limit: int = 10)
    def show_errors(self, limit: int = 20)
    def cleanup_logs(self, days: int = 30, dry_run: bool = True)
```

## 🎉 Conclusion

The new logging system provides a comprehensive solution for managing Pixel Life training logs. It addresses all the previous issues while adding powerful new features for monitoring, analysis, and management.

Key benefits:
- **Centralized management**: All logs in one organized location
- **Performance tracking**: Automatic best model identification
- **CLI tools**: Human-readable interface for log management
- **Structured data**: JSON-based logs with rich metadata
- **Automatic cleanup**: Intelligent log rotation and archiving
- **Fast queries**: SQLite database for efficient searching

Start using the new logging system today to gain better insights into your training sessions! 