# Enhanced Pixel Life CLI - Complete Feature Guide

This document describes all the new CLI features that have been added to make Pixel Life a robust final product.

## 🎯 Overview

The enhanced Pixel Life CLI now includes comprehensive tools for:
- **Model Management**: Versioning, tagging, and lifecycle management
- **Experiment Tracking**: Hyperparameter tracking and result comparison
- **System Monitoring**: Real-time performance monitoring and health checks
- **Advanced Visualization**: Model and experiment comparison tools
- **Workflow Automation**: Batch processing and automated pipelines
- **Debugging & Profiling**: Performance analysis and debugging tools
- **Interactive Help**: Tutorials and guided learning

## 🚀 New CLI Commands

### 1. Model Management (`models`)

Manage AI models with versioning, metadata, and lifecycle management.

```bash
# List all registered models
pixel_life models list

# List models with specific tags
pixel_life models list --tags "production,latest"

# Register a new model
pixel_life models register \
  --model-path ./logs/my_model.zip \
  --name "production-model-v1" \
  --description "Production model for pixel life" \
  --author "alice@company.com" \
  --tags "production,stable"

# Get detailed information about a model
pixel_life models info <model_id>

# Delete a model
pixel_life models delete <model_id>
```

**Features:**
- Automatic model ID generation and checksum calculation
- Metadata tracking (author, version, description, tags)
- Performance metrics storage
- File size and dependency tracking
- Model lifecycle management

### 2. Experiment Management (`experiments`)

Track experiments, hyperparameters, and results for reproducible research.

```bash
# List all experiments
pixel_life experiments list

# List experiments by status
pixel_life experiments list --status completed

# List experiments by tags
pixel_life experiments list --tags "hyperparameter-search"

# Create a new experiment
pixel_life experiments create \
  --name "learning-rate-study" \
  --description "Testing different learning rates" \
  --hyperparams learning_rate=0.001 batch_size=64 n_steps=2048 \
  --tags "hyperparameter-search,production"

# Create experiment with parent (for A/B testing)
pixel_life experiments create \
  --name "lr-study-variant" \
  --parent abc12345 \
  --hyperparams learning_rate=0.0005

# Get experiment details
pixel_life experiments info <experiment_id>
```

**Features:**
- Hyperparameter tracking and versioning
- Experiment lineage (parent-child relationships)
- Status tracking (running, completed, failed)
- Result storage and comparison
- Tag-based organization

### 3. System Monitoring (`monitor`)

Real-time system monitoring and health checks.

```bash
# Show current system status
pixel_life monitor status

# Start real-time monitoring
pixel_life monitor start --interval 2

# Monitor during training (in background)
pixel_life monitor start --interval 5 &
pixel_life train --timesteps 100000
```

**Features:**
- CPU, memory, and disk usage monitoring
- GPU usage tracking (when available)
- Process count monitoring
- Real-time metrics collection
- System information display

### 4. Advanced Visualization (`visualize`)

Compare models and experiments with interactive visualizations.

```bash
# Compare models and experiments
pixel_life visualize compare --show

# Save comparison plot
pixel_life visualize compare --save comparison.png

# Show and save
pixel_life visualize compare --show --save results.png
```

**Features:**
- Automatic data loading from models and experiments
- Multi-metric comparison plots
- Export to various formats (PNG, PDF, etc.)
- Interactive matplotlib plots

### 5. Workflow Automation (`workflow`)

Run predefined workflows for common tasks.

```bash
# Train multiple models with different configurations
pixel_life workflow batch --workflow train-multiple

# Evaluate all registered models
pixel_life workflow batch --workflow evaluate-all
```

**Features:**
- Automated batch training with different hyperparameters
- Model evaluation pipeline
- Experiment creation and tracking
- Performance comparison

### 6. Debugging & Profiling (`debug`)

Performance profiling and debugging tools.

```bash
# Profile basic environment performance
pixel_life debug basic --size 30 --steps 1000

# Profile training performance
pixel_life debug training --size 30 --steps 10000

# Save profile data for analysis
pixel_life debug basic --size 30 --steps 1000 --save profile.stats
```

**Features:**
- CPU profiling with cProfile
- Memory usage analysis
- Performance bottleneck identification
- Profile data export for external analysis

### 7. Interactive Help (`help`)

Guided tutorials and interactive learning.

```bash
# Show basic tutorial
pixel_life help basic

# Show training tutorial
pixel_life help training

# Show experiments tutorial
pixel_life help experiments

# Show monitoring tutorial
pixel_life help monitoring

# Interactive tutorial (when implemented)
pixel_life help basic --interactive
```

**Features:**
- Step-by-step tutorials
- Command examples
- Best practices
- Interactive guided execution (planned)

## 📊 Data Management

### Model Storage Structure
```
models/
├── metadata.json          # Model registry
├── abc12345_model-v1.zip  # Model files
├── def67890_model-v2.zip
└── ...
```

### Experiment Storage Structure
```
experiments/
├── experiments.json       # Experiment registry
└── ...
```

### Log Storage Structure
```
logs/
├── log_metadata.db        # SQLite database
├── structured/            # JSONL log files
├── models/                # Model performance logs
├── performance/           # Performance metrics
└── archive/               # Archived old logs
```

## 🔧 Configuration

### Environment Variables
```bash
# Set default directories
export PIXEL_LIFE_MODELS_DIR="./models"
export PIXEL_LIFE_EXPERIMENTS_DIR="./experiments"
export PIXEL_LIFE_LOGS_DIR="./logs"

# Enable debug mode
export PIXEL_LIFE_DEBUG=1
```

### Configuration Files
```bash
# Generate default config
pixel_life config --generate

# Load specific config
pixel_life config --load my_config.json
```

## 🚀 Advanced Usage Examples

### 1. Complete ML Pipeline
```bash
# Create experiment
exp_id=$(pixel_life experiments create \
  --name "production-training" \
  --hyperparams learning_rate=0.001 batch_size=64)

# Start monitoring
pixel_life monitor start --interval 5 &

# Train model
pixel_life train --timesteps 1000000

# Register model
model_id=$(pixel_life models register \
  --model-path ./logs/final_model.zip \
  --name "production-v1" \
  --tags "production,latest")

# Update experiment with results
pixel_life experiments update $exp_id \
  --status completed \
  --results model_id=$model_id final_reward=95.5

# Compare with previous models
pixel_life visualize compare --save results.png
```

### 2. Hyperparameter Search
```bash
# Create multiple experiments
for lr in 0.001 0.0005 0.0001; do
  pixel_life experiments create \
    --name "lr-$lr" \
    --hyperparams learning_rate=$lr
done

# Run batch training
pixel_life workflow batch --workflow train-multiple

# Compare results
pixel_life visualize compare --show
```

### 3. Model Evaluation Pipeline
```bash
# Evaluate all models
pixel_life workflow batch --workflow evaluate-all

# Find best model
pixel_life models list --tags "evaluated" | grep "best"

# Deploy best model
pixel_life models register \
  --model-path ./models/best_model.zip \
  --name "production-deployed" \
  --tags "production,deployed"
```

## 🔍 Monitoring & Debugging

### System Health Check
```bash
# Quick health check
pixel_life monitor status

# Monitor during training
pixel_life monitor start &
pixel_life train --timesteps 100000
kill %1  # Stop monitoring
```

### Performance Profiling
```bash
# Profile environment performance
pixel_life debug basic --size 50 --steps 1000 --save env_profile.stats

# Profile training performance
pixel_life debug training --size 30 --steps 10000 --save train_profile.stats

# Analyze with external tools
python -c "import pstats; pstats.Stats('env_profile.stats').print_stats(10)"
```

## 📈 Logging & Analytics

### Log Management
```bash
# View log overview
pixel_life logs overview

# Search logs
pixel_life logs search "training" --level INFO

# View performance history
pixel_life logs performance --limit 20

# View recent errors
pixel_life logs errors --limit 10

# Clean up old logs
pixel_life logs cleanup --days 30 --execute
```

### Performance Tracking
```bash
# View model performance
pixel_life logs performance

# Compare model performance
pixel_life visualize compare --show

# Export performance data
pixel_life logs performance --export performance.csv
```

## 🛠️ Troubleshooting

### Common Issues

1. **Model not found**
   ```bash
   # Check if model exists
   pixel_life models list
   
   # Re-register if needed
   pixel_life models register --model-path ./path/to/model.zip --name "my-model"
   ```

2. **Experiment creation fails**
   ```bash
   # Check experiment directory permissions
   ls -la ./experiments/
   
   # Create directory if missing
   mkdir -p ./experiments
   ```

3. **Monitoring not working**
   ```bash
   # Check system requirements
   pixel_life monitor status
   
   # Install missing dependencies
   pip install psutil pynvml
   ```

### Debug Mode
```bash
# Enable debug mode
export PIXEL_LIFE_DEBUG=1

# Run with verbose output
pixel_life models list --verbose
```

## 🔮 Future Enhancements

### Planned Features
- **Collaboration Tools**: Team management and sharing
- **Cloud Integration**: AWS, GCP, Azure support
- **Advanced Analytics**: Statistical analysis and reporting
- **API Integration**: REST API for programmatic access
- **Web Dashboard**: Browser-based interface
- **Scheduled Training**: Cron-like job scheduling
- **Model Serving**: Real-time model inference
- **A/B Testing**: Automated experiment comparison

### Contributing
To add new CLI features:
1. Add command functions in `pixel_life.py`
2. Add argument parsers in the `main()` function
3. Update this documentation
4. Add tests in `test_cli.py`
5. Update requirements if needed

## 📚 Additional Resources

- [Original README](README.md) - Basic usage and installation
- [Apple Silicon Guide](APPLE_SILICON_README.md) - Performance optimization
- [Logging Guide](LOGGING_README.md) - Advanced logging features
- [Examples](examples.md) - Code examples and use cases

---

**Pixel Life Enhanced CLI** - A comprehensive tool for AI research and development in artificial life environments. 