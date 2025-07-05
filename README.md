# Pixel Life: A 2D Artificial Life Environment

Pixel Life is a competitive multi-agent Gym environment where a main agent controls pixel organisms that can split, consume, combine, and forfeit, while an adversarial "spice" agent tries to make survival difficult by expanding the universe and tweaking game parameters.

## Features

- **Dual-agent competition**: Main agent controls organisms, spice agent controls environment
- **Dynamic universe**: Grid can expand during gameplay
- **Emergent behaviors**: Organisms can split, consume dead cells or smaller organisms, combine with others, or sacrifice themselves
- **Adaptive difficulty**: Spice agent learns to challenge the main agent by tweaking game rules
- **Reinforcement learning ready**: Compatible with Stable Baselines3 PPO agents
- **Unified CLI**: Single `pixel_life` command with multiple modes and options

## Quick Installation

```bash
# Clone or download the pixel_life directory
cd pixel_life

# Run the installation script
./install.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Make the `pixel_life` command available system-wide

## Project Structure

```
pixel_life/
├── pixel_life.py         # Unified command-line interface
├── env.py                # Main environment implementation
├── train.py              # Training script for PPO agents
├── per_pixel_ai.py       # Per-pixel AI system
├── continual_learning.py # Continual learning system
├── basic_renderer.py     # Core rendering functionality
├── enhanced_renderer.py  # Enhanced Pygame renderer
├── setup.py              # Package installation script
├── install.sh            # Quick installation script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── tests/               # Test files
└── logs/                # Training logs and model checkpoints
```

## Usage

### Unified Command Line Interface

The `pixel_life` command provides access to all functionality:

```bash
# Show all available commands
pixel_life --help

# Basic environment demonstration
pixel_life basic --render

# AI agent demonstration
pixel_life ai --render

# Per-pixel AI system
pixel_life per-pixel --train --render

# Continual learning system
pixel_life continual --train --render

# Pygame visualization
pixel_life pygame

# Enhanced Pygame visualization with zoom
pixel_life enhanced --size 100

# Apple Silicon accelerated demos (macOS only)
pixel_life accelerated --size 50 --steps 200
pixel_life basic --size 50 --steps 200 --accelerated
pixel_life ai --size 50 --steps 200 --accelerated

# Full training session
pixel_life train --timesteps 100000

# Model evaluation
pixel_life evaluate --model-path ./logs/training_run_20240101_120000/main_agent/final_model

# System information
pixel_life info

# Performance benchmark
pixel_life benchmark --steps 10000

# Configuration management
pixel_life config --generate
```

### Available Modes

- **`basic`**: Simple environment demonstration with random actions
- **`ai`**: AI agent demonstration with quick training
- **`per-pixel`**: Per-pixel AI system where each pixel has its own agent
- **`continual`**: Continual learning system that adapts over time
- **`pygame`**: Pygame-based visualization
- **`enhanced`**: Enhanced Pygame visualization with zoom and resizable window
- **`accelerated`**: Apple Silicon accelerated demo (macOS only)
- **`benchmark-demo`**: Performance benchmark demo (macOS only)
- **`train`**: Full training session with customizable parameters
- **`evaluate`**: Evaluate trained models
- **`info`**: Display system information and available models
- **`benchmark`**: Run performance benchmark
- **`config`**: Generate or load configuration files

### Command Line Options

Each mode supports various options. For example:

```bash
# Run with custom parameters
pixel_life basic --size 50 --steps 500 --render

# Train with specific settings
pixel_life train --size 40 --timesteps 500000 --n-envs 8 --device cuda

# Run per-pixel AI with training
pixel_life per-pixel --train --generations 10 --steps 300 --render

# Run enhanced visualization
pixel_life enhanced --size 200 --initial-zoom 0.005

# Run with Apple Silicon acceleration (macOS only)
pixel_life basic --size 50 --steps 200 --accelerated
pixel_life train --timesteps 100000 --device mps --accelerated
```

Run `pixel_life <mode> --help` for mode-specific options.

## Apple Silicon Acceleration (macOS)

For Apple Silicon Macs (M1, M2, M3, M4), Pixel Life includes optimized acceleration features:

### Installation

```bash
# Quick installation for Apple Silicon
./install_apple_silicon.sh

# Activate environment
source activate_pixel_life.sh
```

### Usage

```bash
# Run with acceleration
pixel_life basic --size 50 --steps 200 --accelerated
pixel_life ai --size 50 --steps 200 --accelerated
pixel_life train --timesteps 100000 --device mps --accelerated

# Performance benchmarks
pixel_life benchmark-demo --size 50
```

### Performance Improvements

- **2-5x faster environment simulation** using Numba JIT compilation
- **3-10x faster neural network training** using PyTorch MPS
- **2-4x faster rendering** using Metal-optimized graphics

See [APPLE_SILICON_README.md](APPLE_SILICON_README.md) for detailed documentation.

## Dependencies

- Python 3.8+
- gymnasium>=1.0.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- stable-baselines3>=2.0.0
- torch>=2.0.0 (with MPS support for Apple Silicon)
- numba>=0.56.0
- tensorboard>=2.10.0
- psutil>=5.9.0
- pygame>=2.1.0
- coremltools>=6.0.0 (Apple Silicon acceleration)

## Programmatic Usage

You can also use the environment programmatically:

```python
from env import PixelLifeEnv

# Create environment
env = PixelLifeEnv(H=30, W=30)

# Reset
obs = env.reset()
obs_main, obs_spice = obs

# Run a few steps with random actions
for _ in range(100):
    # Random spice action
    spice_action = env.spice_action_space.sample()
    
    # Random pixel actions
    pixel_actions = {}
    for coord in env.live_pixels:
        action_type = 1  # Split
        direction = 0    # Up
        pixel_actions[coord] = (action_type, direction)
    
    # Step
    obs, rewards, done, truncated, info = env.step(spice_action, pixel_actions)
    
    # Render (optional)
    env.render()
    
    if done:
        break
```

## Training Agents

### Quick Training

```bash
# Basic training with default parameters
pixel_life train --timesteps 1000000

# Custom training parameters
pixel_life train --size 40 --timesteps 500000 --n-envs 8 --device cuda
```

### Monitor Training

Training logs are saved to `./logs/training_run_TIMESTAMP/`. You can monitor training with TensorBoard:

```bash
tensorboard --logdir ./logs
```

## Examples

### Basic Demo
```bash
pixel_life basic --size 30 --steps 200 --render
```

### AI Training and Demo
```bash
pixel_life train --timesteps 100000 --size 30
pixel_life evaluate --model-path ./logs/training_run_*/main_agent/final_model --render
```

### Per-Pixel AI System
```bash
pixel_life per-pixel --train --generations 5 --steps 1000 --render
```

### Enhanced Visualization
```bash
pixel_life enhanced --size 150 --initial-zoom 0.01
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you've run `./install.sh` or activated the virtual environment
2. **CUDA errors**: Use `--device cpu` for CPU-only training
3. **Memory issues**: Reduce `--size` or `--n-envs` parameters
4. **Pygame not found**: Install with `pip install pygame`

### Performance Tips

- Use smaller environment sizes for faster training
- Reduce the number of parallel environments if memory is limited
- Use CPU training for simpler experiments
- Enable rendering only when needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.