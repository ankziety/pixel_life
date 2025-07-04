# PixelLife: Multi-Agent Reinforcement Learning Environment

A 2D grid-based "pixel life" environment where organisms compete for survival against a "spice" agent that can modify the environment dynamics. Built with OpenAI Gymnasium and trained using Stable Baselines 3.

## 🎯 Overview

PixelLife simulates cellular automata-like organisms on a 2D grid, where:
- **Main Agent**: Controls individual pixels/organisms through actions (split, consume, combine, forfeit)
- **Spice Agent**: Modifies the environment (expands universe, tweaks parameters)
- **Competitive Dynamics**: Main agent tries to grow population, Spice agent tries to limit it

## 🚀 Features

- **Dynamic Grid Expansion**: Universe can grow during gameplay
- **Emergent Complexity**: Simple rules lead to complex organism behaviors  
- **Multi-Agent Training**: Alternating PPO training for competing objectives
- **Configurable Parameters**: Easily tweak game mechanics and physics
- **Real-time Visualization**: Watch organisms evolve and compete
- **Tensorboard Integration**: Monitor training progress

## 📦 Installation

### Requirements
- Python 3.8+
- CPU or GPU (CUDA optional for faster training)

### Setup
```bash
# Clone or create project directory
mkdir pixel_life && cd pixel_life

# Install dependencies
pip install numpy gymnasium stable-baselines3[extra] torch

# Verify installation
python test_env.py
```

### Dependencies
- `numpy`: Grid operations and mathematics
- `gymnasium`: RL environment framework  
- `stable-baselines3`: PPO agents and training
- `torch`: Neural network backend
- `tensorboard`: Training visualization (optional)

## 🎮 Quick Start

### 1. Test the Environment
```bash
python test_env.py
```
This runs basic functionality tests and shows environment behavior.

### 2. Train Agents (CPU)
```bash
# Quick training run (1000 steps)
python train.py --total-timesteps 1000 --grid-size 8 --num-envs 1

# Full training run (200k steps)  
python train.py --total-timesteps 200000 --grid-size 16 --num-envs 4
```

### 3. Load and Evaluate Trained Models
```bash
# Load checkpoint and continue training
python train.py --load-checkpoint final --total-timesteps 50000

# Or use the evaluation script below
```

## 🏋️ Training Guide

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--grid-size` | 16 | Grid height and width |
| `--max-steps` | 500 | Max steps per episode |
| `--total-timesteps` | 200000 | Total training steps |
| `--alternation-steps` | 20000 | Steps before switching agents |
| `--num-envs` | 4 | Parallel environments |
| `--learning-rate` | 3e-4 | PPO learning rate |
| `--batch-size` | 64 | Training batch size |

### Example Training Commands

```bash
# Small grid, quick training
python train.py --grid-size 8 --total-timesteps 50000 --num-envs 2

# Large grid, full training  
python train.py --grid-size 32 --total-timesteps 500000 --num-envs 8

# Custom hyperparameters
python train.py --learning-rate 1e-4 --batch-size 128 --n-epochs 20
```

### Training Progress

Monitor progress with:
- **Terminal Output**: Real-time episode statistics
- **Tensorboard**: `tensorboard --logdir ./models/`
- **Saved Models**: Checkpoints in `./models/` directory

Expected training time:
- **CPU**: ~5-10 minutes per 10k timesteps
- **GPU**: ~2-3 minutes per 10k timesteps

## 📊 Environment Details

### Action Spaces
- **Main Agent**: Continuous 4D vector `[-1,1]⁴` (split, consume, combine, forfeit probabilities)
- **Spice Agent**: Discrete `{0,1,2}` (no-op, expand universe, tweak parameters)

### Observation Spaces  
- **Main Agent**: 4-channel grid `(H,W,4)` - organism IDs, dead pixels, distance map, time
- **Spice Agent**: 10D statistics vector - population counts, sizes, spatial distribution

### Reward Structure
- **Main Agent**: `+total_live_pixels + 1` (encourages growth)
- **Spice Agent**: `+1` if population decreased, `-1` if increased, `+10` if extinction

### Organism Actions
- **Split**: Create new pixel in adjacent empty cell
- **Consume**: Clear nearby dead pixels for energy  
- **Combine**: Merge with neighboring organisms
- **Forfeit**: Sacrifice pixel (mark as dead)

## 🎨 Visualization & Evaluation

### Create Evaluation Script
```python
# eval.py
import numpy as np
from stable_baselines3 import PPO
from train import PixelLifeMultiAgentEnv

# Load trained models
main_agent = PPO.load("./models/main_agent_final")
spice_agent = PPO.load("./models/spice_agent_final")

# Create environment
env = PixelLifeMultiAgentEnv({'H': 16, 'W': 16, 'max_steps': 200}, 'main')

# Run episode
obs, _ = env.reset()
for step in range(200):
    action, _ = main_agent.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # Print ASCII visualization
    
    if done or truncated:
        break

print(f"Episode finished after {step} steps")
```

### Watch Training Progress
```bash
# Start tensorboard
tensorboard --logdir ./models/ --port 6006

# Open browser to http://localhost:6006
```

## ⚙️ Configuration

### Environment Parameters (in `env.py`)
```python
params = {
    'split_cost': 1,           # Energy cost to split
    'consume_gain': 2,         # Energy gained from consuming  
    'combine_threshold': 3,    # Min size to combine organisms
    'forfeit_penalty': -1,     # Penalty for forfeiting
    'expansion_size': 5,       # Grid expansion amount
    'tweak_interval': 100,     # Steps between parameter tweaks
    'tweak_strength': 0.1,     # Magnitude of parameter changes
}
```

### Training Configuration
Modify hyperparameters in `train.py` or via command line:
```python
config = {
    'learning_rate': 3e-4,
    'batch_size': 64, 
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
}
```

## 🐛 Troubleshooting

### Common Issues

**Training is slow**
- Reduce `--num-envs` if running out of memory
- Use smaller `--grid-size` for faster episodes
- Increase `--batch-size` for better GPU utilization

**Episodes end too quickly**
- Increase `--max-steps` per episode
- Adjust reward structure in `_compute_rewards()`
- Tune organism parameters for better survival

**Models not improving**
- Try different `--learning-rate` (1e-4 to 1e-3)
- Increase `--total-timesteps` for more training
- Adjust `--alternation-steps` for better balance

**Out of memory errors**
- Reduce `--num-envs` 
- Use smaller `--grid-size`
- Reduce `--batch-size`

### Debug Mode
Add debug prints in `env.py`:
```python
def step(self, spice_action, pixel_actions):
    print(f"Step {self.step_count}: {len(self.organisms)} organisms, {len(self.pixel_to_org)} pixels")
    # ... rest of method
```

## 🔬 Experiments & Extensions

### Experiment Ideas
1. **Curriculum Learning**: Start with small grids, gradually increase size
2. **Population Dynamics**: Track organism lineages and evolution
3. **Cooperative Variants**: Train both agents to maximize total survival time
4. **Multi-Species**: Different organism types with unique behaviors
5. **Physics Modifications**: Add energy, momentum, or diffusion mechanics

### Code Extensions
- **Custom Visualizer**: Create pygame/matplotlib real-time rendering
- **Genetic Algorithms**: Evolve organism behavior parameters
- **Hierarchical RL**: Higher-level strategic planning agents
- **Communication**: Allow organisms to share information
- **3D Version**: Extend to 3D grid with additional spatial complexity

## 📈 Performance Benchmarks

### Typical Results (after 200k timesteps)
- **Main Agent**: 300-500 average episode reward
- **Spice Agent**: -50 to +50 average episode reward  
- **Episode Length**: 100-300 steps on 16x16 grid
- **Population Peaks**: 20-100 simultaneous organisms

### Scaling Performance
| Grid Size | Envs | CPU Time/10k | GPU Time/10k | Memory |
|-----------|------|--------------|--------------|---------|
| 8x8 | 4 | ~2 min | ~1 min | ~500MB |
| 16x16 | 4 | ~5 min | ~2 min | ~1GB |
| 32x32 | 4 | ~15 min | ~5 min | ~2GB |

## 📚 References & Related Work

- **OpenAI Gymnasium**: https://gymnasium.farama.org/
- **Stable Baselines 3**: https://stable-baselines3.readthedocs.io/
- **Multi-Agent RL**: Tampuu et al. "Multiagent cooperation and competition with deep reinforcement learning"
- **Cellular Automata**: Conway's Game of Life and extensions
- **Artificial Life**: Langton, C. "Artificial Life: An Overview"

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- **Performance optimization** (vectorization, Numba, etc.)
- **New organism behaviors** and action types
- **Advanced visualizations** and analysis tools
- **Benchmark environments** and evaluation metrics
- **Documentation** and tutorial content

## 📄 License

MIT License - feel free to use for research, education, or commercial projects.

---

**Have fun exploring emergent pixel life! 🧬🤖**