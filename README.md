# 🧬 Pixel Life Environment

A sophisticated 2D cellular automata environment built with Gymnasium for reinforcement learning research. This environment combines Conway's Game of Life mechanics with agent interaction, creating a unique sandbox for exploring emergent behavior and intelligent control systems.

## 🎯 Features

- **Gymnasium-compatible Environment**: Full compliance with OpenAI Gym/Gymnasium standards
- **Cellular Automata Simulation**: Based on Conway's Game of Life with custom modifications
- **Agent Interaction**: Control an agent that can move around and place/remove cells
- **Visual Rendering**: Real-time pygame visualization with cell aging effects
- **Reinforcement Learning Ready**: Pre-configured for training with popular RL libraries
- **Comprehensive Testing**: Full test suite ensuring reliability
- **Multiple RL Algorithms**: Support for PPO, DQN, and A2C training

## 📋 Requirements

- Python 3.8+
- NumPy >= 1.21.0
- Gymnasium >= 0.29.0
- PyGame >= 2.1.0
- Stable Baselines3 >= 2.0.0
- PyTorch >= 1.13.0
- Matplotlib >= 3.5.0

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd pixel_life
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv pixel_life_env
   source pixel_life_env/bin/activate  # On Windows: pixel_life_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Quick Start

### Running the Demo

Try the interactive demo to explore the environment:

```bash
python demo.py
```

**Controls:**
- `0` - Move agent up
- `1` - Move agent down
- `2` - Move agent left
- `3` - Move agent right
- `4` - Place/remove cell at agent position
- `q` - Quit demo

### Training an AI Agent

Train a reinforcement learning agent:

```bash
python train.py
```

Choose from the training options:
1. Train PPO Agent (recommended for beginners)
2. Train DQN Agent
3. Train A2C Agent
4. Compare All Algorithms
5. Evaluate Existing Model
6. Quick Training Test

### Running Tests

Verify everything works correctly:

```bash
python test_env.py
```

## 🧠 Environment Details

### Action Space
- **Type**: Discrete(5)
- **Actions**:
  - 0: Move up
  - 1: Move down
  - 2: Move left
  - 3: Move right
  - 4: Place/remove cell

### Observation Space
- **Type**: Box(0, 1, (grid_size, grid_size, 3), float32)
- **Channels**:
  - Channel 0: Cell presence (0 = empty, 1 = alive)
  - Channel 1: Agent position (1 = agent location, 0 = elsewhere)
  - Channel 2: Cell age (normalized, older cells = higher values)

### Reward Structure
- **Movement**: +0.1 for successful movement
- **Cell Placement**: +1.0 for placing a cell
- **Cell Removal**: -0.5 for removing a cell
- **Population Bonus**: +0.1 × current population
- **Stability Bonus**: +0.5 for maintaining 50-200 cells
- **Overpopulation Penalty**: -1.0 for >300 cells
- **Extinction Penalty**: -2.0 for <10 cells

### Cellular Automata Rules
Based on Conway's Game of Life:
- A live cell with 2-3 neighbors survives
- A dead cell with exactly 3 neighbors becomes alive
- All other cells die or stay dead

## 🎯 Usage Examples

### Basic Environment Usage

```python
from env import PixelLifeEnv
import numpy as np

# Create environment
env = PixelLifeEnv(grid_size=32, max_steps=1000, render_mode='human')

# Reset environment
obs, info = env.reset()

# Take random actions
for _ in range(100):
    action = np.random.randint(0, 5)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Training with Custom Parameters

```python
from train import train_agent

# Train PPO agent with custom settings
model = train_agent(
    algorithm='PPO',
    total_timesteps=200000,
    n_envs=8
)
```

### Evaluating Trained Models

```python
from train import evaluate_agent

# Evaluate a trained model
evaluate_agent(
    model_path='models/pixel_life_ppo_final.zip',
    episodes=10,
    render=True
)
```

## 📊 Performance Benchmarks

Typical training results (100k timesteps):
- **PPO**: ~150-250 average reward
- **DQN**: ~100-200 average reward  
- **A2C**: ~120-180 average reward

Training time (on CPU):
- **PPO (4 envs)**: ~10-15 minutes
- **DQN (1 env)**: ~15-20 minutes
- **A2C (4 envs)**: ~8-12 minutes

## 🔧 Configuration

### Environment Parameters

```python
env = PixelLifeEnv(
    grid_size=32,        # Grid dimensions (32x32)
    max_steps=1000,      # Maximum steps per episode
    render_mode='human'  # 'human', 'rgb_array', or None
)
```

### Training Parameters

Modify hyperparameters in `train.py`:
- Learning rates
- Network architectures
- Batch sizes
- Training frequencies

## 🐛 Troubleshooting

### Common Issues

1. **ImportError for pygame/gymnasium**
   ```bash
   pip install --upgrade pygame gymnasium
   ```

2. **Slow rendering on Linux**
   ```bash
   export SDL_VIDEODRIVER=x11
   ```

3. **CUDA/GPU issues with PyTorch**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Permission errors during installation**
   ```bash
   pip install --user -r requirements.txt
   ```

### Performance Tips

- Use smaller grid sizes (16x16) for faster training
- Disable rendering during training (`render_mode=None`)
- Use multiple environments for parallel training
- Consider using GPU acceleration for larger models

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Run all tests
python test_env.py

# Run specific test class
python -m unittest test_env.TestPixelLifeEnv

# Run with verbose output
python test_env.py -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🎯 Future Improvements

- [ ] Multi-agent support
- [ ] Custom cellular automata rules
- [ ] Network-based training
- [ ] 3D environment extension
- [ ] Advanced visualization options
- [ ] Integration with other RL frameworks

## 📚 Research Applications

This environment is suitable for research in:
- **Emergent Behavior**: Study how simple rules create complex patterns
- **Interactive Evolution**: Explore agent influence on evolutionary systems
- **Reward Engineering**: Test different reward structures for complex goals
- **Multi-objective RL**: Balance population control with pattern creation
- **Curriculum Learning**: Progressive difficulty through grid size/complexity

## 🏆 Achievements

Try to achieve these goals:
- 🌱 **Life Creator**: Maintain a stable population of 100+ cells for 500 steps
- 🎨 **Pattern Master**: Create and maintain oscillating patterns
- 🏃 **Speed Runner**: Complete environment in minimum steps
- 🧬 **Evolution Guide**: Guide evolution to create specific formations
- 🎯 **Perfect Balance**: Achieve optimal population density (100-150 cells)

---

Happy experimenting with artificial life! 🧬✨