# Pixel Life: A 2D Artificial Life Environment

Pixel Life is a competitive multi-agent Gym environment where a main agent controls pixel organisms that can split, consume, combine, and forfeit, while an adversarial "spice" agent tries to make survival difficult by expanding the universe and tweaking game parameters.

## Features

- **Dual-agent competition**: Main agent controls organisms, spice agent controls environment
- **Dynamic universe**: Grid can expand during gameplay
- **Emergent behaviors**: Organisms can split, consume dead cells or smaller organisms, combine with others, or sacrifice themselves
- **Adaptive difficulty**: Spice agent learns to challenge the main agent by tweaking game rules
- **Reinforcement learning ready**: Compatible with Stable Baselines3 PPO agents

## Project Structure

```
pixel_life/
├── env.py                 # Main environment implementation
├── train.py              # Training script for PPO agents
├── per_pixel_ai.py       # Per-pixel AI system
├── continual_learning.py # Continual learning system
├── basic_renderer.py     # Core rendering functionality
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── demos/               # Demo scripts and examples
├── tests/               # Test files
└── logs/                # Training logs and model checkpoints
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Install from source

```bash
# Clone or create the pixel_life directory
cd pixel_life

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- gymnasium>=1.0.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- stable-baselines3>=2.0.0
- torch>=1.12.0
- numba>=0.56.0
- tensorboard>=2.10.0
- psutil>=5.9.0
- pygame>=2.1.0

## Quick Start

### Test the environment

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
    for coord in env.pixel_to_org.keys():
        action_type = 1  # Split
        direction = 0    # Up
        pixel_actions[coord] = (action_type, direction)
    
    # Step
    obs, rewards, done, info = env.step(spice_action, pixel_actions)
    
    # Render (optional)
    env.render()
    
    if done:
        break
```

## Training Agents

### Train on CPU (Recommended for beginners)

```bash
# Basic training with default parameters
python train.py
```

### Custom Training Parameters

```python
from train import train_pixel_life

# Custom hyperparameters
hyperparams = {
    'total_timesteps': 1_000_000,
    'n_envs': 4,              # Number of parallel environments
    'learning_rate': 3e-4,
    'n_steps': 2048,          # Steps before PPO update
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'device': 'cpu',          # or 'cuda' for GPU
    'log_dir': './logs'
}

# Train
main_model, spice_model, run_dir = train_pixel_life(**hyperparams)
```

### Monitor Training

Training logs are saved to `./logs/pixel_life_logs/run_TIMESTAMP/`. You can monitor training with TensorBoard:

```bash
tensorboard --logdir ./logs/pixel_life_logs
```

## Loading and Using Checkpoints

### Load trained models

```python
from stable_baselines3 import PPO
from env import PixelLifeEnv
from train import PixelLifeWrapper

# Load models
main_model = PPO.load("./logs/pixel_life_logs/run_20240101_120000/main_agent/final_model")
spice_model = PPO.load("./logs/pixel_life_logs/run_20240101_120000/spice_agent/final_model")

# Create environment
env = PixelLifeEnv(H=30, W=30)

# Run evaluation
obs = env.reset()
for step in range(1000):
    obs_main, obs_spice = obs
    
    # Get spice action
    spice_action, _ = spice_model.predict(obs_spice, deterministic=True)
    
    # Get pixel actions (simplified - in practice you'd want per-pixel predictions)
    pixel_actions = {}
    main_action, _ = main_model.predict(obs_main, deterministic=True)
    for coord in env.pixel_to_org.keys():
        pixel_actions[coord] = (int(main_action[0]), int(main_action[1]))
    
    # Step
    obs, rewards, done, info = env.step(int(spice_action), pixel_actions)
    
    if done:
        print(f"Episode ended at step {step}")
        break
```

## Visualizing Rollouts

### Real-time visualization

```python
from env import PixelLifeEnv
import matplotlib.pyplot as plt

env = PixelLifeEnv(H=30, W=30)
obs = env.reset()

# Enable interactive mode
plt.ion()

for step in range(500):
    # Your agent actions here
    spice_action = 0  # No-op
    pixel_actions = {coord: (1, step % 4) for coord in env.pixel_to_org.keys()}
    
    obs, rewards, done, info = env.step(spice_action, pixel_actions)
    
    # Render
    env.render()
    
    if done:
        break

plt.ioff()
plt.show()
```

## Demo Scripts

Check out the `demos/` directory for various examples:

- `demo.py` - Basic environment demonstration
- `demo_ai.py` - AI agent demonstration
- `demo_per_pixel.py` - Per-pixel AI system demo
- `demo_continual_learning.py` - Continual learning demo
- `demo_pygame.py` - Pygame-based visualization

## Testing

Run tests from the `tests/` directory:

```bash
python tests/test_env.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.