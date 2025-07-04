# Pixel Life: A 2D Artificial Life Environment

Pixel Life is a competitive multi-agent Gym environment where a main agent controls pixel organisms that can split, consume, combine, and forfeit, while an adversarial "spice" agent tries to make survival difficult by expanding the universe and tweaking game parameters.

## Features

- **Dual-agent competition**: Main agent controls organisms, spice agent controls environment
- **Dynamic universe**: Grid can expand during gameplay
- **Emergent behaviors**: Organisms can split, consume dead cells or smaller organisms, combine with others, or sacrifice themselves
- **Adaptive difficulty**: Spice agent learns to challenge the main agent by tweaking game rules
- **Reinforcement learning ready**: Compatible with Stable Baselines3 PPO agents

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
- gym==0.26.2
- numpy>=1.21.0
- matplotlib>=3.5.0
- stable-baselines3>=2.0.0
- torch>=1.12.0

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

Training logs are saved to `./pixel_life_logs/run_TIMESTAMP/`. You can monitor training with TensorBoard:

```bash
tensorboard --logdir ./pixel_life_logs
```

## Loading and Using Checkpoints

### Load trained models

```python
from stable_baselines3 import PPO
from env import PixelLifeEnv
from train import PixelLifeWrapper

# Load models
main_model = PPO.load("./pixel_life_logs/run_20240101_120000/main_agent/final_model")
spice_model = PPO.load("./pixel_life_logs/run_20240101_120000/spice_agent/final_model")

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

### Save rollout as video

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from env import PixelLifeEnv

# Collect frames
env = PixelLifeEnv(H=30, W=30)
frames = []

obs = env.reset()
for step in range(200):
    # Your actions
    spice_action = 0
    pixel_actions = {coord: (1, 0) for coord in env.pixel_to_org.keys()}
    
    obs, rewards, done, info = env.step(spice_action, pixel_actions)
    frames.append(env.grid.copy())
    
    if done:
        break

# Create animation
fig, ax = plt.subplots()
im = ax.imshow(frames[0], cmap='viridis')

def animate(i):
    im.set_array(frames[i])
    return [im]

anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
anim.save('pixel_life_rollout.mp4', writer='ffmpeg')
```

## Environment Details

### Action Spaces

**Main Agent (Pixels)**:
- Action per pixel: `(action_type, direction)`
- `action_type`: 0=no-op, 1=split, 2=consume, 3=combine, 4=forfeit
- `direction`: 0=up, 1=right, 2=down, 3=left

**Spice Agent**:
- Single action: 0=no-op, 1=expand_up, 2=expand_down, 3=expand_left, 4=expand_right, 5=tweak_rule

### Observation Space

Both agents receive:
- `grid`: 2D array where 0=empty, >0=organism ID, -1=dead cell
- `params`: Current game parameters
- `tick`: Current timestep

### Rewards

- **Main agent**: `total_living_pixels + 1`
- **Spice agent**: 
  - +10 if main agent dies
  - +1 if main agent loses pixels
  - -1 if main agent gains pixels
  - 0 otherwise

### Game Parameters (Tweakable by Spice)

- `split_min_size`: Minimum organism size to split
- `split_offspring_size`: Size of new organism after split
- `consume_range`: How far consume can reach
- `combine_max_distance`: Max distance between organisms to combine
- `dead_cell_bonus`: Extra pixels gained from consuming dead cells
- `forfeit_spread`: How many pixels are affected by forfeit

## Advanced Usage

### Custom Environment Configuration

```python
env = PixelLifeEnv(
    H=50,           # Initial height
    W=50,           # Initial width  
    max_size=200    # Maximum dimension after expansions
)
```

### Implement Per-Pixel Policies

For more sophisticated control, implement a policy network that outputs actions for each pixel individually:

```python
class PerPixelPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5 + 4)  # 5 action types + 4 directions
        )
    
    def forward(self, local_obs):
        # local_obs: observation around a single pixel
        logits = self.net(local_obs)
        action_logits = logits[:5]
        direction_logits = logits[5:]
        return action_logits, direction_logits
```

## Troubleshooting

### Common Issues

1. **ImportError with gym**: Make sure you're using gym==0.26.2
2. **CUDA out of memory**: Reduce `n_envs` or use CPU
3. **Training too slow**: Reduce `n_steps` or use fewer environments
4. **Matplotlib backend issues**: Set backend explicitly:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For non-interactive
   ```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is provided as-is for educational and research purposes.