"""Debug script to check observation space dimensions."""

import numpy as np
from env import PixelLifeEnv
from train import PixelLifeWrapper

# Create environment with training parameters
env = PixelLifeEnv(H=30, W=30, max_size=100)
print(f"Environment: {env.H}x{env.W}")

# Create wrapper
wrapper = PixelLifeWrapper(env, agent_type='main')
print(f"Wrapper observation space: {wrapper.observation_space}")

# Reset and get observation
obs, info = env.reset()
print(f"Raw observation type: {type(obs)}")
print(f"Raw observation: {obs}")

# Get main observation
main_obs = env._get_main_observation()
print(f"Main observation keys: {main_obs.keys()}")
print(f"Grid shape: {main_obs['grid'].shape}")
print(f"Params shape: {main_obs['params'].shape}")
print(f"Tick shape: {main_obs['tick'].shape}")

# Flatten observation
flattened = wrapper._flatten_observation(main_obs)
print(f"Flattened observation shape: {flattened.shape}")
print(f"Expected shape: {wrapper.observation_space.shape}")

# Check if they match
print(f"Shapes match: {flattened.shape == wrapper.observation_space.shape}")

# Print individual sizes
grid_size = main_obs['grid'].shape[0] * main_obs['grid'].shape[1]
params_size = main_obs['params'].shape[0]
tick_size = main_obs['tick'].shape[0]
total_size = grid_size + params_size + tick_size

print(f"\nBreakdown:")
print(f"- Grid: {grid_size} (30x30 = 900)")
print(f"- Params: {params_size}")
print(f"- Tick: {tick_size}")
print(f"- Total: {total_size}")
print(f"- Expected: {wrapper.observation_space.shape[0]}") 