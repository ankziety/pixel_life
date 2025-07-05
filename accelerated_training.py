"""
Accelerated Training for Pixel Life using Apple Silicon

This module provides optimized training using PyTorch with MPS (Metal Performance Shaders)
for faster neural network training on Apple Silicon Macs.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from apple_acceleration import create_accelerated_env, accelerator, is_apple_silicon

class AcceleratedPixelLifeWrapper(gym.Env):
    """Accelerated wrapper for Pixel Life environment with Apple Silicon optimizations."""
    
    def __init__(self, env, agent_type='main'):
        self.env = env
        self.agent_type = agent_type
        self.other_model = None
        
        # Copy relevant attributes and flatten action space for SB3 compatibility
        if agent_type == 'main':
            self.action_space = gym.spaces.Discrete(5 * 4)  # 20 possible actions
        else:
            self.action_space = gym.spaces.Discrete(6)  # Spice actions
        
        # Use max_size for observation space
        max_H = env.max_size
        max_W = env.max_size
        grid_size = max_H * max_W
        energy_size = max_H * max_W
        params_size = 7  # Number of parameters
        tick_size = 1
        total_size = grid_size + energy_size + params_size + tick_size
        
        self._max_H = max_H
        self._max_W = max_W
        self._params_size = params_size
        self._tick_size = tick_size
        self.observation_space = gym.spaces.Box(
            low=-1, 
            high=10000, 
            shape=(total_size,), 
            dtype=np.float32
        )
        self.metadata = env.metadata if hasattr(env, 'metadata') else {}
        
    def reset(self, seed=None, options=None):
        obs = self.env.reset(seed=seed)
        return self._flatten_observation(obs), {}
    
    def _flatten_observation(self, obs_dict):
        """Flatten dict observation to array for SB3 compatibility, always max size."""
        grid = obs_dict['grid']
        energy = obs_dict['energy']
        
        # Pad grid and energy to max size
        padded_grid = np.zeros((self._max_H, self._max_W), dtype=np.float32)
        padded_energy = np.zeros((self._max_H, self._max_W), dtype=np.float32)
        h, w = grid.shape
        padded_grid[:h, :w] = grid.astype(np.float32)
        padded_energy[:h, :w] = energy.astype(np.float32)
        
        grid_flat = padded_grid.flatten()
        energy_flat = padded_energy.flatten()
        params = obs_dict['params'].astype(np.float32)
        tick = obs_dict['tick'].astype(np.float32)
        
        return np.concatenate([grid_flat, energy_flat, params, tick])
    
    def step(self, action):
        # Get action from the other agent
        if self.other_model is not None:
            other_obs = self._get_other_obs()
            prediction = self.other_model.predict(other_obs, deterministic=False)
            other_action = prediction[0]
        else:
            # Random actions if other model not set
            if self.agent_type == 'main':
                # We are main, need spice action
                other_action = 0  # No spice action
            else:
                # We are spice, need pixel actions
                other_action = {}
                for coord in self.env.live_pixels:
                    other_action[coord] = (
                        np.random.randint(0, 5),
                        np.random.randint(0, 4)
                    )
        
        # Execute step based on agent type
        if self.agent_type == 'main':
            # Convert main action to pixel actions dict
            pixel_actions = self._convert_main_action(action)
            obs, rewards, done, truncated, info = self.env.step(other_action, pixel_actions)
            reward = rewards[0]  # Main reward
        else:
            # action is spice action, other_action is pixel actions
            obs, rewards, done, truncated, info = self.env.step(action, other_action)
            reward = rewards[1]  # Spice reward
            
        return self._flatten_observation(obs), reward, done, False, info
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def _get_other_obs(self):
        """Get observation for the other agent."""
        obs_dict = self.env._get_observation()
        return self._flatten_observation(obs_dict)
    
    def _convert_main_action(self, action):
        """Convert main agent's flattened action to pixel actions dict."""
        pixel_actions = {}
        
        # Convert flattened action (0-19) back to (action_type, direction)
        base_action_type = action // 4  # 0-4
        base_direction = action % 4     # 0-3
        
        # Generate per-pixel actions using the base action as a seed
        for i, coord in enumerate(self.env.live_pixels):
            # Use pixel position and base action to generate unique action
            y, x = coord
            
            # Create variation based on pixel position and base action
            action_variation = (y * 7 + x * 11 + base_action_type * 13) % 20
            action_type = action_variation // 4
            direction = action_variation % 4
            
            # Ensure valid action types (0-4)
            action_type = min(action_type, 4)
            direction = min(direction, 3)
            
            pixel_actions[coord] = (action_type, direction)
                
        return pixel_actions

def make_accelerated_env(agent_type='main', env_kwargs=None):
    """Create accelerated environment factory for vectorized environments."""
    if env_kwargs is None:
        env_kwargs = {}
        
    def _init():
        env = create_accelerated_env(**env_kwargs)
        wrapped = AcceleratedPixelLifeWrapper(env, agent_type)
        return Monitor(wrapped)
    
    return _init

def train_accelerated_pixel_life(
    total_timesteps=1_000_000,
    n_envs=4,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    checkpoint_freq=10000,
    eval_freq=5000,
    log_dir='./accelerated_logs',
    no_tensorboard=False,
    buffer_size=10000,
    env_kwargs=None,
):
    """Train both main and spice agents using Apple Silicon acceleration.
    
    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        n_steps: Number of steps to collect before update
        batch_size: Minibatch size for PPO
        n_epochs: Number of epochs for PPO
        gamma: Discount factor
        checkpoint_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        log_dir: Directory for logs and checkpoints
    """
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Set device for training
    device = accelerator.device
    print(f"🚀 Training on device: {device}")
    
    # Environment configuration - use passed kwargs or defaults
    if env_kwargs is None:
        env_kwargs = {'H': 50, 'W': 50, 'max_size': 100}
    
    # Create vectorized environments
    print("Creating vectorized environments...")
    try:
        main_env = SubprocVecEnv([make_accelerated_env('main', env_kwargs) for _ in range(n_envs)])
        spice_env = SubprocVecEnv([make_accelerated_env('spice', env_kwargs) for _ in range(n_envs)])
    except Exception as e:
        print(f"SubprocVecEnv failed, trying DummyVecEnv: {e}")
        main_env = DummyVecEnv([make_accelerated_env('main', env_kwargs) for _ in range(n_envs)])
        spice_env = DummyVecEnv([make_accelerated_env('spice', env_kwargs) for _ in range(n_envs)])
    
    # Create evaluation environments
    eval_main_env = DummyVecEnv([make_accelerated_env('main', env_kwargs)])
    eval_spice_env = DummyVecEnv([make_accelerated_env('spice', env_kwargs)])
    
    # Set up callbacks
    callbacks = []
    
    if checkpoint_freq > 0:
        main_checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq // n_envs,
            save_path=os.path.join(log_dir, "main_model"),
            name_prefix="main_model"
        )
        spice_checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq // n_envs,
            save_path=os.path.join(log_dir, "spice_model"),
            name_prefix="spice_model"
        )
        callbacks.extend([main_checkpoint_callback, spice_checkpoint_callback])
    
    if eval_freq > 0:
        main_eval_callback = EvalCallback(
            eval_main_env,
            best_model_save_path=os.path.join(log_dir, "best_main_model"),
            log_path=os.path.join(log_dir, "eval_main"),
            eval_freq=eval_freq // n_envs,
            deterministic=True,
            render=False
        )
        spice_eval_callback = EvalCallback(
            eval_spice_env,
            best_model_save_path=os.path.join(log_dir, "best_spice_model"),
            log_path=os.path.join(log_dir, "eval_spice"),
            eval_freq=eval_freq // n_envs,
            deterministic=True,
            render=False
        )
        callbacks.extend([main_eval_callback, spice_eval_callback])
    
    # Create models with device specification
    print("Creating PPO models...")
    main_model = PPO(
        "MlpPolicy",
        main_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard") if not no_tensorboard else None,
        device=device
    )
    
    spice_model = PPO(
        "MlpPolicy",
        spice_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard") if not no_tensorboard else None,
        device=device
    )
    
    # Link models for coordinated training
    # Note: SubprocVecEnv doesn't expose envs directly, so we'll use a different approach
    # The models will be linked through the environment factory
    
    # Training loop
    print(f"🎯 Starting accelerated training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    # Train both models
    main_model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    spice_model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Save final models
    main_model.save(os.path.join(log_dir, "final_main_model"))
    spice_model.save(os.path.join(log_dir, "final_spice_model"))
    
    # Close environments
    main_env.close()
    spice_env.close()
    eval_main_env.close()
    eval_spice_env.close()
    
    return main_model, spice_model

class AcceleratedNeuralNetwork(nn.Module):
    """Custom neural network optimized for Apple Silicon."""
    
    def __init__(self, input_size, hidden_size=256, output_size=20):
        super(AcceleratedNeuralNetwork, self).__init__()
        
        self.device = accelerator.device
        
        # Use multiple layers for better representation
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        ).to(self.device)
        
        # Initialize weights for better training
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)
        
        return self.layers(x)

def train_custom_model(
    env_size=50,
    episodes=1000,
    learning_rate=1e-3,
    batch_size=32,
    hidden_size=256,
    log_dir='./custom_model_logs'
):
    """Train a custom neural network model using Apple Silicon acceleration."""
    
    print(f"🎯 Training custom model on {accelerator.device}...")
    
    # Create environment
    env = create_accelerated_env(H=env_size, W=env_size)
    
    # Create model
    input_size = env_size * env_size * 2 + 7 + 1  # grid + energy + params + tick
    model = AcceleratedNeuralNetwork(input_size, hidden_size, 20)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training data collection
    observations = []
    actions = []
    rewards = []
    
    print("Collecting training data...")
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(100):  # Max 100 steps per episode
            # Flatten observation
            obs_flat = np.concatenate([
                obs['grid'].flatten(),
                obs['energy'].flatten(),
                obs['params'],
                obs['tick']
            ])
            
            # Get model prediction
            with torch.no_grad():
                action_probs = torch.softmax(model(obs_flat), dim=0)
                action = torch.multinomial(action_probs, 1).item()
            
            # Execute action
            pixel_actions = {}
            for coord in env.live_pixels:
                pixel_actions[coord] = (action // 4, action % 4)
            
            obs, reward, done, truncated, info = env.step(0, pixel_actions)
            episode_reward += reward[0]
            
            # Store experience
            observations.append(obs_flat)
            actions.append(action)
            rewards.append(reward[0])
            
            if done:
                break
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.2f}")
    
    # Convert to tensors
    observations = torch.tensor(observations, dtype=torch.float32, device=accelerator.device)
    actions = torch.tensor(actions, dtype=torch.long, device=accelerator.device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=accelerator.device)
    
    # Normalize rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Create dataset
    dataset = TensorDataset(observations, actions, rewards)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    print("Training neural network...")
    model.train()
    
    for epoch in range(10):
        total_loss = 0
        for batch_obs, batch_actions, batch_rewards in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch_obs)
            
            # Calculate loss with reward weighting
            loss = criterion(logits, batch_actions)
            loss = loss * batch_rewards.mean()  # Weight by average reward
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")
    
    # Save model
    os.makedirs(log_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(log_dir, "custom_model.pth"))
    print(f"✅ Custom model saved to {log_dir}")
    
    return model

def benchmark_training_speed():
    """Benchmark training speed improvements."""
    print("🏃‍♂️ Benchmarking training speed...")
    
    # Test with different environment sizes
    env_sizes = [30, 50, 100]
    timesteps = 10000
    
    for size in env_sizes:
        print(f"\nTesting environment size: {size}x{size}")
        
        # Test accelerated training
        start_time = time.time()
        env = create_accelerated_env(H=size, W=size)
        
        for _ in range(timesteps):
            obs = env.reset()
            for step in range(50):
                pixel_actions = {}
                for coord in env.live_pixels:
                    pixel_actions[coord] = (np.random.randint(0, 4), np.random.randint(0, 4))
                
                obs, rewards, done, truncated, info = env.step(0, pixel_actions)
                if done:
                    break
        
        accelerated_time = time.time() - start_time
        print(f"   Accelerated: {accelerated_time:.3f}s")
        
        # Calculate steps per second
        steps_per_second = timesteps / accelerated_time
        print(f"   Performance: {steps_per_second:.0f} steps/second")

if __name__ == "__main__":
    # Run benchmarks
    benchmark_training_speed()
    
    # Example training
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_accelerated_pixel_life(total_timesteps=100000)
    elif len(sys.argv) > 1 and sys.argv[1] == "custom":
        train_custom_model(episodes=500) 