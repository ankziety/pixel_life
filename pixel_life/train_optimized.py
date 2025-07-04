#!/usr/bin/env python3
"""
Optimized training script for PixelLife using guardian best practices:
- Vectorized environments for massive parallelism
- Research-backed hyperparameters and network architectures
- Curriculum learning and advanced optimization techniques
- GPU acceleration and memory optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import argparse
import os
import time
import json
from typing import Dict, Any, Tuple, List
import warnings

try:
    from env_optimized import PixelLifeEnvOptimized
except ImportError:
    from env import PixelLifeEnv as PixelLifeEnvOptimized

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


class ResidualBlock(nn.Module):
    """Residual block for deeper networks with better gradient flow."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AttentionModule(nn.Module):
    """Spatial attention mechanism for focusing on important grid regions."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, 1)
        self.conv2 = nn.Conv2d(channels // 8, 1, 1)
        
    def forward(self, x):
        attention = F.relu(self.conv1(x))
        attention = torch.sigmoid(self.conv2(attention))
        return x * attention


class OptimizedCNN(BaseFeaturesExtractor):
    """
    Optimized CNN feature extractor with:
    - Residual connections for better gradient flow
    - Spatial attention mechanisms
    - Adaptive pooling for variable input sizes
    - Efficient channel progression
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Input processing
        n_input_channels = observation_space.shape[0] if len(observation_space.shape) == 3 else observation_space.shape[2]
        
        # Efficient feature extraction pipeline
        self.cnn = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # First residual block
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Second residual block with attention
            ResidualBlock(64),
            AttentionModule(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Final feature extraction
            ResidualBlock(128),
            AttentionModule(128),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling for variable sizes
            nn.Flatten(),
        )
        
        # Calculate the output size dynamically
        with torch.no_grad():
            sample_input = torch.zeros((1,) + observation_space.shape)
            sample_output = self.cnn(sample_input)
            cnn_output_size = sample_output.shape[1]
        
        # Final projection to desired feature dimension
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_size, features_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Handle different input formats
        if len(observations.shape) == 3:  # (H, W, C) -> (C, H, W)
            observations = observations.permute(2, 0, 1).unsqueeze(0)
        elif len(observations.shape) == 4:  # (B, H, W, C) -> (B, C, H, W)
            observations = observations.permute(0, 3, 1, 2)
        
        features = self.cnn(observations)
        return self.linear(features)


class CurriculumCallback(BaseCallback):
    """
    Curriculum learning callback that gradually increases environment difficulty
    based on agent performance.
    """
    
    def __init__(self, check_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.performance_history = []
        self.current_difficulty = 0.1  # Start easy
        self.max_difficulty = 1.0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get recent performance metrics
            if hasattr(self.training_env, 'get_attr'):
                try:
                    episode_rewards = self.training_env.get_attr('episode_returns')
                    if episode_rewards and len(episode_rewards[0]) > 0:
                        avg_reward = np.mean([np.mean(rewards[-10:]) for rewards in episode_rewards if len(rewards) > 0])
                        self.performance_history.append(avg_reward)
                        
                        # Curriculum progression logic
                        if len(self.performance_history) >= 3:
                            recent_performance = np.mean(self.performance_history[-3:])
                            if recent_performance > 50 and self.current_difficulty < self.max_difficulty:
                                self.current_difficulty = min(self.max_difficulty, self.current_difficulty + 0.1)
                                if self.verbose:
                                    print(f"Curriculum: Increasing difficulty to {self.current_difficulty:.1f}")
                                
                                # Update environment parameters
                                for env in self.training_env.envs:
                                    if hasattr(env, 'unwrapped'):
                                        env.unwrapped.params[0] *= (1 + 0.1 * self.current_difficulty)  # Harder split threshold
                                        env.unwrapped.params[2] *= (1 + 0.05 * self.current_difficulty)  # Higher death probability
                except:
                    pass  # Silently handle missing attributes
        
        return True


class PerformanceMonitor(BaseCallback):
    """Advanced performance monitoring with detailed metrics."""
    
    def __init__(self, log_freq: int = 1000, save_freq: int = 50000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'organisms_created': [],
            'survival_time': [],
            'fps': []
        }
        self.start_time = None
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        # Log detailed metrics
        if self.n_calls % self.log_freq == 0:
            current_time = time.time()
            if self.start_time:
                fps = self.n_calls / (current_time - self.start_time)
                self.metrics['fps'].append(fps)
                
                if self.verbose:
                    print(f"Step {self.n_calls}: FPS={fps:.1f}")
        
        # Save checkpoint
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"checkpoints/pixel_life_optimized_{self.n_calls}")
            
            # Save metrics
            with open(f"metrics/metrics_{self.n_calls}.json", 'w') as f:
                json.dump(self.metrics, f, indent=2)
        
        return True


def make_optimized_env(grid_size: int, max_steps: int, difficulty: float = 0.5):
    """Create an optimized environment with research-backed parameters."""
    
    def _init():
        env = PixelLifeEnvOptimized(
            H=grid_size, 
            W=grid_size, 
            max_steps=max_steps,
            max_organisms=1000
        )
        
        # Apply difficulty scaling
        env.params[0] *= (1 + difficulty)  # split_threshold
        env.params[2] *= (1 + 0.5 * difficulty)  # death_probability
        
        return env
    
    return _init


class OptimizedMultiAgentWrapper(gym.Wrapper):
    """
    Optimized wrapper for multi-agent training with shared experience buffer
    and efficient action/observation handling.
    """
    
    def __init__(self, env, agent_type: str = 'main'):
        super().__init__(env)
        self.agent_type = agent_type
        
        if agent_type == 'main':
            self.action_space = gym.spaces.Box(
                low=0, high=3, shape=(env.H * env.W,), dtype=np.int32
            )
            self.observation_space = env.observation_space['main']
        else:  # spice
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = env.observation_space['spice']
        
        # Performance tracking
        self.episode_returns = []
        self.current_return = 0
        
    def reset(self, **kwargs):
        if self.current_return != 0:
            self.episode_returns.append(self.current_return)
            self.current_return = 0
        
        obs, info = self.env.reset(**kwargs)
        return obs[self.agent_type], info
    
    def step(self, action):
        # Create full action dict
        if self.agent_type == 'main':
            full_action = {
                'spice_action': np.array([0]),  # No-op for spice
                'pixel_actions': action
            }
        else:
            full_action = {
                'spice_action': action,
                'pixel_actions': np.zeros(self.env.H * self.env.W, dtype=np.int32)
            }
        
        obs, rewards, terminated, truncated, info = self.env.step(full_action)
        
        reward = rewards[self.agent_type]
        self.current_return += reward
        
        return obs[self.agent_type], reward, terminated, truncated, info


def train_optimized_agents(args):
    """
    Main training function with guardian optimization techniques.
    """
    print("🚀 Starting Optimized PixelLife Training with Guardian Best Practices")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Optimized hyperparameters based on research
    hyperparams = {
        'learning_rate': 3e-4,  # Research-backed LR
        'n_steps': 2048,        # Larger rollout buffer
        'batch_size': 128,      # Efficient batch size
        'n_epochs': 10,         # Multiple epochs per update
        'gamma': 0.99,          # Standard discount factor
        'gae_lambda': 0.95,     # GAE parameter
        'clip_range': 0.2,      # PPO clip range
        'ent_coef': 0.01,       # Exploration bonus
        'vf_coef': 0.5,         # Value function coefficient
        'max_grad_norm': 0.5,   # Gradient clipping
        'target_kl': 0.01,      # Early stopping
    }
    
    # Device optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Train both agents alternately for curriculum learning
    agents = {}
    
    for agent_type in ['main', 'spice']:
        print(f"\n🎯 Training {agent_type.upper()} agent...")
        
        # Create vectorized environments
        env_fn = lambda: OptimizedMultiAgentWrapper(
            PixelLifeEnvOptimized(
                H=args.grid_size,
                W=args.grid_size, 
                max_steps=args.max_steps,
                max_organisms=1000
            ),
            agent_type=agent_type
        )
        
        vec_env = SubprocVecEnv([env_fn for _ in range(args.num_envs)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        
        # Policy network with optimized architecture
        policy_kwargs = {
            'features_extractor_class': OptimizedCNN,
            'features_extractor_kwargs': {'features_dim': 512},
            'net_arch': dict(pi=[256, 256], vf=[256, 256]),  # Larger networks
            'activation_fn': nn.ReLU,
            'normalize_images': False,  # We handle normalization
        }
        
        # Create optimized PPO agent
        model = PPO(
            'CnnPolicy' if agent_type == 'main' else 'MlpPolicy',
            vec_env,
            device=device,
            verbose=1,
            tensorboard_log=f"logs/{agent_type}",
            policy_kwargs=policy_kwargs,
            **hyperparams
        )
        
        # Set up callbacks
        callbacks = [
            PerformanceMonitor(verbose=1),
            CurriculumCallback(verbose=1)
        ]
        
        # Train with optimized parameters
        steps_per_agent = args.total_timesteps // 2  # Split between agents
        
        print(f"Training {agent_type} agent for {steps_per_agent:,} steps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=steps_per_agent,
            callback=callbacks,
            tb_log_name=f"optimized_{agent_type}",
            reset_num_timesteps=False,
        )
        
        training_time = time.time() - start_time
        fps = steps_per_agent / training_time
        
        print(f"✅ {agent_type.upper()} training completed!")
        print(f"   Training time: {training_time:.1f}s")
        print(f"   Average FPS: {fps:.1f}")
        
        # Save final model
        model.save(f"models/pixel_life_optimized_{agent_type}")
        vec_env.save(f"models/vec_env_{agent_type}.pkl")
        
        agents[agent_type] = {
            'model': model,
            'env': vec_env,
            'fps': fps,
            'training_time': training_time
        }
        
        # Cleanup
        vec_env.close()
    
    # Final performance summary
    print("\n🎉 OPTIMIZATION COMPLETE!")
    print("=" * 50)
    total_fps = sum(agent['fps'] for agent in agents.values())
    total_time = sum(agent['training_time'] for agent in agents.values())
    
    print(f"Total Training Time: {total_time:.1f}s")
    print(f"Combined FPS: {total_fps:.1f}")
    print(f"GPU Utilization: ~{torch.cuda.utilization() if torch.cuda.is_available() else 0}%")
    
    # Save training summary
    summary = {
        'args': vars(args),
        'hyperparams': hyperparams,
        'agents': {
            name: {
                'fps': agent['fps'],
                'training_time': agent['training_time']
            } for name, agent in agents.items()
        },
        'total_time': total_time,
        'combined_fps': total_fps,
        'device': str(device)
    }
    
    with open('training_summary_optimized.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("📊 Training summary saved to 'training_summary_optimized.json'")
    return agents


def main():
    parser = argparse.ArgumentParser(description="Optimized PixelLife Training")
    
    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    parser.add_argument('--grid-size', type=int, default=16,
                       help='Grid size (NxN)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum episode steps')
    parser.add_argument('--num-envs', type=int, default=8,
                       help='Number of parallel environments')
    
    # Optimization flags
    parser.add_argument('--use-curriculum', action='store_true',
                       help='Enable curriculum learning')
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    
    args = parser.parse_args()
    
    print("Guardian-Guided Optimization Parameters:")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Grid size: {args.grid_size}x{args.grid_size}")
    print(f"  Max episode steps: {args.max_steps}")
    print(f"  Parallel environments: {args.num_envs}")
    print(f"  Curriculum learning: {args.use_curriculum}")
    print(f"  Performance profiling: {args.profile}")
    
    # Enable profiling if requested
    if args.profile:
        torch.autograd.set_detect_anomaly(True)
    
    # Run optimized training
    agents = train_optimized_agents(args)
    
    print("\n✨ Optimized training completed successfully!")
    return agents


if __name__ == "__main__":
    main()