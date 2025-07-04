#!/usr/bin/env python3
"""
Training script for PixelLifeEnv with dual PPO agents.
Main agent controls pixels, Spice agent modifies environment.
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import torch
import os
import argparse
from typing import Dict, Any, Tuple
import json
import time

from env import PixelLifeEnv


class PixelLifeMultiAgentEnv(gym.Env):
    """
    Wrapper to make PixelLifeEnv compatible with single-agent RL libraries.
    Alternates between training main agent and spice agent.
    """
    
    def __init__(self, env_config: Dict = None, agent_type: str = 'main'):
        super().__init__()
        
        self.env_config = env_config or {}
        self.agent_type = agent_type  # 'main' or 'spice'
        
        # Create base environment
        self.pixel_env = PixelLifeEnv(**self.env_config)
        
        if agent_type == 'main':
            # Main agent observes the grid
            self.observation_space = self.pixel_env.main_observation_space
            # Main agent controls pixels - use flattened action space for SB3 compatibility
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(4,), dtype=np.float32  # Will be interpreted as action probabilities
            )
        elif agent_type == 'spice':
            # Spice agent observes aggregate statistics
            self.observation_space = self.pixel_env.spice_observation_space
            # Spice agent has 3 discrete actions
            self.action_space = self.pixel_env.spice_action_space
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")
            
        # State for episode management
        self.current_obs = None
        self.episode_rewards = {'main': 0, 'spice': 0}
        
    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation."""
        obs_dict, info = self.pixel_env.reset(seed=seed, options=options)
        self.current_obs = obs_dict
        self.episode_rewards = {'main': 0, 'spice': 0}
        
        # Return observation for this agent
        agent_obs = self.current_obs[self.agent_type]
        
        return agent_obs, info
    
    def step(self, action):
        """Execute one step with current agent's action."""
        
        if self.agent_type == 'main':
            # Convert continuous action to pixel actions
            pixel_actions = self._convert_main_action(action)
            spice_action = 0  # Spice does nothing when main agent is acting
        else:
            # Spice agent action
            spice_action = int(action)
            pixel_actions = {}  # No pixel actions when spice is acting
            
        # Execute step in environment
        obs_dict, rewards_dict, terminated, truncated, info = self.pixel_env.step(
            spice_action, pixel_actions
        )
        
        # Update state
        self.current_obs = obs_dict
        
        # Get reward for current agent
        reward = rewards_dict[self.agent_type]
        self.episode_rewards[self.agent_type] += reward
        
        # Get observation for current agent
        agent_obs = self.current_obs[self.agent_type]
        
        # Add episode reward info
        if terminated or truncated:
            info['episode_rewards'] = self.episode_rewards.copy()
            info['episode_length'] = self.pixel_env.step_count
        
        return agent_obs, reward, terminated, truncated, info
    
    def _convert_main_action(self, action: np.ndarray) -> Dict:
        """Convert continuous action vector to discrete pixel actions."""
        pixel_actions = {}
        
        # Get live pixels
        live_pixels = list(self.pixel_env.pixel_to_org.keys())
        
        if not live_pixels:
            return pixel_actions
        
        # Use action values to determine which pixels act and what they do
        # action is 4D vector: [split_prob, consume_prob, combine_prob, forfeit_prob]
        action_probs = (action + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Apply actions to a subset of pixels based on probabilities
        num_acting_pixels = min(len(live_pixels), max(1, int(len(live_pixels) * 0.3)))
        
        for i in range(num_acting_pixels):
            pixel = live_pixels[i]
            # Choose action based on highest probability
            pixel_action = np.argmax(action_probs)
            pixel_actions[pixel] = pixel_action
            
        return pixel_actions
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.pixel_env.render(mode=mode)


class MultiAgentTrainer:
    """Training manager for both agents."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.save_dir = config.get('save_dir', './models/')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Environment configuration
        env_config = {
            'H': config.get('grid_height', 16),
            'W': config.get('grid_width', 16),
            'max_steps': config.get('max_episode_steps', 500),
        }
        
        # Create environments for both agents
        self.main_env = DummyVecEnv([
            lambda: Monitor(PixelLifeMultiAgentEnv(env_config, 'main'))
            for _ in range(config.get('num_envs', 4))
        ])
        
        self.spice_env = DummyVecEnv([
            lambda: Monitor(PixelLifeMultiAgentEnv(env_config, 'spice'))
            for _ in range(config.get('num_envs', 4))
        ])
        
        # Create agents
        self.main_agent = None
        self.spice_agent = None
        self._create_agents()
        
        # Training state
        self.training_stats = {
            'main_rewards': [],
            'spice_rewards': [],
            'episode_lengths': [],
            'total_episodes': 0
        }
    
    def _create_agents(self):
        """Create PPO agents for both main and spice."""
        
        # Common PPO configuration
        common_config = {
            'learning_rate': self.config.get('learning_rate', 3e-4),
            'n_steps': self.config.get('n_steps', 2048),
            'batch_size': self.config.get('batch_size', 64),
            'n_epochs': self.config.get('n_epochs', 10),
            'gamma': self.config.get('gamma', 0.99),
            'gae_lambda': self.config.get('gae_lambda', 0.95),
            'clip_range': self.config.get('clip_range', 0.2),
            'ent_coef': self.config.get('ent_coef', 0.01),
            'vf_coef': self.config.get('vf_coef', 0.5),
            'max_grad_norm': self.config.get('max_grad_norm', 0.5),
            'verbose': 1,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Create main agent (controls pixels)
        self.main_agent = PPO(
            'MlpPolicy',
            self.main_env,
            **common_config,
            tensorboard_log=f"{self.save_dir}/main_tensorboard/"
        )
        
        # Create spice agent (modifies environment)  
        self.spice_agent = PPO(
            'MlpPolicy',
            self.spice_env,
            **common_config,
            tensorboard_log=f"{self.save_dir}/spice_tensorboard/"
        )
        
        print(f"Created agents on device: {common_config['device']}")
    
    def train(self):
        """Main training loop alternating between agents."""
        total_timesteps = self.config.get('total_timesteps', 100000)
        alternation_steps = self.config.get('alternation_steps', 10000)
        eval_freq = self.config.get('eval_freq', 5000)
        
        print(f"Starting training for {total_timesteps} total timesteps")
        print(f"Alternating every {alternation_steps} steps")
        
        steps_completed = 0
        
        while steps_completed < total_timesteps:
            remaining_steps = total_timesteps - steps_completed
            current_steps = min(alternation_steps, remaining_steps)
            
            # Train main agent
            print(f"\n--- Training Main Agent (steps {steps_completed}-{steps_completed + current_steps}) ---")
            self.main_agent.learn(
                total_timesteps=current_steps,
                reset_num_timesteps=False,
                tb_log_name="main_agent"
            )
            
            steps_completed += current_steps
            
            if steps_completed >= total_timesteps:
                break
                
            remaining_steps = total_timesteps - steps_completed
            current_steps = min(alternation_steps, remaining_steps)
            
            # Train spice agent
            print(f"\n--- Training Spice Agent (steps {steps_completed}-{steps_completed + current_steps}) ---")
            self.spice_agent.learn(
                total_timesteps=current_steps,
                reset_num_timesteps=False,
                tb_log_name="spice_agent"
            )
            
            steps_completed += current_steps
            
            # Periodic evaluation and saving
            if steps_completed % eval_freq == 0:
                self._evaluate_and_save(steps_completed)
        
        # Final save
        self._save_models("final")
        print(f"\nTraining completed! Models saved to {self.save_dir}")
    
    def _evaluate_and_save(self, step: int):
        """Evaluate current policies and save checkpoints."""
        print(f"\n--- Evaluation at step {step} ---")
        
        # Save current models
        checkpoint_name = f"checkpoint_{step}"
        self._save_models(checkpoint_name)
        
        # Quick evaluation with current policies
        eval_stats = self._quick_evaluation()
        
        # Update training stats
        self.training_stats['total_episodes'] = step // 1000  # Rough estimate
        
        # Save training statistics
        with open(f"{self.save_dir}/training_stats.json", 'w') as f:
            json.dump({
                **self.training_stats,
                'eval_stats': eval_stats,
                'last_eval_step': step
            }, f, indent=2)
        
        print(f"Evaluation completed. Stats saved.")
    
    def _quick_evaluation(self, num_episodes: int = 5) -> Dict:
        """Run quick evaluation episodes."""
        print(f"Running {num_episodes} evaluation episodes...")
        
        # Create single evaluation environment
        eval_env = PixelLifeMultiAgentEnv(
            {'H': 12, 'W': 12, 'max_steps': 200}, 
            'main'  # Evaluate from main agent perspective
        )
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < 200:  # Max steps per episode
                # Get action from main agent
                action, _ = self.main_agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        eval_stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episodes': num_episodes
        }
        
        print(f"Eval results: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f} reward, "
              f"{eval_stats['mean_length']:.1f} avg length")
        
        return eval_stats
    
    def _save_models(self, suffix: str):
        """Save both agent models."""
        main_path = f"{self.save_dir}/main_agent_{suffix}"
        spice_path = f"{self.save_dir}/spice_agent_{suffix}"
        
        self.main_agent.save(main_path)
        self.spice_agent.save(spice_path)
        
        print(f"Models saved: {main_path}, {spice_path}")
    
    def load_models(self, suffix: str):
        """Load both agent models."""
        main_path = f"{self.save_dir}/main_agent_{suffix}"
        spice_path = f"{self.save_dir}/spice_agent_{suffix}"
        
        if os.path.exists(main_path + ".zip"):
            self.main_agent = PPO.load(main_path, env=self.main_env)
            print(f"Loaded main agent from {main_path}")
        
        if os.path.exists(spice_path + ".zip"):
            self.spice_agent = PPO.load(spice_path, env=self.spice_env)
            print(f"Loaded spice agent from {spice_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PixelLife Multi-Agent RL')
    
    # Environment settings
    parser.add_argument('--grid-size', type=int, default=16, help='Grid height and width')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')
    
    # Training settings
    parser.add_argument('--total-timesteps', type=int, default=200000, help='Total training timesteps')
    parser.add_argument('--alternation-steps', type=int, default=20000, help='Steps before switching agents')
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel environments')
    
    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs per update')
    
    # Misc
    parser.add_argument('--save-dir', type=str, default='./models/', help='Directory to save models')
    parser.add_argument('--load-checkpoint', type=str, help='Load from checkpoint (suffix)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Configuration
    config = {
        'grid_height': args.grid_size,
        'grid_width': args.grid_size,
        'max_episode_steps': args.max_steps,
        'total_timesteps': args.total_timesteps,
        'alternation_steps': args.alternation_steps,
        'num_envs': args.num_envs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'save_dir': args.save_dir
    }
    
    print("=== PixelLife Multi-Agent Training ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create trainer
    trainer = MultiAgentTrainer(config)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_models(args.load_checkpoint)
    
    # Start training
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    

if __name__ == "__main__":
    main()