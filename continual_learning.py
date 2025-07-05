"""Continual Learning System for Pixel Life Environment.

This system implements continuous learning as described in:
- https://mlconf.com/blog/how-to-apply-continual-learning-to-your-machine-learning-models/
- https://docs.aiaengine.com/guides/continuous-learning/

The system monitors model performance, collects new data, and automatically retrains
when performance degrades or new patterns emerge.
"""

import os
import time
import json
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import threading
import queue
from typing import Dict, List, Tuple, Optional
import argparse

import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from env import PixelLifeEnv
from train import PixelLifeWrapper, make_env
from basic_renderer import PixelLifeRenderer


class PerformanceMonitor:
    """Monitors model performance and detects degradation."""
    
    def __init__(self, window_size=100, degradation_threshold=0.1):
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.survival_rates = deque(maxlen=window_size)
        
    def add_episode(self, total_reward: float, episode_length: int, 
                   final_pixels: int, initial_pixels: int):
        """Add episode results to monitoring."""
        self.rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        
        # Calculate survival rate
        survival_rate = final_pixels / max(initial_pixels, 1)
        self.survival_rates.append(survival_rate)
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if len(self.rewards) < 10:  # Need minimum data
            return {
                'avg_reward': 0.0,
                'avg_episode_length': 0.0,
                'avg_survival_rate': 0.0,
                'performance_score': 0.0
            }
        
        avg_reward = np.mean(list(self.rewards))
        avg_episode_length = np.mean(list(self.episode_lengths))
        avg_survival_rate = np.mean(list(self.survival_rates))
        
        # Combined performance score
        performance_score = (avg_reward * 0.4 + 
                           avg_episode_length * 0.3 + 
                           avg_survival_rate * 0.3)
        
        return {
            'avg_reward': float(avg_reward),
            'avg_episode_length': float(avg_episode_length),
            'avg_survival_rate': float(avg_survival_rate),
            'performance_score': float(performance_score)
        }
    
    def should_retrain(self) -> bool:
        """Determine if retraining is needed based on performance degradation."""
        if len(self.rewards) < self.window_size:
            return False
            
        # Calculate performance trend
        recent_performance = np.mean(list(self.rewards)[-self.window_size//2:])
        older_performance = np.mean(list(self.rewards)[:self.window_size//2])
        
        degradation = (older_performance - recent_performance) / max(float(abs(older_performance)), 1e-6)
        
        return bool(degradation > self.degradation_threshold)


class DataCollector:
    """Collects and stores training data for continual learning."""
    
    def __init__(self, max_buffer_size=10000):
        self.max_buffer_size = max_buffer_size
        self.episode_data = deque(maxlen=max_buffer_size)
        self.performance_history = []
        
    def add_episode_data(self, observations: List, actions: List, 
                        rewards: List, info: Dict):
        """Add episode data to the buffer."""
        episode_data = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'info': info,
            'timestamp': datetime.now().isoformat()
        }
        self.episode_data.append(episode_data)
        
    def get_recent_data(self, n_episodes: int = 100) -> List:
        """Get recent episode data for retraining."""
        return list(self.episode_data)[-n_episodes:]
    
    def save_data(self, filepath: str):
        """Save collected data to file."""
        data = {
            'episodes': list(self.episode_data),
            'performance_history': self.performance_history,
            'metadata': {
                'total_episodes': len(self.episode_data),
                'last_updated': datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_data(self, filepath: str):
        """Load data from file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.episode_data = deque(data.get('episodes', []), 
                                        maxlen=self.max_buffer_size)
                self.performance_history = data.get('performance_history', [])


class ContinualLearningCallback(BaseCallback):
    """Callback for continual learning that monitors and triggers retraining."""
    
    def __init__(self, performance_monitor: PerformanceMonitor, 
                 data_collector: DataCollector, retrain_interval: int = 1000):
        super().__init__()
        self.performance_monitor = performance_monitor
        self.data_collector = data_collector
        self.retrain_interval = retrain_interval
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        """Called after each step."""
        # Collect step data
        self.episode_observations.append(self.locals['observations'])
        self.episode_actions.append(self.locals['actions'])
        self.episode_rewards.append(self.locals['rewards'])
        
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            # Episode ended, process data
            episode_info = {
                'total_reward': sum(self.episode_rewards),
                'episode_length': len(self.episode_rewards),
                'final_pixels': self.locals.get('infos', [{}])[0].get('live_pixels', 0),
                'initial_pixels': 3  # Default initial pixels
            }
            
            # Add to performance monitor
            self.performance_monitor.add_episode(**episode_info)
            
            # Add to data collector
            self.data_collector.add_episode_data(
                self.episode_observations,
                self.episode_actions,
                self.episode_rewards,
                episode_info
            )
            
            # Reset episode data
            self.episode_observations = []
            self.episode_actions = []
            self.episode_rewards = []
            
            # Check if retraining is needed
            if (self.num_timesteps % self.retrain_interval == 0 and 
                self.performance_monitor.should_retrain()):
                print(f"🚨 Performance degradation detected at step {self.num_timesteps}")
                print("🔄 Triggering retraining...")
                return False  # Stop training to trigger retraining
        
        return True


class ContinualLearningSystem:
    """Main continual learning system that orchestrates the entire process."""
    
    def __init__(self, 
                 env_kwargs: Dict = None,
                 log_dir: str = "./continual_learning_logs",
                 retrain_threshold: float = 0.1,
                 retrain_interval: int = 1000,
                 max_episodes_per_retrain: int = 100):
        
        self.env_kwargs = env_kwargs or {'H': 30, 'W': 30, 'max_size': 100}
        self.log_dir = log_dir
        self.retrain_threshold = retrain_threshold
        self.retrain_interval = retrain_interval
        self.max_episodes_per_retrain = max_episodes_per_retrain
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        self.models_dir = os.path.join(log_dir, "models")
        self.data_dir = os.path.join(log_dir, "data")
        self.metrics_dir = os.path.join(log_dir, "metrics")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(degradation_threshold=retrain_threshold)
        self.data_collector = DataCollector()
        
        # Load existing data if available
        data_file = os.path.join(self.data_dir, "training_data.json")
        self.data_collector.load_data(data_file)
        
        # Model state
        self.main_model = None
        self.spice_model = None
        self.current_generation = 0
        
        # Training state
        self.is_training = False
        self.training_queue = queue.Queue()
        
    def initialize_models(self):
        """Initialize or load existing models."""
        main_model_path = os.path.join(self.models_dir, "main_model_latest.zip")
        spice_model_path = os.path.join(self.models_dir, "spice_model_latest.json")
        
        if os.path.exists(main_model_path) and os.path.exists(spice_model_path):
            print("📂 Loading existing models...")
            self.main_model = PPO.load(main_model_path)
            self.spice_model = PPO.load(spice_model_path)
            print("✅ Models loaded successfully")
        else:
            print("🆕 Initializing new models...")
            self._train_initial_models()
    
    def _train_initial_models(self):
        """Train initial models from scratch."""
        print("🎯 Training initial models...")
        
        # Create vectorized environments
        from stable_baselines3.common.vec_env import make_vec_env, DummyVecEnv
        
        main_vec_env = make_vec_env(
            make_env('main', self.env_kwargs),
            n_envs=1,
            vec_env_cls=DummyVecEnv
        )
        spice_vec_env = make_vec_env(
            make_env('spice', self.env_kwargs),
            n_envs=1,
            vec_env_cls=DummyVecEnv
        )
        
        # Create models
        self.main_model = PPO("MlpPolicy", main_vec_env, verbose=1)
        self.spice_model = PPO("MlpPolicy", spice_vec_env, verbose=1)
        
        # Initial training
        self.main_model.learn(total_timesteps=10000)
        self.spice_model.learn(total_timesteps=10000)
        
        # Save models
        self._save_models()
        print("✅ Initial training complete")
    
    def _save_models(self):
        """Save current models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save latest versions
        self.main_model.save(os.path.join(self.models_dir, "main_model_latest"))
        self.spice_model.save(os.path.join(self.models_dir, "spice_model_latest"))
        
        # Save versioned copies
        self.main_model.save(os.path.join(self.models_dir, f"main_model_gen_{self.current_generation}_{timestamp}"))
        self.spice_model.save(os.path.join(self.models_dir, f"spice_model_gen_{self.current_generation}_{timestamp}"))
        
        # Save training data
        data_file = os.path.join(self.data_dir, "training_data.json")
        self.data_collector.save_data(data_file)
        
        # Save performance metrics
        metrics = self.performance_monitor.get_performance_metrics()
        metrics['generation'] = float(self.current_generation)
        metrics['timestamp'] = timestamp
        
        metrics_file = os.path.join(self.metrics_dir, f"metrics_gen_{self.current_generation}_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def run_episode(self, max_steps: int = 1000, render: bool = False) -> Dict:
        """Run a single episode and collect data."""
        env = PixelLifeEnv(**self.env_kwargs)
        
        # Create wrappers
        main_wrapper = PixelLifeWrapper(env, agent_type='main')
        spice_wrapper = PixelLifeWrapper(env, agent_type='spice')
        
        # Connect models
        main_wrapper.other_model = self.spice_model  # type: ignore
        spice_wrapper.other_model = self.main_model  # type: ignore
        
        # Reset
        obs = env.reset()
        initial_pixels = len(env.live_pixels)
        
        # Run episode
        step = 0
        total_reward_main = 0
        total_reward_spice = 0
        observations = []
        actions = []
        rewards = []
        
        while step < max_steps and not env.done:
            # Get AI actions
            main_obs = main_wrapper._flatten_observation(env._get_main_observation())
            spice_obs = spice_wrapper._flatten_observation(env._get_spice_observation())
            
            # Predict actions
            main_action, _ = self.main_model.predict(main_obs, deterministic=False)
            spice_action, _ = self.spice_model.predict(spice_obs, deterministic=False)
            
            # Convert main action to pixel actions
            pixel_actions = main_wrapper._convert_main_action(main_action)
            
            # Execute step
            obs, rewards_step, terminated, truncated, info = env.step(spice_action, pixel_actions)
            done = terminated or truncated
            total_reward_main += rewards_step[0]
            total_reward_spice += rewards_step[1]
            
            # Collect data
            observations.append(main_obs)
            actions.append(main_action)
            rewards.append(rewards_step[0])
            
            step += 1
            
            if done:
                break
        
        # Episode results
        episode_info = {
            'total_reward_main': total_reward_main,
            'total_reward_spice': total_reward_spice,
            'episode_length': step,
            'final_pixels': len(env.live_pixels),
            'initial_pixels': initial_pixels,
            'survival_rate': len(env.live_pixels) / max(initial_pixels, 1)
        }
        
        # Add to monitoring
        self.performance_monitor.add_episode(
            total_reward_main, step, len(env.live_pixels), initial_pixels
        )
        
        # Add to data collector
        self.data_collector.add_episode_data(observations, actions, rewards, episode_info)
        
        return episode_info
    
    def run_continual_learning(self, total_episodes: int = 1000, 
                              episodes_per_retrain: int = 100):
        """Run the continual learning loop."""
        print("🚀 Starting continual learning system...")
        print(f"📊 Target episodes: {total_episodes}")
        print(f"🔄 Retrain every: {episodes_per_retrain} episodes")
        
        self.initialize_models()
        
        episode = 0
        last_retrain_episode = 0
        
        try:
            while episode < total_episodes:
                # Run episode
                episode_info = self.run_episode()
                episode += 1
                
                # Print progress
                if episode % 10 == 0:
                    metrics = self.performance_monitor.get_performance_metrics()
                    print(f"Episode {episode}/{total_episodes}")
                    print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
                    print(f"  Avg Survival Rate: {metrics['avg_survival_rate']:.2f}")
                    print(f"  Performance Score: {metrics['performance_score']:.2f}")
                    print(f"  Current Episode: {episode_info['final_pixels']} pixels survived")
                
                # Check if retraining is needed
                if (episode - last_retrain_episode >= episodes_per_retrain and 
                    self.performance_monitor.should_retrain()):
                    
                    print(f"\n🔄 Retraining at episode {episode}")
                    print("📈 Performance degradation detected")
                    
                    # Retrain models
                    self._retrain_models()
                    last_retrain_episode = episode
                    self.current_generation += 1
                    
                    # Save models and data
                    self._save_models()
                    
                    print(f"✅ Retraining complete. Generation {self.current_generation}")
                
        except KeyboardInterrupt:
            print("\n⏹️ Continual learning interrupted by user")
        
        # Final save
        self._save_models()
        print(f"\n🎉 Continual learning complete!")
        print(f"📊 Final generation: {self.current_generation}")
        print(f"📁 Models saved to: {self.models_dir}")
    
    def _retrain_models(self):
        """Retrain models using collected data."""
        print("🎯 Starting model retraining...")
        
        # Get recent data for retraining
        recent_data = self.data_collector.get_recent_data(self.max_episodes_per_retrain)
        
        if len(recent_data) < 10:
            print("⚠️ Not enough data for retraining, continuing...")
            return
        
        # Create vectorized environments for retraining
        from stable_baselines3.common.vec_env import make_vec_env, DummyVecEnv
        
        main_vec_env = make_vec_env(
            make_env('main', self.env_kwargs),
            n_envs=1,
            vec_env_cls=DummyVecEnv
        )
        spice_vec_env = make_vec_env(
            make_env('spice', self.env_kwargs),
            n_envs=1,
            vec_env_cls=DummyVecEnv
        )
        
        # Create new models (or continue training existing ones)
        new_main_model = PPO("MlpPolicy", main_vec_env, verbose=1)
        new_spice_model = PPO("MlpPolicy", spice_vec_env, verbose=1)
        
        # Transfer learning: copy weights from previous models
        if self.main_model is not None:
            new_main_model.set_parameters(self.main_model.get_parameters())
        if self.spice_model is not None:
            new_spice_model.set_parameters(self.spice_model.get_parameters())
        
        # Retrain with new data
        print("🔄 Retraining main agent...")
        new_main_model.learn(total_timesteps=5000)
        
        print("🔄 Retraining spice agent...")
        new_spice_model.learn(total_timesteps=5000)
        
        # Update models
        self.main_model = new_main_model
        self.spice_model = new_spice_model
        
        print("✅ Retraining complete")


def main():
    """Main function to run continual learning."""
    parser = argparse.ArgumentParser(description='Continual Learning for Pixel Life')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Total episodes to run (default: 1000)')
    parser.add_argument('--retrain-interval', type=int, default=100,
                       help='Episodes between retraining (default: 100)')
    parser.add_argument('--retrain-threshold', type=float, default=0.1,
                       help='Performance degradation threshold (default: 0.1)')
    parser.add_argument('--log-dir', type=str, default='./continual_learning_logs',
                       help='Directory for logs (default: ./continual_learning_logs)')
    parser.add_argument('--env-size', type=int, default=30,
                       help='Environment size (default: 30)')
    
    args = parser.parse_args()
    
    # Create continual learning system
    env_kwargs = {'H': args.env_size, 'W': args.env_size, 'max_size': 100}
    
    cl_system = ContinualLearningSystem(
        env_kwargs=env_kwargs,
        log_dir=args.log_dir,
        retrain_threshold=args.retrain_threshold,
        retrain_interval=args.retrain_interval
    )
    
    # Run continual learning
    cl_system.run_continual_learning(
        total_episodes=args.episodes,
        episodes_per_retrain=args.retrain_interval
    )


if __name__ == "__main__":
    main() 