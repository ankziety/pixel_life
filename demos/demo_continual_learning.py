"""Demo script for continual learning system."""

import os
import time
import numpy as np
from datetime import datetime
import argparse

from env import PixelLifeEnv
from train import PixelLifeWrapper, make_env
from stable_baselines3 import PPO, DQN


class SimpleContinualLearning:
    """Simplified continual learning system for demonstration."""
    
    def __init__(self, env_size=30, log_dir="./continual_learning_demo"):
        self.env_size = env_size
        self.log_dir = log_dir
        self.env_kwargs = {'H': env_size, 'W': env_size, 'max_size': 100}
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        self.models_dir = os.path.join(log_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Performance tracking
        self.rewards_history = []
        self.survival_rates = []
        self.generation = 0
        
        # Models
        self.main_model = None
        self.spice_model = None
        
    def initialize_models(self):
        """Initialize or load models."""
        main_path = os.path.join(self.models_dir, "main_model_latest.zip")
        spice_path = os.path.join(self.models_dir, "spice_model_latest.json")
        
        if os.path.exists(main_path) and os.path.exists(spice_path):
            print("📂 Loading existing models...")
            self.main_model = PPO.load(main_path)
            self.spice_model = DQN.load(spice_path)
        else:
            print("🆕 Creating new models...")
            self._create_initial_models()
    
    def _create_initial_models(self):
        """Create initial models."""
        # Create environments
        main_env = make_env('main', self.env_kwargs)()
        spice_env = make_env('spice', self.env_kwargs)()
        
        # Create models
        self.main_model = PPO("MlpPolicy", main_env, verbose=0)
        self.spice_model = DQN("MlpPolicy", spice_env, verbose=0)
        
        # Initial training
        print("🎯 Initial training...")
        self.main_model.learn(total_timesteps=5000)
        self.spice_model.learn(total_timesteps=5000)
        
        self._save_models()
    
    def _save_models(self):
        """Save current models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save latest
        self.main_model.save(os.path.join(self.models_dir, "main_model_latest"))
        self.spice_model.save(os.path.join(self.models_dir, "spice_model_latest"))
        
        # Save versioned
        self.main_model.save(os.path.join(self.models_dir, f"main_gen_{self.generation}_{timestamp}"))
        self.spice_model.save(os.path.join(self.models_dir, f"spice_gen_{self.generation}_{timestamp}"))
        
        print(f"💾 Models saved (generation {self.generation})")
    
    def run_episode(self, max_steps=500):
        """Run a single episode and return performance metrics."""
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
        total_reward = 0
        
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
            obs, rewards, done, info = env.step(spice_action, pixel_actions)
            total_reward += rewards[0]
            step += 1
            
            if done:
                break
        
        # Calculate metrics
        final_pixels = len(env.live_pixels)
        survival_rate = final_pixels / max(initial_pixels, 1)
        
        return {
            'total_reward': total_reward,
            'episode_length': step,
            'final_pixels': final_pixels,
            'initial_pixels': initial_pixels,
            'survival_rate': survival_rate
        }
    
    def should_retrain(self, window_size=20):
        """Check if retraining is needed based on recent performance."""
        if len(self.rewards_history) < window_size:
            return False
        
        # Get recent performance
        recent_rewards = self.rewards_history[-window_size//2:]
        older_rewards = self.rewards_history[:window_size//2]
        
        if len(recent_rewards) < 5 or len(older_rewards) < 5:
            return False
        
        recent_avg = np.mean(recent_rewards)
        older_avg = np.mean(older_rewards)
        
        # Check for performance degradation
        degradation = (older_avg - recent_avg) / max(abs(older_avg), 1e-6)
        
        return degradation > 0.1  # 10% degradation threshold
    
    def retrain_models(self):
        """Retrain models with new data."""
        print(f"🔄 Retraining models (generation {self.generation + 1})...")
        
        # Create new environments
        main_env = make_env('main', self.env_kwargs)()
        spice_env = make_env('spice', self.env_kwargs)()
        
        # Create new models
        new_main = PPO("MlpPolicy", main_env, verbose=0)
        new_spice = DQN("MlpPolicy", spice_env, verbose=0)
        
        # Transfer learning (copy weights)
        if self.main_model is not None:
            new_main.set_parameters(self.main_model.get_parameters())
        if self.spice_model is not None:
            new_spice.set_parameters(self.spice_model.get_parameters())
        
        # Retrain
        new_main.learn(total_timesteps=3000)
        new_spice.learn(total_timesteps=3000)
        
        # Update models
        self.main_model = new_main
        self.spice_model = new_spice
        self.generation += 1
        
        # Save
        self._save_models()
        print("✅ Retraining complete")
    
    def run_continual_learning(self, total_episodes=200, retrain_interval=50):
        """Run the continual learning loop."""
        print("🚀 Starting continual learning demo...")
        print(f"📊 Episodes: {total_episodes}")
        print(f"🔄 Retrain interval: {retrain_interval}")
        print(f"🌍 Environment: {self.env_size}x{self.env_size}")
        
        self.initialize_models()
        
        episode = 0
        last_retrain = 0
        
        try:
            while episode < total_episodes:
                # Run episode
                episode_info = self.run_episode()
                episode += 1
                
                # Track performance
                self.rewards_history.append(episode_info['total_reward'])
                self.survival_rates.append(episode_info['survival_rate'])
                
                # Print progress
                if episode % 10 == 0:
                    avg_reward = np.mean(self.rewards_history[-10:])
                    avg_survival = np.mean(self.survival_rates[-10:])
                    print(f"Episode {episode}/{total_episodes}")
                    print(f"  Avg Reward: {avg_reward:.2f}")
                    print(f"  Avg Survival: {avg_survival:.2f}")
                    print(f"  Current: {episode_info['final_pixels']} pixels")
                    print(f"  Generation: {self.generation}")
                
                # Check for retraining
                if (episode - last_retrain >= retrain_interval and 
                    self.should_retrain()):
                    
                    print(f"\n🚨 Performance degradation detected!")
                    print(f"🔄 Retraining at episode {episode}")
                    
                    self.retrain_models()
                    last_retrain = episode
                    
                    # Reset performance tracking
                    self.rewards_history = self.rewards_history[-20:]  # Keep recent
                    self.survival_rates = self.survival_rates[-20:]
                
        except KeyboardInterrupt:
            print("\n⏹️ Continual learning interrupted")
        
        # Final save
        self._save_models()
        
        # Print final stats
        print(f"\n🎉 Continual learning complete!")
        print(f"📊 Final generation: {self.generation}")
        print(f"📁 Models saved to: {self.models_dir}")
        
        if self.rewards_history:
            final_avg_reward = np.mean(self.rewards_history[-10:])
            final_avg_survival = np.mean(self.survival_rates[-10:])
            print(f"📈 Final avg reward: {final_avg_reward:.2f}")
            print(f"📈 Final avg survival: {final_avg_survival:.2f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Continual Learning Demo')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Total episodes (default: 200)')
    parser.add_argument('--retrain-interval', type=int, default=50,
                       help='Episodes between retraining (default: 50)')
    parser.add_argument('--env-size', type=int, default=30,
                       help='Environment size (default: 30)')
    parser.add_argument('--log-dir', type=str, default='./continual_learning_demo',
                       help='Log directory (default: ./continual_learning_demo)')
    
    args = parser.parse_args()
    
    # Create and run continual learning system
    cl_system = SimpleContinualLearning(
        env_size=args.env_size,
        log_dir=args.log_dir
    )
    
    cl_system.run_continual_learning(
        total_episodes=args.episodes,
        retrain_interval=args.retrain_interval
    )


if __name__ == "__main__":
    main() 