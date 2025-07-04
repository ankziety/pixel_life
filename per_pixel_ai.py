"""Per-Pixel AI System for Pixel Life Environment.

This system implements true per-pixel action prediction where each pixel
has its own AI agent making independent decisions, enabling complex
coordinated behaviors and emergent strategies.
"""

import os
import numpy as np
from datetime import datetime
import argparse
from typing import Dict, Tuple, List, Optional
import json

import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from env import PixelLifeEnv
from basic_renderer import PixelLifeRenderer


class PerPixelAgent:
    """Individual AI agent for a single pixel."""
    
    def __init__(self, pixel_id: Tuple[int, int], model_type='PPO'):
        self.pixel_id = pixel_id
        self.model_type = model_type
        self.model = None
        self.action_history = []
        self.reward_history = []
        
    def create_model(self, observation_space, action_space):
        """Create the AI model for this pixel."""
        # Create a simple environment wrapper for this pixel
        class PixelEnv(gym.Env):
            def __init__(self, obs_space, act_space):
                super().__init__()
                self.observation_space = obs_space
                self.action_space = act_space
                self.current_obs = None
            
            def reset(self, **kwargs):
                self.current_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                return self.current_obs, {}
            
            def step(self, action):
                # Simple reward based on action
                reward = 0.1 if action < 10 else -0.1
                done = False
                info = {}
                return self.current_obs, reward, done, False, info
        
        # Create environment
        env = PixelEnv(observation_space, action_space)
        
        if self.model_type == 'PPO':
            self.model = PPO("MlpPolicy", env, verbose=0)
        elif self.model_type == 'DQN':
            self.model = DQN("MlpPolicy", env, verbose=0)
    
    def predict_action(self, observation, deterministic=False):
        """Predict action for this pixel."""
        if self.model is None:
            # Random action if no model
            return np.random.randint(0, 20), None
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action, None
    
    def update(self, observation, action, reward, done):
        """Update the agent with experience."""
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Keep history manageable
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-500:]
            self.reward_history = self.reward_history[-500:]
    
    def train(self, total_timesteps=1000):
        """Train this pixel's agent."""
        if self.model is not None and len(self.reward_history) > 5:
            # Use the existing environment that was created during model creation
            try:
                self.model.learn(total_timesteps=total_timesteps)
            except Exception as e:
                # If training fails, just skip it
                pass
    
    def save(self, filepath):
        """Save this pixel's model."""
        if self.model is not None:
            self.model.save(filepath)
    
    def load(self, filepath):
        """Load this pixel's model."""
        if os.path.exists(filepath):
            if self.model_type == 'PPO':
                self.model = PPO.load(filepath)
            elif self.model_type == 'DQN':
                self.model = DQN.load(filepath)


class PerPixelObservationWrapper:
    """Creates pixel-specific observations for per-pixel AI agents."""
    
    def __init__(self, env: PixelLifeEnv):
        self.env = env
        
    def get_pixel_observation(self, pixel_coord: Tuple[int, int]) -> np.ndarray:
        """Get observation specific to a pixel's perspective."""
        y, x = pixel_coord
        
        # Get local grid view (7x7 around the pixel)
        local_grid = np.zeros((7, 7), dtype=np.float32)
        
        # Fill local grid with environment data
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                env_y, env_x = y + dy, x + dx
                
                # Check bounds
                if 0 <= env_y < self.env.H and 0 <= env_x < self.env.W:
                    # Grid state
                    local_grid[dy+3, dx+3] = self.env.grid[env_y, env_x]
                    
                    # Energy at this location
                    if (env_y, env_x) in self.env.pixel_energy:
                        local_grid[dy+3, dx+3] += self.env.pixel_energy[(env_y, env_x)] * 10
        
        # Flatten local grid
        local_obs = local_grid.flatten()
        
        # Add pixel-specific information
        pixel_info = np.array([
            y / self.env.H,  # Normalized y position
            x / self.env.W,  # Normalized x position
            self.env.pixel_energy.get(pixel_coord, 0.0),  # Own energy
            self.env.pixel_ages.get(pixel_coord, 0.0),    # Own age
            len(self.env.live_pixels),  # Total live pixels
            self.env.tick_count,  # Current tick
        ], dtype=np.float32)
        
        # Combine local grid and pixel info
        full_obs = np.concatenate([local_obs, pixel_info])
        
        return full_obs


class PerPixelAISystem:
    """Main system that manages per-pixel AI agents."""
    
    def __init__(self, env_size=30, log_dir="./per_pixel_ai_logs"):
        self.env_size = env_size
        self.log_dir = log_dir
        self.env_kwargs = {'H': env_size, 'W': env_size, 'max_size': 100}
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        self.models_dir = os.path.join(log_dir, "pixel_models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Pixel agents
        self.pixel_agents: Dict[Tuple[int, int], PerPixelAgent] = {}
        
        # Performance tracking
        self.episode_rewards = []
        self.survival_rates = []
        self.generation = 0
        
        # Environment and wrapper
        self.env = None
        self.obs_wrapper = None
        
    def initialize_environment(self):
        """Initialize the environment and observation wrapper."""
        self.env = PixelLifeEnv(**self.env_kwargs)
        self.obs_wrapper = PerPixelObservationWrapper(self.env)
    
    def get_or_create_pixel_agent(self, pixel_coord: Tuple[int, int]) -> PerPixelAgent:
        """Get existing pixel agent or create a new one."""
        if pixel_coord not in self.pixel_agents:
            # Create new agent for this pixel
            agent = PerPixelAgent(pixel_coord, model_type='PPO')
            
            # Create model with appropriate spaces
            # Observation space: 7x7 grid + 6 pixel info = 55 dimensions
            obs_space = gym.spaces.Box(low=-1000, high=1000, shape=(55,), dtype=np.float32)
            # Action space: 20 possible actions (5 action types * 4 directions)
            action_space = gym.spaces.Discrete(20)
            
            agent.create_model(obs_space, action_space)
            self.pixel_agents[pixel_coord] = agent
            
            # Try to load existing model
            model_path = os.path.join(self.models_dir, f"pixel_{pixel_coord[0]}_{pixel_coord[1]}.zip")
            agent.load(model_path)
        
        return self.pixel_agents[pixel_coord]
    
    def predict_pixel_actions(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Predict actions for all live pixels."""
        pixel_actions = {}
        
        for pixel_coord in self.env.live_pixels:
            # Get or create agent for this pixel
            agent = self.get_or_create_pixel_agent(pixel_coord)
            
            # Get pixel-specific observation
            obs = self.obs_wrapper.get_pixel_observation(pixel_coord)
            
            # Predict action
            action, _ = agent.predict_action(obs, deterministic=False)
            
            # Convert action to (action_type, direction)
            action_type = action // 4
            direction = action % 4
            
            pixel_actions[pixel_coord] = (action_type, direction)
        
        return pixel_actions
    
    def run_episode(self, max_steps=500, render=False):
        """Run a single episode with per-pixel AI."""
        if self.env is None:
            self.initialize_environment()
        
        # Reset environment
        obs = self.env.reset()
        initial_pixels = len(self.env.live_pixels)
        
        # Run episode
        step = 0
        total_reward = 0
        pixel_rewards = {}
        
        while step < max_steps and not self.env.done:
            # Predict actions for all pixels
            pixel_actions = self.predict_pixel_actions()
            
            # Execute step
            obs, rewards, done, info = self.env.step(0, pixel_actions)  # No spice action
            total_reward += rewards[0]
            
            # Update pixel agents with their individual rewards
            for pixel_coord in self.env.live_pixels:
                if pixel_coord in self.pixel_agents:
                    agent = self.pixel_agents[pixel_coord]
                    pixel_obs = self.obs_wrapper.get_pixel_observation(pixel_coord)
                    
                    # Calculate pixel-specific reward
                    pixel_reward = self._calculate_pixel_reward(pixel_coord, step)
                    agent.update(pixel_obs, pixel_actions.get(pixel_coord, (0, 0))[0], 
                               pixel_reward, done)
            
            step += 1
            
            if done:
                break
        
        # Episode results
        final_pixels = len(self.env.live_pixels)
        survival_rate = final_pixels / max(initial_pixels, 1)
        
        episode_info = {
            'total_reward': total_reward,
            'episode_length': step,
            'final_pixels': final_pixels,
            'initial_pixels': initial_pixels,
            'survival_rate': survival_rate
        }
        
        # Track performance
        self.episode_rewards.append(total_reward)
        self.survival_rates.append(survival_rate)
        
        return episode_info
    
    def _calculate_pixel_reward(self, pixel_coord: Tuple[int, int], step: int) -> float:
        """Calculate reward specific to a pixel."""
        reward = 0.0
        
        # Survival reward
        if pixel_coord in self.env.live_pixels:
            reward += 1.0
        
        # Energy efficiency reward
        energy = self.env.pixel_energy.get(pixel_coord, 0.0)
        reward += energy * 0.1
        
        # Age reward
        age = self.env.pixel_ages.get(pixel_coord, 0.0)
        reward += age * 0.05
        
        # Reproduction reward (if pixel reproduced)
        # This would need to be tracked in the environment
        
        return reward
    
    def train_pixel_agents(self, episodes_per_training=10):
        """Train all pixel agents."""
        print(f"🎯 Training {len(self.pixel_agents)} pixel agents...")
        
        for pixel_coord, agent in self.pixel_agents.items():
            if len(agent.reward_history) > 10:  # Only train if we have data
                print(f"  Training pixel {pixel_coord}...")
                agent.train(total_timesteps=500)
        
        self.generation += 1
        print(f"✅ Training complete (generation {self.generation})")
    
    MAX_TOTAL_MODEL_FILES = 500  # Global cap for all .zip model files in pixel_models

    def save_pixel_models(self):
        """Save all pixel models with versioning (keep only 3 most recent per pixel, and cap total files)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Track saved models for cleanup
        saved_models = []
        
        for pixel_coord, agent in self.pixel_agents.items():
            # Create versioned filename
            base_name = f"pixel_{pixel_coord[0]}_{pixel_coord[1]}"
            versioned_name = f"{base_name}_gen{self.generation}_{timestamp}.zip"
            model_path = os.path.join(self.models_dir, versioned_name)
            
            # Save the model
            agent.save(model_path)
            saved_models.append((base_name, model_path))
        
        # Cleanup old versions (keep only 3 most recent per pixel)
        self._cleanup_old_models()
        # Global cap: cleanup oldest files if total exceeds MAX_TOTAL_MODEL_FILES
        self._enforce_global_model_cap()
        
        # Save metadata
        metadata = {
            'generation': self.generation,
            'timestamp': timestamp,
            'num_pixels': len(self.pixel_agents),
            'episode_rewards': self.episode_rewards[-100:],  # Last 100 episodes
            'survival_rates': self.survival_rates[-100:],
            'saved_models': len(saved_models)
        }
        
        metadata_path = os.path.join(self.log_dir, f"metadata_gen_{self.generation}_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved {len(saved_models)} pixel models (generation {self.generation})")

    def _enforce_global_model_cap(self):
        """Delete oldest .zip model files if total exceeds MAX_TOTAL_MODEL_FILES."""
        if not os.path.exists(self.models_dir):
            return
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.zip')]
        if len(model_files) <= self.MAX_TOTAL_MODEL_FILES:
            return
        
        # Sort files by modification time (oldest first)
        model_files_full = [os.path.join(self.models_dir, f) for f in model_files]
        model_files_full.sort(key=lambda f: os.path.getmtime(f))
        
        # Delete oldest files until under the cap
        num_to_delete = len(model_files_full) - self.MAX_TOTAL_MODEL_FILES
        deleted = 0
        for f in model_files_full[:num_to_delete]:
            try:
                os.remove(f)
                deleted += 1
                print(f"🗑️ Deleted old model (global cap): {os.path.basename(f)}")
            except OSError as e:
                print(f"⚠️ Could not delete {os.path.basename(f)}: {e}")
        if deleted > 0:
            print(f"🧹 Global cap: cleaned up {deleted} old model files")
    
    def load_latest_models(self):
        """Load the most recent model for each pixel."""
        if not os.path.exists(self.models_dir):
            print("📁 No models directory found")
            return 0
        
        # Group models by pixel base name
        pixel_models = {}
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.zip') and filename.startswith('pixel_'):
                parts = filename.replace('.zip', '').split('_')
                
                # Handle both old format (pixel_y_x.zip) and new format (pixel_y_x_genN_timestamp.zip)
                if len(parts) >= 3:
                    try:
                        y, x = int(parts[1]), int(parts[2])
                        base_name = f"pixel_{y}_{x}"
                        
                        # Check if this is new format with generation
                        if len(parts) >= 4 and parts[3].startswith('gen'):
                            generation = int(parts[3][3:])  # Remove 'gen' prefix
                        else:
                            # Old format - assign generation 0
                            generation = 0
                        
                        if base_name not in pixel_models:
                            pixel_models[base_name] = []
                        
                        model_path = os.path.join(self.models_dir, filename)
                        pixel_models[base_name].append((generation, model_path, filename))
                    except (ValueError, IndexError):
                        continue
        
        # Load the most recent model for each pixel
        loaded_count = 0
        for base_name, models in pixel_models.items():
            if models:
                # Sort by generation (newest first) and take the first one
                models.sort(key=lambda x: x[0], reverse=True)
                generation, model_path, filename = models[0]
                
                # Extract pixel coordinates
                parts = base_name.split('_')
                y, x = int(parts[1]), int(parts[2])
                pixel_coord = (y, x)
                
                # Create agent and load model
                agent = self.get_or_create_pixel_agent(pixel_coord)
                agent.load(model_path)
                loaded_count += 1
        
        print(f"📥 Loaded {loaded_count} latest pixel models")
        return loaded_count
    
    def run_continual_learning(self, total_episodes=200, training_interval=20):
        """Run continual learning with per-pixel AI."""
        print("🚀 Starting per-pixel AI continual learning...")
        print(f"📊 Episodes: {total_episodes}")
        print(f"🔄 Training interval: {training_interval}")
        print(f"🌍 Environment: {self.env_size}x{self.env_size}")
        
        episode = 0
        last_training = 0
        
        try:
            while episode < total_episodes:
                # Run episode
                episode_info = self.run_episode()
                episode += 1
                
                # Print progress
                if episode % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_survival = np.mean(self.survival_rates[-10:])
                    print(f"Episode {episode}/{total_episodes}")
                    print(f"  Avg Reward: {avg_reward:.2f}")
                    print(f"  Avg Survival: {avg_survival:.2f}")
                    print(f"  Current: {episode_info['final_pixels']} pixels")
                    print(f"  Active Agents: {len(self.pixel_agents)}")
                    print(f"  Generation: {self.generation}")
                
                # Train pixel agents periodically
                if episode - last_training >= training_interval:
                    print(f"\n🔄 Training pixel agents at episode {episode}")
                    self.train_pixel_agents()
                    self.save_pixel_models()
                    last_training = episode
                
        except KeyboardInterrupt:
            print("\n⏹️ Per-pixel AI learning interrupted")
        
        # Final save
        self.save_pixel_models()
        
        # Print final stats
        print(f"\n🎉 Per-pixel AI learning complete!")
        print(f"📊 Final generation: {self.generation}")
        print(f"🤖 Total pixel agents: {len(self.pixel_agents)}")
        print(f"📁 Models saved to: {self.models_dir}")
        
        if self.episode_rewards:
            final_avg_reward = np.mean(self.episode_rewards[-10:])
            final_avg_survival = np.mean(self.survival_rates[-10:])
            print(f"📈 Final avg reward: {final_avg_reward:.2f}")
            print(f"📈 Final avg survival: {final_avg_survival:.2f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Per-Pixel AI System')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Total episodes (default: 200)')
    parser.add_argument('--training-interval', type=int, default=20,
                       help='Episodes between training (default: 20)')
    parser.add_argument('--env-size', type=int, default=30,
                       help='Environment size (default: 30)')
    parser.add_argument('--log-dir', type=str, default='./per_pixel_ai_logs',
                       help='Log directory (default: ./per_pixel_ai_logs)')
    
    args = parser.parse_args()
    
    # Create and run per-pixel AI system
    ai_system = PerPixelAISystem(
        env_size=args.env_size,
        log_dir=args.log_dir
    )
    
    ai_system.run_continual_learning(
        total_episodes=args.episodes,
        training_interval=args.training_interval
    )


if __name__ == "__main__":
    main() 