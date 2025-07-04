"""Combined Continual Learning + Renderer Demo for Per-Pixel AI.

This demo shows real-time rendering of the per-pixel AI system while it's learning,
allowing you to watch the AI adapt and improve visually.
"""

import os
import time
import numpy as np
import argparse
import pygame
from typing import Dict, Tuple
from datetime import datetime

from env import PixelLifeEnv
from basic_renderer import PixelLifeRenderer
from per_pixel_ai import PerPixelAISystem, PerPixelObservationWrapper


class ContinualLearningRenderer:
    """Combined continual learning and rendering system."""
    
    def __init__(self, env_size=20, log_dir="./continual_learning_renderer_logs"):
        self.env_size = env_size
        self.log_dir = log_dir
        
        # Create AI system
        self.ai_system = PerPixelAISystem(env_size=env_size, log_dir=log_dir)
        self.ai_system.initialize_environment()
        
        # Create renderer
        self.renderer = PixelLifeRenderer(env=self.ai_system.env)
        
        # Learning parameters
        self.episode = 0
        self.total_episodes = 0
        self.training_interval = 10
        self.last_training = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.survival_rates = []
        self.learning_stats = []
        
        # Display info
        self.font = None
        self.info_surface = None
        
    def setup_display(self):
        """Setup pygame display and fonts."""
        pygame.init()
        
        # Create info surface for stats
        self.info_surface = pygame.Surface((400, 200))
        self.info_surface.fill((0, 0, 0))
        
        # Setup font
        try:
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
        except:
            self.font = pygame.font.SysFont('arial', 24)
            self.small_font = pygame.font.SysFont('arial', 18)
    
    def render_stats(self):
        """Render learning statistics on screen."""
        if not self.font:
            return
            
        # Clear info surface
        self.info_surface.fill((0, 0, 0))
        
        # Episode info
        episode_text = self.font.render(f"Episode: {self.episode}/{self.total_episodes}", True, (255, 255, 255))
        self.info_surface.blit(episode_text, (10, 10))
        
        # Current stats
        current_pixels = len(self.ai_system.env.live_pixels)
        active_agents = len(self.ai_system.pixel_agents)
        current_reward = sum(self.episode_rewards[-10:]) if self.episode_rewards else 0
        
        pixels_text = self.font.render(f"Pixels: {current_pixels}", True, (255, 255, 255))
        agents_text = self.font.render(f"Agents: {active_agents}", True, (255, 255, 255))
        reward_text = self.font.render(f"Reward: {current_reward:.1f}", True, (255, 255, 255))
        
        self.info_surface.blit(pixels_text, (10, 40))
        self.info_surface.blit(agents_text, (10, 70))
        self.info_surface.blit(reward_text, (10, 100))
        
        # Learning info
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-10:])
            avg_survival = np.mean(self.survival_rates[-10:]) if self.survival_rates else 0
            
            avg_reward_text = self.font.render(f"Avg Reward: {avg_reward:.1f}", True, (0, 255, 0))
            avg_survival_text = self.font.render(f"Avg Survival: {avg_survival:.2f}", True, (0, 255, 0))
            
            self.info_surface.blit(avg_reward_text, (10, 130))
            self.info_surface.blit(avg_survival_text, (10, 160))
        
        # Training status
        if self.episode - self.last_training >= self.training_interval:
            training_text = self.font.render("TRAINING...", True, (255, 255, 0))
            self.info_surface.blit(training_text, (200, 10))
        else:
            next_training = self.training_interval - (self.episode - self.last_training)
            next_text = self.small_font.render(f"Next training in: {next_training}", True, (128, 128, 128))
            self.info_surface.blit(next_text, (200, 10))
        
        # Generation info
        gen_text = self.small_font.render(f"Generation: {self.ai_system.generation}", True, (128, 128, 128))
        self.info_surface.blit(gen_text, (200, 40))
        
        # Blit info surface to main screen
        self.renderer.screen.blit(self.info_surface, (10, 10))
    
    def run_episode(self, max_steps=200):
        """Run a single episode with rendering."""
        # Reset environment
        obs = self.ai_system.env.reset()
        initial_pixels = len(self.ai_system.env.live_pixels)
        
        # Run episode
        step = 0
        total_reward = 0
        
        while step < max_steps and not self.ai_system.env.done:
            # Predict actions for all pixels
            pixel_actions = self.ai_system.predict_pixel_actions()
            
            # Execute step
            obs, rewards, done, info = self.ai_system.env.step(0, pixel_actions)
            total_reward += rewards[0]
            
            # Update pixel agents with individual rewards
            for pixel_coord in self.ai_system.env.live_pixels:
                if pixel_coord in self.ai_system.pixel_agents:
                    agent = self.ai_system.pixel_agents[pixel_coord]
                    pixel_obs = self.ai_system.obs_wrapper.get_pixel_observation(pixel_coord)
                    
                    # Calculate pixel-specific reward
                    pixel_reward = self.ai_system._calculate_pixel_reward(pixel_coord, step)
                    agent.update(pixel_obs, pixel_actions.get(pixel_coord, (0, 0))[0], 
                               pixel_reward, done)
            
            # Render
            self.renderer.update_from_env()
            self.renderer.render()
            self.render_stats()
            
            # Handle events
            self.renderer.handle_events()
            if not self.renderer.running:
                return False  # User quit
            
            # Check for pause
            if self.renderer.paused:
                # Wait for unpause
                while self.renderer.paused:
                    self.renderer.handle_events()
                    if not self.renderer.running:
                        return False
                    time.sleep(0.1)
            
            step += 1
            
            if done:
                break
        
        # Episode results
        final_pixels = len(self.ai_system.env.live_pixels)
        survival_rate = final_pixels / max(initial_pixels, 1)
        
        # Track performance
        self.episode_rewards.append(total_reward)
        self.survival_rates.append(survival_rate)
        
        return True  # Continue running
    
    def train_pixel_agents(self):
        """Train all pixel agents with visual feedback."""
        print(f"\n🔄 Training {len(self.ai_system.pixel_agents)} pixel agents...")
        
        # Show training progress on screen
        training_surface = pygame.Surface((400, 100))
        training_surface.fill((0, 0, 0))
        
        training_text = self.font.render("TRAINING PIXEL AGENTS...", True, (255, 255, 0))
        training_surface.blit(training_text, (10, 10))
        
        # Train each agent
        for i, (pixel_coord, agent) in enumerate(self.ai_system.pixel_agents.items()):
            if len(agent.reward_history) > 10:  # Only train if we have data
                # Update training progress
                progress_text = self.small_font.render(f"Training pixel {pixel_coord}... ({i+1}/{len(self.ai_system.pixel_agents)})", True, (255, 255, 255))
                training_surface.fill((0, 0, 0))
                training_surface.blit(training_text, (10, 10))
                training_surface.blit(progress_text, (10, 40))
                
                # Render training progress
                self.renderer.screen.blit(training_surface, (10, 220))
                pygame.display.flip()
                
                # Train the agent
                agent.train(total_timesteps=200)
                
                # Handle events during training
                self.renderer.handle_events()
                if not self.renderer.running:
                    return False
        
        self.ai_system.generation += 1
        print(f"✅ Training complete (generation {self.ai_system.generation})")
        return True
    
    def run_continual_learning(self, total_episodes=100, training_interval=10):
        """Run continual learning with real-time rendering."""
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        
        print("🚀 Starting Continual Learning with Real-time Rendering...")
        print(f"📊 Episodes: {total_episodes}")
        print(f"🔄 Training interval: {training_interval}")
        print(f"🌍 Environment: {self.env_size}x{self.env_size}")
        print("\n🎮 Controls:")
        print("  SPACE - Pause/Unpause")
        print("  ESC - Quit")
        print("  Watch the AI learn in real-time!")
        
        # Setup display
        self.setup_display()
        
        try:
            while self.episode < total_episodes:
                # Run episode
                if not self.run_episode():
                    break  # User quit
                
                self.episode += 1
                
                # Print progress
                if self.episode % 5 == 0:
                    avg_reward = np.mean(self.episode_rewards[-5:])
                    avg_survival = np.mean(self.survival_rates[-5:])
                    print(f"Episode {self.episode}/{total_episodes}")
                    print(f"  Avg Reward: {avg_reward:.2f}")
                    print(f"  Avg Survival: {avg_survival:.2f}")
                    print(f"  Current: {len(self.ai_system.env.live_pixels)} pixels")
                    print(f"  Active Agents: {len(self.ai_system.pixel_agents)}")
                    print(f"  Generation: {self.ai_system.generation}")
                
                # Train pixel agents periodically
                if self.episode - self.last_training >= training_interval:
                    print(f"\n🔄 Training pixel agents at episode {self.episode}")
                    if not self.train_pixel_agents():
                        break  # User quit during training
                    
                    # Save models (with automatic cleanup of old versions)
                    self.ai_system.save_pixel_models()
                    self.last_training = self.episode
                
        except KeyboardInterrupt:
            print("\n⏹️ Continual learning interrupted")
        
        # Final save
        self.ai_system.save_pixel_models()
        
        # Print final stats
        print(f"\n🎉 Continual learning with rendering complete!")
        print(f"📊 Final generation: {self.ai_system.generation}")
        print(f"🤖 Total pixel agents: {len(self.ai_system.pixel_agents)}")
        print(f"📁 Models saved to: {self.ai_system.models_dir}")
        
        if self.episode_rewards:
            final_avg_reward = np.mean(self.episode_rewards[-10:])
            final_avg_survival = np.mean(self.survival_rates[-10:])
            print(f"📈 Final avg reward: {final_avg_reward:.2f}")
            print(f"📈 Final avg survival: {final_avg_survival:.2f}")
        
        # Cleanup
        pygame.quit()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Continual Learning with Real-time Rendering')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Total episodes (default: 50)')
    parser.add_argument('--training-interval', type=int, default=10,
                       help='Episodes between training (default: 10)')
    parser.add_argument('--env-size', type=int, default=20,
                       help='Environment size (default: 20)')
    parser.add_argument('--log-dir', type=str, default='./continual_learning_renderer_logs',
                       help='Log directory')
    
    args = parser.parse_args()
    
    # Create and run continual learning renderer
    cl_renderer = ContinualLearningRenderer(
        env_size=args.env_size,
        log_dir=args.log_dir
    )
    
    cl_renderer.run_continual_learning(
        total_episodes=args.episodes,
        training_interval=args.training_interval
    )


if __name__ == "__main__":
    main() 