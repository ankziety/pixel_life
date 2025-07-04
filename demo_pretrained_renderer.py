"""Demo script for pre-trained per-pixel AI models with rendering."""

import os
import time
import numpy as np
import argparse
import pygame
from typing import Dict, Tuple

from env import PixelLifeEnv
from basic_renderer import PixelLifeRenderer
from per_pixel_ai import PerPixelAISystem, PerPixelObservationWrapper


def demo_pretrained_models(env_size=15, steps=1000, models_dir=None):
    """Demo pre-trained per-pixel AI models with rendering."""
    print("🎮 Pre-trained Per-Pixel AI Demo")
    print("=" * 50)
    print(f"🌍 Environment: {env_size}x{env_size}")
    print(f"⏱️ Steps: {steps}")
    
    # Auto-detect models directory if not specified
    if models_dir is None:
        possible_dirs = [
            "./continual_learning_renderer_logs/pixel_models",
            "./per_pixel_ai_logs/pixel_models", 
            "./per_pixel_demo_logs/pixel_models"
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                models_dir = dir_path
                break
        
        if models_dir is None:
            print("❌ No pre-trained models found!")
            print("Run one of these first to train models:")
            print("  python3 demo_continual_learning_renderer.py")
            print("  python3 per_pixel_ai.py")
            print("  python3 demo_per_pixel.py")
            return
    
    print(f"📁 Loading models from: {models_dir}")
    
    # Count available models
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    print(f"📦 Found {len(model_files)} pre-trained pixel models")
    
    if len(model_files) == 0:
        print("❌ No model files found!")
        return
    
    # Create AI system
    ai_system = PerPixelAISystem(env_size=env_size, log_dir="./pretrained_demo_logs")
    ai_system.initialize_environment()
    
    # Load latest models using the new versioning system
    loaded_models = ai_system.load_latest_models()
    
    if loaded_models == 0:
        print("❌ No models could be loaded!")
        return
    
    # Create renderer
    renderer = PixelLifeRenderer(env=ai_system.env)
    
    # Setup display info
    pygame.init()
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)
    
    # Create info surface
    info_surface = pygame.Surface((400, 150))
    info_surface.fill((0, 0, 0))
    
    # Reset environment
    obs = ai_system.env.reset()
    initial_pixels = len(ai_system.env.live_pixels)
    print(f"✅ Environment reset with {initial_pixels} initial pixels")
    
    # Run simulation
    step = 0
    total_reward = 0
    
    print("\n🚀 Starting pre-trained AI simulation...")
    print("Controls: SPACE=pause, ESC=quit")
    
    try:
        while step < steps and not ai_system.env.done:
            # Predict actions for all pixels
            pixel_actions = ai_system.predict_pixel_actions()
            
            # Execute step
            obs, rewards, done, info = ai_system.env.step(0, pixel_actions)
            total_reward += rewards[0]
            
            # Render
            renderer.update_from_env()
            renderer.render()
            
            # Render stats
            info_surface.fill((0, 0, 0))
            
            # Current stats
            current_pixels = len(ai_system.env.live_pixels)
            active_agents = len(ai_system.pixel_agents)
            survival_rate = current_pixels / max(initial_pixels, 1)
            
            step_text = font.render(f"Step: {step}/{steps}", True, (255, 255, 255))
            pixels_text = font.render(f"Pixels: {current_pixels}", True, (255, 255, 255))
            agents_text = font.render(f"Agents: {active_agents}", True, (255, 255, 255))
            survival_text = font.render(f"Survival: {survival_rate:.2f}", True, (0, 255, 0))
            reward_text = font.render(f"Reward: {total_reward:.1f}", True, (255, 255, 0))
            
            info_surface.blit(step_text, (10, 10))
            info_surface.blit(pixels_text, (10, 40))
            info_surface.blit(agents_text, (10, 70))
            info_surface.blit(survival_text, (10, 100))
            info_surface.blit(reward_text, (10, 130))
            
            # Blit info to screen
            renderer.screen.blit(info_surface, (10, 10))
            
            # Handle events
            renderer.handle_events()
            if not renderer.running:
                print("\n⏹️ Simulation stopped by user")
                break
            
            # Check for pause
            if renderer.paused:
                # Wait for unpause
                while renderer.paused:
                    renderer.handle_events()
                    if not renderer.running:
                        break
                    time.sleep(0.1)
            
            # Print progress
            if step % 100 == 0:
                print(f"Step {step:4d}: {current_pixels:2d} pixels, "
                      f"{active_agents:3d} agents, "
                      f"survival: {survival_rate:.2f}, "
                      f"reward: {total_reward:.1f}")
            
            step += 1
            
            if done:
                print(f"\n✅ Episode ended at step {step}")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ Simulation interrupted")
    
    # Final statistics
    final_pixels = len(ai_system.env.live_pixels)
    final_agents = len(ai_system.pixel_agents)
    survival_rate = final_pixels / max(initial_pixels, 1)
    
    print(f"\n📊 Final Statistics:")
    print(f"  Total steps: {step}")
    print(f"  Final pixels: {final_pixels}")
    print(f"  Active agents: {final_agents}")
    print(f"  Survival rate: {survival_rate:.2f}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Models loaded: {loaded_models}")
    
    # Cleanup
    pygame.quit()
    
    print(f"\n🎉 Pre-trained AI demo complete!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Pre-trained Per-Pixel AI Demo')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of steps (default: 1000)')
    parser.add_argument('--env-size', type=int, default=15,
                       help='Environment size (default: 15)')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='Directory containing pre-trained models')
    
    args = parser.parse_args()
    
    demo_pretrained_models(
        env_size=args.env_size,
        steps=args.steps,
        models_dir=args.models_dir
    )


if __name__ == "__main__":
    main() 