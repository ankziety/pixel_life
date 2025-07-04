"""Demo script for per-pixel AI system with visualization."""

import os
import time
import numpy as np
import argparse
import pygame
from typing import Dict, Tuple

from env import PixelLifeEnv
from basic_renderer import PixelLifeRenderer
from per_pixel_ai import PerPixelAISystem, PerPixelObservationWrapper


def demo_per_pixel_ai(env_size=30, steps=1000, render=True, save_models=True):
    """Demo the per-pixel AI system."""
    print("🎮 Per-Pixel AI Demo")
    print("=" * 50)
    print(f"🌍 Environment: {env_size}x{env_size}")
    print(f"⏱️ Steps: {steps}")
    print(f"🎨 Render: {render}")
    print(f"💾 Save models: {save_models}")
    print()
    
    # Create per-pixel AI system
    ai_system = PerPixelAISystem(env_size=env_size, log_dir="./per_pixel_demo_logs")
    ai_system.initialize_environment()
    
    # Create renderer if requested
    renderer = None
    if render:
        renderer = PixelLifeRenderer(env=ai_system.env)
        print("✅ Renderer initialized")
    
    # Reset environment
    obs = ai_system.env.reset()
    initial_pixels = len(ai_system.env.live_pixels)
    print(f"✅ Environment reset with {initial_pixels} initial pixels")
    
    # Run simulation
    step = 0
    total_reward = 0
    
    print("\n🚀 Starting per-pixel AI simulation...")
    print("Controls: SPACE=pause, ESC=quit, R=random, C=clear")
    
    try:
        while step < steps and not ai_system.env.done:
            # Predict actions for all pixels
            pixel_actions = ai_system.predict_pixel_actions()
            
            # Execute step
            obs, rewards, done, info = ai_system.env.step(0, pixel_actions)  # No spice action
            total_reward += rewards[0]
            
            # Update pixel agents with individual rewards
            for pixel_coord in ai_system.env.live_pixels:
                if pixel_coord in ai_system.pixel_agents:
                    agent = ai_system.pixel_agents[pixel_coord]
                    pixel_obs = ai_system.obs_wrapper.get_pixel_observation(pixel_coord)
                    
                    # Calculate pixel-specific reward
                    pixel_reward = ai_system._calculate_pixel_reward(pixel_coord, step)
                    agent.update(pixel_obs, pixel_actions.get(pixel_coord, (0, 0))[0], 
                               pixel_reward, done)
            
            # Render if requested
            if render and renderer:
                renderer.update_from_env()
                renderer.render()
                
                # Handle events
                renderer.handle_events()
                if not renderer.running:
                    print("\n⏹️ Simulation stopped by user")
                    return
            
            # Print progress
            if step % 50 == 0:
                current_pixels = len(ai_system.env.live_pixels)
                active_agents = len(ai_system.pixel_agents)
                survival_rate = current_pixels / max(initial_pixels, 1)
                
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
    
    # Save models if requested
    if save_models:
        print(f"\n💾 Saving {final_agents} pixel models...")
        ai_system.save_pixel_models()
        print(f"✅ Models saved to: {ai_system.models_dir}")
    
    # Cleanup
    if renderer:
        pygame.quit()
    
    print(f"\n🎉 Per-pixel AI demo complete!")


def demo_with_pretrained_models(env_size=30, steps=1000, render=True):
    """Demo using pre-trained per-pixel models."""
    print("🎮 Per-Pixel AI Demo (Pre-trained Models)")
    print("=" * 50)
    
    # Check if models exist
    models_dir = "./per_pixel_demo_logs/pixel_models"
    if not os.path.exists(models_dir):
        print(f"❌ No pre-trained models found at: {models_dir}")
        print("Run the demo first to train models.")
        return
    
    # Count available models
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    print(f"📁 Found {len(model_files)} pre-trained pixel models")
    
    # Create AI system and load models
    ai_system = PerPixelAISystem(env_size=env_size, log_dir="./per_pixel_demo_logs")
    ai_system.initialize_environment()
    
    # Load existing models
    loaded_models = 0
    for model_file in model_files:
        # Extract pixel coordinates from filename
        parts = model_file.replace('.zip', '').split('_')
        if len(parts) >= 3:
            try:
                y, x = int(parts[1]), int(parts[2])
                pixel_coord = (y, x)
                
                # Create agent and load model
                agent = ai_system.get_or_create_pixel_agent(pixel_coord)
                model_path = os.path.join(models_dir, model_file)
                agent.load(model_path)
                loaded_models += 1
                
            except ValueError:
                continue
    
    print(f"✅ Loaded {loaded_models} pixel models")
    
    # Run demo
    demo_per_pixel_ai(env_size, steps, render, save_models=False)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Per-Pixel AI Demo')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of steps (default: 1000)')
    parser.add_argument('--env-size', type=int, default=30,
                       help='Environment size (default: 30)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable model saving')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pre-trained models')
    
    args = parser.parse_args()
    
    if args.pretrained:
        demo_with_pretrained_models(
            env_size=args.env_size,
            steps=args.steps,
            render=not args.no_render
        )
    else:
        demo_per_pixel_ai(
            env_size=args.env_size,
            steps=args.steps,
            render=not args.no_render,
            save_models=not args.no_save
        )


if __name__ == "__main__":
    main() 