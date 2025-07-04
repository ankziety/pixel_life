#!/usr/bin/env python3
"""
Pixel Life: Unified Command Line Interface
A comprehensive tool for running all Pixel Life environment modes.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import PixelLifeEnv
from train import train_pixel_life, PixelLifeWrapper, make_env
from per_pixel_ai import PerPixelAISystem, PerPixelObservationWrapper
from continual_learning import ContinualLearningSystem
from basic_renderer import PixelLifeRenderer
from enhanced_renderer import EnhancedPixelLifeRenderer
from stable_baselines3 import PPO, DQN


def run_basic_demo(args):
    """Run basic environment demonstration."""
    print("Running Basic Environment Demo")
    print("=" * 40)
    
    env = PixelLifeEnv(H=args.size, W=args.size)
    obs = env.reset()
    
    total_steps = args.steps
    for step in range(total_steps):
        # Random spice action
        spice_action = env.spice_action_space.sample()
        
        # Random pixel actions
        pixel_actions = {}
        for coord in env.live_pixels:
            action_type = np.random.randint(0, 4)
            direction = np.random.randint(0, 4)
            pixel_actions[coord] = (action_type, direction)
        
        obs, rewards, done, info = env.step(spice_action, pixel_actions)
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Pixels={len(env.live_pixels):2d}, "
                  f"Main reward={rewards[0]:6.2f}, Spice reward={rewards[1]:6.2f}")
        
        if args.render and step % 10 == 0:
            env.render()
            plt.pause(0.1)
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Final state: {len(env.live_pixels)} pixels alive")
    if args.render:
        plt.show()


def run_ai_demo(args):
    """Run AI agent demonstration."""
    print("Running AI Agent Demo")
    print("=" * 40)
    
    env = PixelLifeEnv(H=args.size, W=args.size)
    renderer = PixelLifeRenderer(env)
    
    # Create simple AI models
    main_env = PixelLifeWrapper(env, 'main')
    spice_env = PixelLifeWrapper(env, 'spice')
    
    main_model = PPO("MlpPolicy", main_env, verbose=0)
    spice_model = PPO("MlpPolicy", spice_env, verbose=0)
    
    # Quick training
    print("Training AI agents...")
    main_model.learn(total_timesteps=10000)
    spice_model.learn(total_timesteps=10000)
    
    # Run demo
    obs = env.reset()
    for step in range(args.steps):
        obs_main, obs_spice = obs
        
        # Get AI actions - use wrapped environment observations
        spice_action, _ = spice_model.predict(spice_env._flatten_observation(obs_spice), deterministic=True)
        main_action, _ = main_model.predict(main_env._flatten_observation(obs_main), deterministic=True)
        
        # Convert main action to pixel actions
        pixel_actions = {}
        for coord in env.live_pixels:
            pixel_actions[coord] = (int(main_action % 4), int(main_action // 4))
        
        obs, rewards, done, info = env.step(int(spice_action), pixel_actions)
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Pixels={len(env.live_pixels):2d}, "
                  f"Main reward={rewards[0]:6.2f}, Spice reward={rewards[1]:6.2f}")
        
        if args.render:
            renderer.render()
            time.sleep(0.1)
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Final state: {len(env.live_pixels)} pixels alive")


def run_per_pixel_demo(args):
    """Run per-pixel AI system demonstration."""
    print("Running Per-Pixel AI Demo")
    print("=" * 40)
    
    ai_system = PerPixelAISystem(env_size=args.size, log_dir="./logs/per_pixel_demo")
    
    if args.train:
        print("Training per-pixel AI system...")
        ai_system.run_continual_learning(total_episodes=args.generations)
    
    print("Running per-pixel AI demo...")
    episode_info = ai_system.run_episode(max_steps=args.steps)
    print(f"Episode completed: {episode_info}")


def run_continual_learning_demo(args):
    """Run continual learning demonstration."""
    print("Running Continual Learning Demo")
    print("=" * 40)
    
    cl_system = ContinualLearningSystem(
        env_kwargs={'H': args.size, 'W': args.size},
        log_dir="./logs/continual_learning_demo"
    )
    
    if args.train:
        print("Training continual learning system...")
        cl_system.run_continual_learning(total_episodes=args.episodes)
    
    print("Running continual learning demo...")
    episode_info = cl_system.run_episode(max_steps=args.steps, render=args.render)
    print(f"Episode completed: {episode_info}")


def run_pygame_demo(args):
    """Run Pygame-based visualization demo."""
    print("Running Pygame Demo")
    print("=" * 40)
    
    try:
        import pygame
    except ImportError:
        print("Pygame not installed. Install with: pip install pygame")
        return
    
    env = PixelLifeEnv(H=args.size, W=args.size)
    renderer = PixelLifeRenderer(env)
    
    obs = env.reset()
    clock = pygame.time.Clock()
    
    for step in range(args.steps):
        # Random actions
        spice_action = env.spice_action_space.sample()
        pixel_actions = {}
        for coord in env.live_pixels:
            action_type = np.random.randint(0, 4)
            direction = np.random.randint(0, 4)
            pixel_actions[coord] = (action_type, direction)
        
        obs, rewards, done, info = env.step(spice_action, pixel_actions)
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Pixels={len(env.live_pixels):2d}")
        
        renderer.render()
        clock.tick(10)  # 10 FPS
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Final state: {len(env.live_pixels)} pixels alive")


def run_enhanced_demo(args):
    """Run Enhanced Pygame-based visualization demo with zoom and resizable window."""
    print("Running Enhanced Pygame Demo")
    print("=" * 40)
    
    try:
        import pygame
    except ImportError:
        print("Pygame not installed. Install with: pip install pygame")
        return
    
    env = PixelLifeEnv(H=args.size, W=args.size)
    
    # Reset environment first to initialize the grid
    obs = env.reset()
    
    renderer = EnhancedPixelLifeRenderer(
        env=env,
        width=1400,
        height=900,
        initial_zoom=None  # Auto-fit to window
    )
    
    print(f"Enhanced renderer started with {args.initial_zoom}x zoom")
    print("Controls: Mouse wheel zoom, middle drag pan, F fullscreen, R reset view")
    
    # Run the enhanced renderer
    renderer.run_with_env()


def run_training(args):
    """Run full training session."""
    print("Running Full Training Session")
    print("=" * 40)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/training_run_{timestamp}"
    
    print(f"Training log directory: {log_dir}")
    print(f"Environment size: {args.size}x{args.size}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Device: {args.device}")
    
    # Training parameters
    hyperparams = {
        'total_timesteps': args.timesteps,
        'n_envs': args.n_envs,
        'learning_rate': args.learning_rate,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'device': args.device,
        'log_dir': log_dir,
        'no_tensorboard': args.no_tensorboard,
    }
    
    # Run training
    main_model, spice_model, run_dir = train_pixel_life(**hyperparams)
    
    print(f"\nTraining complete!")
    print(f"Models saved to: {run_dir}")
    print(f"View logs with: tensorboard --logdir {log_dir}")


def run_evaluation(args):
    """Run model evaluation."""
    print("Running Model Evaluation")
    print("=" * 40)
    
    if not args.model_path:
        print("Error: --model-path is required for evaluation")
        return
    
    # Load model
    try:
        model = PPO.load(args.model_path)
        print(f"Loaded model from: {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create environment
    env = PixelLifeEnv(H=args.size, W=args.size)
    wrapped_env = PixelLifeWrapper(env, 'main')
    
    # Run evaluation
    total_rewards = []
    for episode in range(args.episodes):
        obs = wrapped_env.reset()
        episode_reward = 0
        
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = wrapped_env.step(action)
            episode_reward += reward
            
            if args.render and step % 10 == 0:
                env.render()
                plt.pause(0.1)
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nEvaluation Results:")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {min(total_rewards):.2f}")
    print(f"Max reward: {max(total_rewards):.2f}")
    
    if args.render:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog='pixel_life',
        description='Pixel Life: A 2D Artificial Life Environment',
        epilog='Run "pixel_life <mode> --help" for mode-specific options'
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Available modes')
    
    # Basic demo parser
    basic_parser = subparsers.add_parser('basic', help='Basic environment demonstration')
    basic_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    basic_parser.add_argument('--steps', type=int, default=200, help='Number of steps (default: 200)')
    basic_parser.add_argument('--render', action='store_true', help='Enable rendering')
    basic_parser.set_defaults(func=run_basic_demo)
    
    # AI demo parser
    ai_parser = subparsers.add_parser('ai', help='AI agent demonstration')
    ai_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    ai_parser.add_argument('--steps', type=int, default=200, help='Number of steps (default: 200)')
    ai_parser.add_argument('--render', action='store_true', help='Enable rendering')
    ai_parser.set_defaults(func=run_ai_demo)
    
    # Per-pixel demo parser
    per_pixel_parser = subparsers.add_parser('per-pixel', help='Per-pixel AI system demonstration')
    per_pixel_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    per_pixel_parser.add_argument('--steps', type=int, default=200, help='Number of steps (default: 200)')
    per_pixel_parser.add_argument('--render', action='store_true', help='Enable rendering')
    per_pixel_parser.add_argument('--train', action='store_true', help='Train the system first')
    per_pixel_parser.add_argument('--generations', type=int, default=5, help='Number of generations (default: 5)')
    per_pixel_parser.add_argument('--steps-per-gen', type=int, default=1000, help='Steps per generation (default: 1000)')
    per_pixel_parser.set_defaults(func=run_per_pixel_demo)
    
    # Continual learning demo parser
    cl_parser = subparsers.add_parser('continual', help='Continual learning demonstration')
    cl_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    cl_parser.add_argument('--steps', type=int, default=200, help='Number of steps (default: 200)')
    cl_parser.add_argument('--render', action='store_true', help='Enable rendering')
    cl_parser.add_argument('--train', action='store_true', help='Train the system first')
    cl_parser.add_argument('--episodes', type=int, default=10, help='Number of episodes (default: 10)')
    cl_parser.add_argument('--steps-per-episode', type=int, default=500, help='Steps per episode (default: 500)')
    cl_parser.set_defaults(func=run_continual_learning_demo)
    
    # Pygame demo parser
    pygame_parser = subparsers.add_parser('pygame', help='Pygame-based visualization')
    pygame_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    pygame_parser.add_argument('--steps', type=int, default=200, help='Number of steps (default: 200)')
    pygame_parser.set_defaults(func=run_pygame_demo)
    
    # Enhanced pygame demo parser
    enhanced_parser = subparsers.add_parser('enhanced', help='Enhanced Pygame visualization with zoom and resizable window')
    enhanced_parser.add_argument('--size', type=int, default=100, help='Environment size (default: 100)')
    enhanced_parser.add_argument('--initial-zoom', type=float, default=0.01, help='Initial zoom level (default: 0.01 = 100x smaller pixels)')
    enhanced_parser.set_defaults(func=run_enhanced_demo)
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Full training session')
    train_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    train_parser.add_argument('--timesteps', type=int, default=1000000, help='Total timesteps (default: 1000000)')
    train_parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments (default: 4)')
    train_parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    train_parser.add_argument('--n-steps', type=int, default=2048, help='Steps before update (default: 2048)')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    train_parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs (default: 10)')
    train_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    train_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device (default: cpu)')
    train_parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard logging')
    train_parser.set_defaults(func=run_training)
    
    # Evaluation parser
    eval_parser = subparsers.add_parser('evaluate', help='Model evaluation')
    eval_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    eval_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    eval_parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes (default: 10)')
    eval_parser.add_argument('--steps', type=int, default=500, help='Steps per episode (default: 500)')
    eval_parser.add_argument('--render', action='store_true', help='Enable rendering')
    eval_parser.set_defaults(func=run_evaluation)
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # Create logs directory if it doesn't exist
    os.makedirs('./logs', exist_ok=True)
    
    # Run the selected mode
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        if args.mode in ['train', 'per-pixel', 'continual']:
            print("This might be due to missing dependencies or insufficient resources.")
            print("Try running with --size 20 for a smaller environment.")


if __name__ == "__main__":
    main() 