#!/usr/bin/env python3
"""
Evaluation script for trained PixelLife agents.
Loads saved models and runs episodes with visualization.
"""

import numpy as np
import argparse
import os
import time
from stable_baselines3 import PPO
from train import PixelLifeMultiAgentEnv
from env import PixelLifeEnv


def evaluate_agents(main_model_path: str, spice_model_path: str, 
                   num_episodes: int = 5, max_steps: int = 200,
                   grid_size: int = 16, render: bool = True,
                   deterministic: bool = True):
    """
    Evaluate trained agents by running episodes.
    
    Args:
        main_model_path: Path to saved main agent model
        spice_model_path: Path to saved spice agent model  
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        grid_size: Size of the grid (H=W)
        render: Whether to print visual output
        deterministic: Whether to use deterministic actions
    """
    
    print(f"=== PixelLife Agent Evaluation ===")
    print(f"Episodes: {num_episodes}, Max steps: {max_steps}, Grid: {grid_size}x{grid_size}")
    
    # Load models
    try:
        main_agent = PPO.load(main_model_path)
        spice_agent = PPO.load(spice_model_path)
        print(f"✅ Loaded models: {main_model_path}, {spice_model_path}")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Create environment configuration
    env_config = {
        'H': grid_size,
        'W': grid_size, 
        'max_steps': max_steps
    }
    
    # Track statistics
    episode_stats = {
        'main_rewards': [],
        'spice_rewards': [],
        'episode_lengths': [],
        'final_populations': [],
        'max_populations': []
    }
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # Create fresh environments for both agents
        main_env = PixelLifeMultiAgentEnv(env_config, 'main')
        spice_env = PixelLifeMultiAgentEnv(env_config, 'spice')
        
        # Reset environments (they share the same underlying PixelLifeEnv)
        main_obs, _ = main_env.reset(seed=episode * 42)
        spice_obs = main_env.current_obs['spice']  # Get spice observation from same env
        
        # Episode tracking
        episode_main_reward = 0
        episode_spice_reward = 0
        max_population = 0
        step = 0
        
        if render:
            print("Initial state:")
            main_env.render()
        
        # Run episode
        for step in range(max_steps):
            # Get actions from both agents
            main_action, _ = main_agent.predict(main_obs, deterministic=deterministic)
            spice_action, _ = spice_agent.predict(spice_obs, deterministic=deterministic)
            
            # Convert actions and step environment
            pixel_actions = main_env._convert_main_action(main_action)
            spice_action_int = int(spice_action)
            
            # Execute step in the underlying environment
            obs_dict, rewards_dict, terminated, truncated, info = main_env.pixel_env.step(
                spice_action_int, pixel_actions
            )
            
            # Update observations and rewards
            main_obs = obs_dict['main']
            spice_obs = obs_dict['spice']
            episode_main_reward += rewards_dict['main']
            episode_spice_reward += rewards_dict['spice']
            
            # Track population
            current_population = info['total_pixels']
            max_population = max(max_population, current_population)
            
            # Render periodically
            if render and (step % 20 == 0 or terminated or truncated):
                print(f"\nStep {step}: {info['total_organisms']} organisms, {current_population} pixels")
                main_env.render()
                time.sleep(0.1)  # Brief pause for readability
            
            if terminated or truncated:
                break
        
        # Episode summary
        final_population = info['total_pixels']
        print(f"\n📊 Episode {episode + 1} Summary:")
        print(f"   Length: {step + 1} steps")
        print(f"   Main reward: {episode_main_reward:.1f}")
        print(f"   Spice reward: {episode_spice_reward:.1f}")
        print(f"   Final population: {final_population}")
        print(f"   Max population: {max_population}")
        print(f"   Termination: {'Environment limit' if terminated else 'Episode complete'}")
        
        # Store statistics
        episode_stats['main_rewards'].append(episode_main_reward)
        episode_stats['spice_rewards'].append(episode_spice_reward)
        episode_stats['episode_lengths'].append(step + 1)
        episode_stats['final_populations'].append(final_population)
        episode_stats['max_populations'].append(max_population)
    
    # Overall statistics
    print(f"\n🎯 Overall Results ({num_episodes} episodes):")
    print(f"   Average main reward: {np.mean(episode_stats['main_rewards']):.2f} ± {np.std(episode_stats['main_rewards']):.2f}")
    print(f"   Average spice reward: {np.mean(episode_stats['spice_rewards']):.2f} ± {np.std(episode_stats['spice_rewards']):.2f}")
    print(f"   Average episode length: {np.mean(episode_stats['episode_lengths']):.1f} ± {np.std(episode_stats['episode_lengths']):.1f}")
    print(f"   Average final population: {np.mean(episode_stats['final_populations']):.1f} ± {np.std(episode_stats['final_populations']):.1f}")
    print(f"   Average max population: {np.mean(episode_stats['max_populations']):.1f} ± {np.std(episode_stats['max_populations']):.1f}")
    
    return episode_stats


def interactive_demo(model_dir: str = "./models/", grid_size: int = 12):
    """
    Interactive demonstration where user can watch single episodes.
    """
    print("🎮 Interactive PixelLife Demo")
    print("Press Enter to step through the simulation...")
    
    # Try to load latest models
    main_path = os.path.join(model_dir, "main_agent_final")
    spice_path = os.path.join(model_dir, "spice_agent_final")
    
    if not (os.path.exists(main_path + ".zip") and os.path.exists(spice_path + ".zip")):
        print("❌ No trained models found. Run training first:")
        print("   python train.py --total-timesteps 10000 --grid-size 8")
        return
    
    # Load agents
    main_agent = PPO.load(main_path)
    spice_agent = PPO.load(spice_path)
    
    # Create environment
    env = PixelLifeEnv(H=grid_size, W=grid_size, max_steps=300)
    obs_dict, _ = env.reset()
    
    step = 0
    print("\nInitial state:")
    env.render()
    
    while step < 300:
        input("Press Enter for next step...")
        
        # Get actions (simplified for demo)
        main_obs = obs_dict['main']
        spice_obs = obs_dict['spice']
        
        main_action, _ = main_agent.predict(main_obs, deterministic=True)
        spice_action, _ = spice_agent.predict(spice_obs, deterministic=True)
        
        # Convert main action to pixel actions (simplified)
        live_pixels = list(env.pixel_to_org.keys())
        pixel_actions = {}
        if live_pixels:
            # Simple policy: make some pixels split
            for i, pixel in enumerate(live_pixels[:min(3, len(live_pixels))]):
                pixel_actions[pixel] = i % 4  # Cycle through actions
        
        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(int(spice_action), pixel_actions)
        
        step += 1
        print(f"\nStep {step}:")
        print(f"Rewards - Main: {rewards['main']}, Spice: {rewards['spice']}")
        print(f"Population: {info['total_pixels']} pixels, {info['total_organisms']} organisms")
        env.render()
        
        if terminated or truncated:
            print(f"\nEpisode ended after {step} steps!")
            break


def benchmark_models(model_dir: str = "./models/"):
    """
    Benchmark different saved model checkpoints.
    """
    print("📈 Benchmarking Model Checkpoints")
    
    # Find all model checkpoints
    checkpoints = []
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.startswith("main_agent_") and filename.endswith(".zip"):
                checkpoint = filename.replace("main_agent_", "").replace(".zip", "")
                if os.path.exists(os.path.join(model_dir, f"spice_agent_{checkpoint}.zip")):
                    checkpoints.append(checkpoint)
    
    if not checkpoints:
        print("❌ No model checkpoints found.")
        return
    
    print(f"Found checkpoints: {checkpoints}")
    
    results = {}
    for checkpoint in checkpoints[:3]:  # Limit to 3 most recent
        print(f"\n--- Evaluating checkpoint: {checkpoint} ---")
        
        main_path = os.path.join(model_dir, f"main_agent_{checkpoint}")
        spice_path = os.path.join(model_dir, f"spice_agent_{checkpoint}")
        
        stats = evaluate_agents(
            main_path, spice_path,
            num_episodes=3, max_steps=150, grid_size=8,  # Use same size as training
            render=False, deterministic=True
        )
        
        results[checkpoint] = {
            'main_reward': np.mean(stats['main_rewards']),
            'episode_length': np.mean(stats['episode_lengths']),
            'final_population': np.mean(stats['final_populations'])
        }
    
    # Compare results
    print(f"\n🏆 Checkpoint Comparison:")
    print(f"{'Checkpoint':<15} {'Main Reward':<12} {'Avg Length':<12} {'Final Pop':<10}")
    print("-" * 50)
    for checkpoint, metrics in results.items():
        print(f"{checkpoint:<15} {metrics['main_reward']:<12.1f} {metrics['episode_length']:<12.1f} {metrics['final_population']:<10.1f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate PixelLife trained agents')
    
    subparsers = parser.add_subparsers(dest='mode', help='Evaluation mode')
    
    # Standard evaluation
    eval_parser = subparsers.add_parser('eval', help='Run standard evaluation')
    eval_parser.add_argument('--main-model', type=str, default='./models/main_agent_final', 
                           help='Path to main agent model')
    eval_parser.add_argument('--spice-model', type=str, default='./models/spice_agent_final',
                           help='Path to spice agent model')  
    eval_parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    eval_parser.add_argument('--max-steps', type=int, default=200, help='Max steps per episode')
    eval_parser.add_argument('--grid-size', type=int, default=16, help='Grid size')
    eval_parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    eval_parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions')
    
    # Interactive demo
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    demo_parser.add_argument('--grid-size', type=int, default=12, help='Grid size')
    demo_parser.add_argument('--model-dir', type=str, default='./models/', help='Model directory')
    
    # Benchmark
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark multiple checkpoints')
    bench_parser.add_argument('--model-dir', type=str, default='./models/', help='Model directory')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    if args.mode == 'eval':
        evaluate_agents(
            args.main_model, args.spice_model,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            grid_size=args.grid_size,
            render=not args.no_render,
            deterministic=not args.stochastic
        )
    elif args.mode == 'demo':
        interactive_demo(args.model_dir, args.grid_size)
    elif args.mode == 'benchmark':
        benchmark_models(args.model_dir)
    else:
        print("Please specify a mode: eval, demo, or benchmark")
        print("Examples:")
        print("  python eval.py eval --episodes 10")
        print("  python eval.py demo")
        print("  python eval.py benchmark")


if __name__ == "__main__":
    main()