"""Demo script using trained AI models for the Pixel Life environment."""

import numpy as np
from env import PixelLifeEnv
from basic_renderer import PixelLifeRenderer
import time
import argparse
import os
from stable_baselines3 import PPO, DQN
from train import PixelLifeWrapper


def load_trained_models(model_dir="pixel_life_logs/run_20250704_033948"):
    """Load the trained main and spice agents."""
    main_model_path = os.path.join(model_dir, "main_agent", "final_model.zip")
    spice_model_path = os.path.join(model_dir, "spice_agent", "final_model.zip")
    
    if not os.path.exists(main_model_path):
        print(f"❌ Main agent model not found at: {main_model_path}")
        return None, None
    
    if not os.path.exists(spice_model_path):
        print(f"❌ Spice agent model not found at: {spice_model_path}")
        return None, None
    
    print("🤖 Loading trained AI models...")
    
    # Load main agent (PPO)
    main_model = PPO.load(main_model_path)
    print(f"✅ Loaded main agent: {main_model_path}")
    
    # Load spice agent (DQN)
    spice_model = DQN.load(spice_model_path)
    print(f"✅ Loaded spice agent: {spice_model_path}")
    
    return main_model, spice_model


def main():
    """Run the demo with trained AI models."""
    parser = argparse.ArgumentParser(description='Pixel Life Demo with Trained AI')
    parser.add_argument('--steps', type=int, default=5000, 
                       help='Maximum number of steps (default: 5000)')
    parser.add_argument('--run-forever', action='store_true',
                       help='Run until environment naturally ends (ignores --steps)')
    parser.add_argument('--model-dir', type=str, 
                       default="pixel_life_logs/run_20250704_033948",
                       help='Directory containing trained models')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic actions (no exploration)')
    args = parser.parse_args()
    
    print("=" * 50)
    print("PIXEL LIFE DEMONSTRATION (TRAINED AI)")
    print("=" * 50)
    
    # Load trained models
    main_model, spice_model = load_trained_models(args.model_dir)
    if main_model is None or spice_model is None:
        print("❌ Failed to load models. Run training first:")
        print("   python train.py --total-timesteps 100000")
        return
    
    # Create environment (must match training parameters)
    env = PixelLifeEnv(H=30, W=30, max_size=100)
    print(f"\n✓ Created environment: {env.H}x{env.W}")
    
    # Create wrappers for the models
    main_wrapper = PixelLifeWrapper(env, agent_type='main')
    spice_wrapper = PixelLifeWrapper(env, agent_type='spice')
    
    # Connect the models to each other
    main_wrapper.other_model = spice_model  # type: ignore
    spice_wrapper.other_model = main_model  # type: ignore
    
    # Reset
    obs = env.reset()
    print(f"✓ Initial pixels: {len(env.live_pixels)} pixels")
    
    # Create pygame renderer
    renderer = PixelLifeRenderer(
        env=env,
        width=800, 
        height=600, 
        grid_size=15
    )
    
    # Run demonstration
    print("\nRunning AI demonstration...")
    print("- Main agent: Trained AI trying to keep pixels alive")
    print("- Spice agent: Trained AI expanding universe and tweaking rules")
    print(f"- Max steps: {'∞ (run forever)' if args.run_forever else args.steps}")
    print(f"- Mode: {'Deterministic' if args.deterministic else 'Exploratory'}")
    print("\nControls:")
    print("- SPACE: Pause/Unpause")
    print("- ESC/Q: Quit")
    print("- Click: Toggle pixels (for testing)")
    print("\nPress Ctrl+C to stop\n")
    
    step = 0
    total_reward_main = 0
    total_reward_spice = 0
    
    try:
        # Determine the loop condition
        if args.run_forever:
            loop_condition = lambda: not env.done
        else:
            loop_condition = lambda: not env.done and step < args.steps
        
        while loop_condition():
            # Get AI actions
            main_obs = main_wrapper._flatten_observation(env._get_main_observation())
            spice_obs = spice_wrapper._flatten_observation(env._get_spice_observation())
            
            # Predict actions
            main_action, _ = main_model.predict(main_obs, deterministic=args.deterministic)
            spice_action, _ = spice_model.predict(spice_obs, deterministic=args.deterministic)
            
            # Convert main action to pixel actions
            pixel_actions = main_wrapper._convert_main_action(main_action)
            
            # Execute step
            obs, rewards, done, info = env.step(spice_action, pixel_actions)
            total_reward_main += rewards[0]
            total_reward_spice += rewards[1]
            step += 1
            
            # Update renderer from environment
            renderer.update_from_env()
            
            # Print periodic updates
            if step % 50 == 0:
                avg_reward_main = total_reward_main / step
                avg_reward_spice = total_reward_spice / step
                print(f"Step {step}: {info['live_pixels']} pixels, "
                      f"avg energy: {info['avg_energy']:.1f}, "
                      f"grid {info['grid_size']}, "
                      f"main reward: {avg_reward_main:.1f}, "
                      f"spice reward: {avg_reward_spice:.1f}")
            
            if done:
                print(f"\nEpisode ended at step {step}!")
                print(f"Final state: {info['live_pixels']} pixels")
                break
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    # Show final state
    print("\nFinal statistics:")
    print(f"- Total steps: {step}")
    print(f"- Final grid size: {env.H}x{env.W}")
    print(f"- Living pixels: {len(env.live_pixels)}")
    print(f"- Dead pixels: {len(env.dead_cells)}")
    print(f"- Average energy: {np.mean(list(env.pixel_energy.values())) if env.pixel_energy else 0:.2f}")
    print(f"- Average age: {np.mean(list(env.pixel_ages.values())) if env.pixel_ages else 0:.1f}")
    print(f"- Total main agent reward: {total_reward_main:.1f}")
    print(f"- Total spice agent reward: {total_reward_spice:.1f}")
    print(f"- Average main agent reward: {total_reward_main/max(step,1):.1f}")
    print(f"- Average spice agent reward: {total_reward_spice/max(step,1):.1f}")
    
    # Run the pygame renderer
    print("\nStarting pygame renderer... Close window or press ESC to exit.")
    renderer.run_with_env()


if __name__ == "__main__":
    main() 