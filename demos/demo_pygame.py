"""Demo script using pygame renderer with the main Pixel Life environment."""

import numpy as np
from env import PixelLifeEnv
from basic_renderer import PixelLifeRenderer
import time
import argparse


def main():
    """Run the demo with pygame renderer."""
    parser = argparse.ArgumentParser(description='Pixel Life Demo with Pygame Renderer')
    parser.add_argument('--steps', type=int, default=5000, 
                       help='Maximum number of steps (default: 5000)')
    parser.add_argument('--run-forever', action='store_true',
                       help='Run until environment naturally ends (ignores --steps)')
    args = parser.parse_args()
    
    print("=" * 50)
    print("PIXEL LIFE DEMONSTRATION (PYGAME RENDERER)")
    print("=" * 50)
    
    # Create environment
    env = PixelLifeEnv(H=40, W=40, max_size=100)
    print(f"\n✓ Created environment: {env.H}x{env.W}")
    
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
    print("\nRunning demonstration...")
    print("- Main agent: Tries to keep pixels alive as long as possible")
    print("- Spice agent: Expands universe and tweaks rules")
    print(f"- Max steps: {'∞ (run forever)' if args.run_forever else args.steps}")
    print("\nControls:")
    print("- SPACE: Pause/Unpause")
    print("- ESC/Q: Quit")
    print("- Click: Toggle pixels (for testing)")
    print("\nPress Ctrl+C to stop\n")
    
    step = 0
    try:
        # Determine the loop condition
        if args.run_forever:
            loop_condition = lambda: not env.done
        else:
            loop_condition = lambda: not env.done and step < args.steps
        
        while loop_condition():
            # Spice strategy: Expand universe periodically, tweak rules
            if step % 20 == 0 and step > 0:
                spice_action = 1 + (step // 20) % 4  # Cycle through expansions
            elif step % 30 == 0:
                spice_action = 5  # Tweak rule
            else:
                spice_action = 0  # No-op
            
            # Pixel strategy: Focus on survival and energy management
            pixel_actions = {}
            for i, (y, x) in enumerate(env.live_pixels):
                energy = env.pixel_energy.get((y, x), 0)
                
                if step < 50:
                    # Early game: focus on reproduction to grow population
                    if energy >= env.params['min_energy_to_reproduce']:
                        action_type = 3  # Reproduce
                        direction = (step + i) % 4
                    else:
                        action_type = 2  # Consume to gain energy
                        direction = (step + i) % 4
                elif energy < env.params['min_energy_to_move']:
                    # Low energy: try to consume
                    action_type = 2  # Consume
                    direction = (step + i) % 4
                elif step % 10 == 0:
                    # Occasionally move to explore
                    action_type = 1  # Move
                    direction = (step + i) % 4
                else:
                    # Default: no-op to conserve energy
                    action_type = 0  # No-op
                    direction = 0
                
                pixel_actions[(y, x)] = (action_type, direction)
            
            # Execute step
            obs, rewards, done, info = env.step(spice_action, pixel_actions)
            step += 1
            
            # Update renderer from environment
            renderer.update_from_env()
            
            # Print periodic updates
            if step % 50 == 0:
                print(f"Step {step}: {info['live_pixels']} pixels, avg energy: {info['avg_energy']:.1f}, grid {info['grid_size']}")
            
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
    
    # Run the pygame renderer
    print("\nStarting pygame renderer... Close window or press ESC to exit.")
    renderer.run_with_env()


if __name__ == "__main__":
    main() 