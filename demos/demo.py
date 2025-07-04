"""Simple demonstration of the Pixel Life environment."""

from env import PixelLifeEnv
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse


def demo_pixel_life():
    """Run a visual demonstration of Pixel Life."""
    parser = argparse.ArgumentParser(description='Pixel Life Demo with Matplotlib Renderer')
    parser.add_argument('--steps', type=int, default=500, 
                       help='Maximum number of steps (default: 500)')
    parser.add_argument('--run-forever', action='store_true',
                       help='Run until environment naturally ends (ignores --steps)')
    args = parser.parse_args()
    
    print("=" * 50)
    print("PIXEL LIFE DEMONSTRATION")
    print("=" * 50)
    
    # Create environment
    env = PixelLifeEnv(H=40, W=40, max_size=100)
    print(f"\n✓ Created environment: {env.H}x{env.W}")
    
    # Reset
    obs = env.reset()
    print(f"✓ Initial pixels: {len(env.live_pixels)} pixels")
    
    # Set up matplotlib for animation
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Run demonstration
    print("\nRunning demonstration...")
    print("- Main agent: Tries to keep pixels alive as long as possible")
    print("- Spice agent: Expands universe and tweaks rules")
    print(f"- Max steps: {'∞ (run forever)' if args.run_forever else args.steps}")
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
            
            # Update visualization
            ax1.clear()
            ax1.imshow(env.grid, cmap='viridis', interpolation='nearest')
            ax1.set_title(f'Step {step}: {info["live_pixels"]} pixels, avg energy: {info["avg_energy"]:.1f}')
            ax1.grid(True, alpha=0.3)
            
            # Show info panel
            ax2.clear()
            ax2.axis('off')
            info_text = f"Step: {step}\n"
            info_text += f"Grid Size: {info['grid_size']}\n"
            info_text += f"Live Pixels: {info['live_pixels']}\n"
            info_text += f"Dead Pixels: {info['dead_pixels']}\n"
            info_text += f"Avg Energy: {info['avg_energy']:.2f}\n"
            info_text += f"Avg Age: {info['avg_age']:.1f}\n"
            info_text += f"\nMain Reward: {rewards[0]:.1f}\n"
            info_text += f"Spice Reward: {rewards[1]:.1f}\n"
            info_text += f"\nLast Spice Action: {['No-op', 'Expand Up', 'Expand Down', 'Expand Left', 'Expand Right', 'Tweak Rule'][spice_action]}\n"
            
            if spice_action == 5 and info['spice_success']:
                info_text += "\nParameters:\n"
                for param, value in info['params'].items():
                    info_text += f"  {param}: {value}\n"
            
            ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes, 
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
            
            plt.tight_layout()
            plt.pause(0.05)
            
            # Print periodic updates
            if step % 50 == 0:
                print(f"Step {step}: {info['live_pixels']} pixels, avg energy: {info['avg_energy']:.1f}, grid {info['grid_size']}")
            
            if done:
                print(f"\nEpisode ended at step {step}!")
                print(f"Final state: {info['live_pixels']} pixels")
                break
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    plt.ioff()
    
    # Show final state
    print("\nFinal statistics:")
    print(f"- Total steps: {step}")
    print(f"- Final grid size: {env.H}x{env.W}")
    print(f"- Living pixels: {len(env.live_pixels)}")
    print(f"- Dead pixels: {len(env.dead_cells)}")
    print(f"- Average energy: {np.mean(list(env.pixel_energy.values())) if env.pixel_energy else 0:.2f}")
    print(f"- Average age: {np.mean(list(env.pixel_ages.values())) if env.pixel_ages else 0:.1f}")
    
    # Keep plot open
    print("\nClose the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    demo_pixel_life()