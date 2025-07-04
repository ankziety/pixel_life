"""Simple demonstration of the Pixel Life environment."""

from env import PixelLifeEnv
import matplotlib.pyplot as plt
import numpy as np
import time

def demo_pixel_life():
    """Run a visual demonstration of Pixel Life."""
    print("=" * 50)
    print("PIXEL LIFE DEMONSTRATION")
    print("=" * 50)
    
    # Create environment
    env = PixelLifeEnv(H=40, W=40, max_size=100)
    print(f"\n✓ Created environment: {env.H}x{env.W}")
    
    # Reset
    obs = env.reset()
    print(f"✓ Initial organism at {env.origin} with {len(env.pixel_to_org)} pixels")
    
    # Set up matplotlib for animation
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Run demonstration
    print("\nRunning demonstration...")
    print("- Main agent: Tries to grow organisms")
    print("- Spice agent: Expands universe and tweaks rules")
    print("\nPress Ctrl+C to stop\n")
    
    step = 0
    try:
        while not env.done and step < 500:
            # Spice strategy: Expand universe periodically, tweak rules
            if step % 20 == 0 and step > 0:
                spice_action = 1 + (step // 20) % 4  # Cycle through expansions
            elif step % 30 == 0:
                spice_action = 5  # Tweak rule
            else:
                spice_action = 0  # No-op
            
            # Pixel strategy: Mix of splitting and consuming
            pixel_actions = {}
            for i, (y, x) in enumerate(env.pixel_to_org.keys()):
                if step < 50:
                    # Early game: focus on splitting
                    action_type = 1  # Split
                    direction = (step + i) % 4
                elif step % 5 == 0:
                    # Occasionally try to consume
                    action_type = 2  # Consume
                    direction = np.random.randint(0, 4)
                else:
                    # Default: split in various directions
                    action_type = 1
                    direction = (step + i) % 4
                
                pixel_actions[(y, x)] = (action_type, direction)
            
            # Execute step
            obs, rewards, done, info = env.step(spice_action, pixel_actions)
            step += 1
            
            # Update visualization
            ax1.clear()
            ax1.imshow(env.grid, cmap='viridis', interpolation='nearest')
            ax1.set_title(f'Step {step}: {info["organisms"]} organisms, {info["live_pixels"]} pixels')
            ax1.grid(True, alpha=0.3)
            
            # Mark origin
            if env.origin:
                ax1.plot(env.origin[1], env.origin[0], 'r*', markersize=10)
            
            # Show info panel
            ax2.clear()
            ax2.axis('off')
            info_text = f"Step: {step}\n"
            info_text += f"Grid Size: {info['grid_size']}\n"
            info_text += f"Organisms: {info['organisms']}\n"
            info_text += f"Live Pixels: {info['live_pixels']}\n"
            info_text += f"Dead Pixels: {info['dead_pixels']}\n"
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
                print(f"Step {step}: {info['organisms']} organisms, {info['live_pixels']} pixels, grid {info['grid_size']}")
            
            if done:
                print(f"\nEpisode ended at step {step}!")
                print(f"Final state: {info['organisms']} organisms, {info['live_pixels']} pixels")
                break
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    plt.ioff()
    
    # Show final state
    print("\nFinal statistics:")
    print(f"- Total steps: {step}")
    print(f"- Final grid size: {env.H}x{env.W}")
    print(f"- Living organisms: {len(env.organisms)}")
    print(f"- Total living pixels: {len(env.pixel_to_org)}")
    print(f"- Dead pixels: {len(env.dead_cells)}")
    
    # Keep plot open
    print("\nClose the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    demo_pixel_life()