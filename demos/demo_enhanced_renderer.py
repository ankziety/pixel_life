"""Demo script using enhanced renderer with zoom and resizable window."""

import numpy as np
import sys
import os
import time
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import PixelLifeEnv
from enhanced_renderer import EnhancedPixelLifeRenderer


def main():
    """Run the demo with enhanced renderer."""
    parser = argparse.ArgumentParser(description='Pixel Life Demo with Enhanced Renderer')
    parser.add_argument('--steps', type=int, default=5000, 
                       help='Maximum number of steps (default: 5000)')
    parser.add_argument('--run-forever', action='store_true',
                       help='Run until environment naturally ends (ignores --steps)')
    parser.add_argument('--size', type=int, default=100,
                       help='Environment size (default: 100)')
    parser.add_argument('--initial-zoom', type=float, default=0.01,
                       help='Initial zoom level (default: 0.01 = 100x smaller pixels)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("PIXEL LIFE ENHANCED RENDERER DEMONSTRATION")
    print("=" * 60)
    
    # Create environment
    env = PixelLifeEnv(H=args.size, W=args.size, max_size=200)
    print(f"\n✓ Created environment: {env.H}x{env.W}")
    
    # Reset
    obs = env.reset()
    print(f"✓ Initial pixels: {len(env.live_pixels)} pixels")
    
    # Create enhanced pygame renderer
    renderer = EnhancedPixelLifeRenderer(
        env=env,
        width=1400, 
        height=900, 
        initial_zoom=None  # Auto-fit to window
    )
    
    # Run demonstration
    print("\nRunning demonstration...")
    print("- Main agent: Tries to keep pixels alive as long as possible")
    print("- Spice agent: Expands universe and tweaks rules")
    print(f"- Max steps: {'∞ (run forever)' if args.run_forever else args.steps}")
    print(f"- Initial zoom: {args.initial_zoom}x (pixels are {1/args.initial_zoom:.0f}x smaller)")
    print("\nEnhanced Controls:")
    print("- SPACE: Pause/Unpause")
    print("- F: Toggle fullscreen")
    print("- ESC/Q: Quit")
    print("- Mouse wheel: Zoom in/out")
    print("- Middle mouse drag: Pan camera")
    print("- Click: Toggle pixels")
    print("- R: Reset zoom and camera")
    print("- C: Center camera on live pixels")
    print("- +/-: Zoom in/out")
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
            renderer.update_from_env(env)
            
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
    
    # Run the enhanced pygame renderer
    print("\nStarting enhanced pygame renderer...")
    print("Use mouse wheel to zoom, middle drag to pan, F for fullscreen.")
    print("Close window or press ESC to exit.")
    renderer.run_with_env()


if __name__ == "__main__":
    main() 