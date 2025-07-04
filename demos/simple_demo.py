#!/usr/bin/env python3
"""
Simple demo of the Pixel Life environment.
Shows basic environment usage with random actions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import PixelLifeEnv


def simple_demo():
    """Run a simple demonstration of the Pixel Life environment."""
    print("Pixel Life Environment Demo")
    print("=" * 30)
    
    # Create environment
    env = PixelLifeEnv(H=20, W=20)
    print(f"Created environment: {env.H}x{env.W}")
    
    # Reset environment
    obs = env.reset()
    obs_main, obs_spice = obs
    print(f"Initial observation shapes: main={obs_main['grid'].shape}, spice={obs_spice['grid'].shape}")
    
    # Run simulation
    total_steps = 100
    print(f"\nRunning simulation for {total_steps} steps...")
    
    for step in range(total_steps):
        # Random spice action (0-5)
        spice_action = env.spice_action_space.sample()
        
        # Random pixel actions for each living pixel
        pixel_actions = {}
        for coord in env.live_pixels:
            action_type = np.random.randint(0, 4)  # 0=no-op, 1=move, 2=consume, 3=reproduce
            direction = np.random.randint(0, 4)    # 0=up, 1=right, 2=down, 3=left
            pixel_actions[coord] = (action_type, direction)
        
        # Step environment
        obs, rewards, done, info = env.step(spice_action, pixel_actions)
        obs_main, obs_spice = obs
        reward_main, reward_spice = rewards
        
        # Print progress
        if step % 20 == 0:
            print(f"Step {step:3d}: Pixels={len(env.live_pixels):2d}, "
                  f"Main reward={reward_main:6.2f}, Spice reward={reward_spice:6.2f}")
        
        # Render every 10 steps
        if step % 10 == 0:
            env.render()
            plt.pause(0.1)
        
        if done:
            print(f"\nEpisode ended at step {step}")
            break
    
    print(f"\nFinal state: {len(env.live_pixels)} pixels alive")
    print("Demo complete!")
    
    # Keep plot open
    plt.show()


if __name__ == "__main__":
    simple_demo() 