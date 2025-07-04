#!/usr/bin/env python3
"""
Simple test script for PixelLifeEnv to verify basic functionality.
"""

import numpy as np
from env import PixelLifeEnv


def test_basic_functionality():
    """Test basic environment functionality."""
    print("=== Testing PixelLifeEnv Basic Functionality ===")
    
    # Create environment
    env = PixelLifeEnv(H=10, W=10, max_steps=50)
    print(f"Created environment: {env.H}x{env.W}, max_steps={env.max_steps}")
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"Reset successful. Initial organisms: {len(env.organisms)}")
    print(f"Initial total pixels: {env._get_total_live_pixels()}")
    print(f"Origin at: {env.origin}")
    
    # Print initial state
    print("\nInitial state:")
    env.render(mode='human')
    
    # Test observations
    print(f"Main observation shape: {obs['main'].shape}")
    print(f"Spice observation shape: {obs['spice'].shape}")
    print(f"Spice observation: {obs['spice']}")
    
    return env, obs


def test_actions():
    """Test different types of actions."""
    print("\n=== Testing Actions ===")
    
    env, _ = test_basic_functionality()
    
    # Test spice actions
    print("\nTesting spice actions:")
    for spice_action in [0, 1, 2]:  # no-op, expand, tweak
        print(f"\nSpice action {spice_action}:")
        
        # Get current live pixels for pixel actions
        live_pixels = list(env.pixel_to_org.keys())
        if live_pixels:
            # Create pixel actions (some pixels try to split)
            pixel_actions = {}
            for i, coord in enumerate(live_pixels[:3]):  # Test first 3 pixels
                pixel_actions[coord] = i % 4  # Cycle through actions
        else:
            pixel_actions = {}
        
        print(f"Live pixels: {live_pixels}")
        print(f"Pixel actions: {pixel_actions}")
        
        # Execute step
        obs, rewards, terminated, truncated, info = env.step(spice_action, pixel_actions)
        
        print(f"Step result:")
        print(f"  Rewards: {rewards}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Info: {info}")
        print(f"  Grid size: {env.H}x{env.W}")
        print(f"  Total organisms: {len(env.organisms)}")
        
        env.render(mode='human')
        
        if terminated:
            print("Episode terminated!")
            break


def test_reset_options():
    """Test different reset options."""
    print("\n=== Testing Reset Options ===")
    
    env = PixelLifeEnv(H=8, W=8, max_steps=100)
    
    # Test different start modes
    start_modes = ['single_seed', 'random_seeds', 'empty']
    
    for mode in start_modes:
        print(f"\nTesting start_mode: {mode}")
        
        if mode == 'random_seeds':
            options = {'start_mode': mode, 'num_seeds': 3}
        else:
            options = {'start_mode': mode}
        
        obs, info = env.reset(seed=123, options=options)
        print(f"Organisms: {len(env.organisms)}")
        print(f"Total pixels: {env._get_total_live_pixels()}")
        
        env.render(mode='human')


def test_pixel_actions():
    """Test specific pixel actions in detail."""
    print("\n=== Testing Pixel Actions in Detail ===")
    
    env = PixelLifeEnv(H=6, W=6, max_steps=20)
    obs, info = env.reset(seed=456)
    
    print("Initial state:")
    env.render(mode='human')
    
    # Test each action type
    action_names = ['split', 'consume', 'combine', 'forfeit']
    
    for step, action_type in enumerate(action_names):
        live_pixels = list(env.pixel_to_org.keys())
        if not live_pixels:
            print(f"No live pixels for {action_type} test")
            continue
            
        print(f"\nStep {step}: Testing {action_type} action")
        
        # Pick first live pixel and apply specific action
        target_pixel = live_pixels[0]
        pixel_actions = {target_pixel: step}  # 0=split, 1=consume, 2=combine, 3=forfeit
        
        print(f"Applying {action_type} to pixel {target_pixel}")
        
        obs, rewards, terminated, truncated, info = env.step(0, pixel_actions)  # spice no-op
        
        print(f"Result - Organisms: {len(env.organisms)}, Pixels: {env._get_total_live_pixels()}")
        print(f"Rewards: {rewards}")
        
        env.render(mode='human')
        
        if terminated:
            break


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n=== Testing Edge Cases ===")
    
    env = PixelLifeEnv(H=5, W=5, max_steps=10)
    
    # Test empty grid
    obs, info = env.reset(options={'start_mode': 'empty'})
    print("Empty grid test:")
    print(f"Organisms: {len(env.organisms)}, Pixels: {env._get_total_live_pixels()}")
    
    # Try step with no pixels
    obs, rewards, terminated, truncated, info = env.step(0, {})
    print(f"Step with no pixels - Terminated: {terminated}")
    
    # Test invalid actions (should be handled gracefully)
    env.reset(seed=789)
    live_pixels = list(env.pixel_to_org.keys())
    if live_pixels:
        invalid_pixel_actions = {(99, 99): 0}  # Invalid coordinate
        obs, rewards, terminated, truncated, info = env.step(0, invalid_pixel_actions)
        print("Invalid coordinate handled gracefully")


def run_episode():
    """Run a full episode to test complete flow."""
    print("\n=== Running Full Episode ===")
    
    env = PixelLifeEnv(H=8, W=8, max_steps=30)
    obs, info = env.reset(seed=999)
    
    print("Starting episode...")
    
    step = 0
    while step < 20:  # Limit to 20 steps for demo
        live_pixels = list(env.pixel_to_org.keys())
        
        if not live_pixels:
            print("No live pixels remaining!")
            break
        
        # Random actions
        spice_action = np.random.choice([0, 1, 2])
        pixel_actions = {}
        
        # Give random actions to some pixels
        for pixel in live_pixels[:min(3, len(live_pixels))]:
            pixel_actions[pixel] = np.random.choice([0, 1, 2, 3])
        
        obs, rewards, terminated, truncated, info = env.step(spice_action, pixel_actions)
        
        if step % 5 == 0:  # Print every 5 steps
            print(f"\nStep {step}:")
            print(f"  Rewards: {rewards}")
            print(f"  Organisms: {len(env.organisms)}, Pixels: {env._get_total_live_pixels()}")
            print(f"  Grid size: {env.H}x{env.W}")
            env.render(mode='human')
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"Final rewards: {rewards}")
            break
        
        step += 1
    
    print("Episode completed!")


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_actions()
        test_reset_options()
        test_pixel_actions()
        test_edge_cases()
        run_episode()
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()