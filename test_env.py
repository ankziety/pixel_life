"""Test script to verify PixelLifeEnv functionality."""

from env import PixelLifeEnv
import numpy as np

def test_env_basic():
    """Test basic environment functionality."""
    print("Testing PixelLifeEnv scaffold...")
    
    # Create environment
    env = PixelLifeEnv(H=30, W=30)
    print(f"✓ Environment created with grid size {env.H}x{env.W}")
    
    # Test reset
    obs = env.reset()
    obs_main, obs_spice = obs
    print(f"✓ Reset successful")
    print(f"  - Initial organisms: {len(env.organisms)}")
    print(f"  - Origin at: {env.origin}") 
    print(f"  - Grid shape: {env.grid.shape}")
    print(f"  - Living pixels: {len(env.pixel_to_org)}")
    
    # Verify initial organism placement
    living_cells = np.sum(env.grid > 0)
    print(f"✓ Initial organism has {living_cells} pixels")
    
    return env

def test_atomic_actions(env):
    """Test all atomic actions."""
    print("\nTesting atomic actions...")
    
    # Get initial organism info
    org_id = list(env.organisms.keys())[0]
    pixels = list(env.organisms[org_id])
    y, x = pixels[0]
    
    # Test split
    print("\n1. Testing split action...")
    initial_orgs = len(env.organisms)
    success = env.do_split(y, x, direction=1)  # Split right
    print(f"  - Split {'succeeded' if success else 'failed'}")
    print(f"  - Organisms: {initial_orgs} → {len(env.organisms)}")
    
    # Test consume (need to create a dead cell first)
    print("\n2. Testing consume action...")
    # Forfeit a pixel to create dead cell
    if env.pixel_to_org:
        test_pixel = list(env.pixel_to_org.keys())[0]
        env.do_forfeit(test_pixel[0], test_pixel[1])
        print(f"  - Created dead cell at {test_pixel}")
        
        # Try to consume it
        if env.pixel_to_org:
            consumer = list(env.pixel_to_org.keys())[0]
            initial_pixels = len(env.pixel_to_org)
            success = env.do_consume(consumer[0], consumer[1], direction=0)
            print(f"  - Consume {'succeeded' if success else 'failed'}")
            print(f"  - Live pixels: {initial_pixels} → {len(env.pixel_to_org)}")
    
    # Test combine
    print("\n3. Testing combine action...")
    if len(env.organisms) > 1:
        initial_orgs = len(env.organisms)
        combiner = list(env.pixel_to_org.keys())[0]
        success = env.do_combine(combiner[0], combiner[1], direction=1)
        print(f"  - Combine {'succeeded' if success else 'failed'}")
        print(f"  - Organisms: {initial_orgs} → {len(env.organisms)}")
    
    # Test universe expansion
    print("\n4. Testing universe expansion...")
    initial_size = (env.H, env.W)
    success = env._expand_universe(direction=1)  # Expand right
    print(f"  - Expansion {'succeeded' if success else 'failed'}")
    print(f"  - Grid size: {initial_size} → ({env.H}, {env.W})")
    
    # Test parameter tweaking
    print("\n5. Testing parameter tweaking...")
    initial_params = env.params.copy()
    tweaked = env._apply_tweak()
    print(f"  - Tweaked parameter: {tweaked}")
    print(f"  - Old value: {initial_params[tweaked]} → New value: {env.params[tweaked]}")

def test_step_logic(env):
    """Test the step function."""
    print("\n\nTesting step logic...")
    
    # Reset for clean test
    env.reset()
    
    # Create pixel actions for all living pixels
    pixel_actions = {}
    for y, x in env.pixel_to_org.keys():
        # Random actions for testing
        action_type = np.random.randint(0, 5)  # 0-4: no-op, split, consume, combine, forfeit
        direction = np.random.randint(0, 4)    # 0-3: up, right, down, left
        pixel_actions[(y, x)] = (action_type, direction)
    
    # Test step with various spice actions
    for spice_action in range(6):  # Test all spice actions
        print(f"\n  Step {spice_action + 1}: Spice action = {spice_action}")
        
        obs, rewards, done, info = env.step(spice_action, pixel_actions)
        obs_main, obs_spice = obs
        reward_main, reward_spice = rewards
        
        print(f"    - Main reward: {reward_main}, Spice reward: {reward_spice}")
        print(f"    - Living pixels: {info['live_pixels']}, Dead: {info['dead_pixels']}")
        print(f"    - Grid size: {info['grid_size']}")
        print(f"    - Done: {done}")
        
        if done:
            break
            
        # Update pixel actions for remaining pixels
        pixel_actions = {(y, x): (0, 0) for y, x in env.pixel_to_org.keys()}

def test_rendering():
    """Test visualization."""
    print("\n\nTesting rendering...")
    env = PixelLifeEnv(H=20, W=20)
    env.reset()
    
    # Run a few steps with rendering
    for i in range(5):
        # Simple actions
        pixel_actions = {(y, x): (1, i % 4) for y, x in env.pixel_to_org.keys()}  # Split in different directions
        spice_action = 0  # No-op
        
        env.step(spice_action, pixel_actions)
        grid = env.render(mode='human')
        
    print("✓ Rendering test complete")

if __name__ == "__main__":
    print("=" * 50)
    print("PIXEL LIFE ENVIRONMENT TEST SUITE")
    print("=" * 50)
    
    # Run tests
    env = test_env_basic()
    test_atomic_actions(env)
    test_step_logic(env)
    test_rendering()
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETE!")
    print("=" * 50)