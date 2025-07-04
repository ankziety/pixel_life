#!/usr/bin/env python3

import time
import numpy as np
from env import PixelLifeEnv


def main():
    print("🧬 Pixel Life Environment Demo")
    print("=" * 40)
    
    # Create environment
    env = PixelLifeEnv(grid_size=24, render_mode='human')
    
    print("Controls:")
    print("  0 - Move Up")
    print("  1 - Move Down") 
    print("  2 - Move Left")
    print("  3 - Move Right")
    print("  4 - Place/Remove Cell")
    print("  q - Quit")
    print("\nStarting demo...")
    
    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    
    try:
        while True:
            env.render()
            
            # Get user input
            try:
                action_input = input(f"Step {step_count} | Population: {np.sum(env.grid)} | Reward: {total_reward:.2f} | Action: ")
                
                if action_input.lower() == 'q':
                    break
                
                action = int(action_input)
                if action not in range(5):
                    print("Invalid action! Please enter 0-4")
                    continue
                    
            except (ValueError, KeyboardInterrupt):
                print("\nDemo interrupted!")
                break
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            print(f"Reward: {reward:.2f} | Population: {info['population']}")
            
            if terminated or truncated:
                print("\nEpisode ended!")
                break
                
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error during demo: {e}")
    finally:
        env.close()
        print("Demo completed!")


def automated_demo():
    """Run an automated demo with predefined actions."""
    print("\n🤖 Running Automated Demo")
    print("=" * 30)
    
    env = PixelLifeEnv(grid_size=16, render_mode='human')
    obs, info = env.reset()
    
    # Predefined action sequence to create interesting patterns
    actions = [
        4, 3, 4, 3, 4, 1, 4, 2, 4,  # Create a line pattern
        0, 0, 4, 3, 4, 1, 1, 4,     # Create another pattern
        2, 2, 4, 0, 4, 3, 3, 4,     # More patterns
    ]
    
    total_reward = 0
    
    try:
        for i, action in enumerate(actions):
            env.render()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step {i+1}: Action {action}, Reward: {reward:.2f}, Population: {info['population']}")
            
            if terminated or truncated:
                break
                
            time.sleep(0.8)
        
        # Let it evolve for a few more steps
        print("\nLetting life evolve...")
        for i in range(20):
            env.render()
            obs, reward, terminated, truncated, info = env.step(np.random.randint(0, 4))  # Random movement
            total_reward += reward
            
            if terminated or truncated:
                break
                
            time.sleep(0.5)
            
    except Exception as e:
        print(f"Error during automated demo: {e}")
    finally:
        env.close()
        print(f"Automated demo completed! Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    try:
        print("Choose demo mode:")
        print("1. Interactive Demo")
        print("2. Automated Demo")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            main()
        elif choice == "2":
            automated_demo()
        else:
            print("Invalid choice, running automated demo...")
            automated_demo()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user!")
    except Exception as e:
        print(f"Unexpected error: {e}")