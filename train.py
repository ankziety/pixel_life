#!/usr/bin/env python3

import os
import time
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env import PixelLifeEnv


def make_env(rank: int, seed: int = 0) -> Callable:
    def _init():
        env = PixelLifeEnv(grid_size=16, max_steps=500)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_agent(algorithm='PPO', total_timesteps=100000, n_envs=4):
    print(f"🤖 Training {algorithm} agent on Pixel Life Environment")
    print("=" * 60)
    
    # Create training environment
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])
    
    # Create evaluation environment
    eval_env = PixelLifeEnv(grid_size=16, max_steps=500)
    eval_env = Monitor(eval_env)
    
    # Create model directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Setup model
    if algorithm == 'PPO':
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./logs/"
        )
    elif algorithm == 'DQN':
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=10000,
            batch_size=32,
            target_update_interval=1000,
            train_freq=4,
            gradient_steps=1,
            exploration_final_eps=0.1,
            exploration_fraction=0.3,
            verbose=1,
            tensorboard_log="./logs/"
        )
    elif algorithm == 'A2C':
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./logs/"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix=f"pixel_life_{algorithm.lower()}"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Train the model
    print(f"Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds!")
        
        # Save final model
        model.save(f"models/pixel_life_{algorithm.lower()}_final")
        print(f"Model saved as models/pixel_life_{algorithm.lower()}_final")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user!")
        model.save(f"models/pixel_life_{algorithm.lower()}_interrupted")
        print(f"Model saved as models/pixel_life_{algorithm.lower()}_interrupted")
    
    finally:
        env.close()
        eval_env.close()
    
    return model


def evaluate_agent(model_path: str, episodes: int = 10, render: bool = True):
    print(f"🔍 Evaluating agent: {model_path}")
    print("=" * 40)
    
    # Load model
    if 'ppo' in model_path.lower():
        model = PPO.load(model_path)
    elif 'dqn' in model_path.lower():
        model = DQN.load(model_path)
    elif 'a2c' in model_path.lower():
        model = A2C.load(model_path)
    else:
        print("❌ Cannot determine model type from filename")
        return
    
    # Create evaluation environment
    env = PixelLifeEnv(grid_size=16, max_steps=500, render_mode='human' if render else None)
    
    episode_rewards = []
    episode_populations = []
    
    try:
        for episode in range(episodes):
            obs, info = env.reset()
            total_reward = 0
            step_count = 0
            max_population = 0
            
            print(f"\n--- Episode {episode + 1} ---")
            
            while True:
                if render:
                    env.render()
                    time.sleep(0.1)
                
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                step_count += 1
                max_population = max(max_population, info['population'])
                
                if step_count % 50 == 0:
                    print(f"Step {step_count}, Population: {info['population']}, Reward: {reward:.2f}")
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            episode_populations.append(max_population)
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Steps: {step_count}")
            print(f"  Max Population: {max_population}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user!")
    
    finally:
        env.close()
    
    # Print summary statistics
    if episode_rewards:
        print(f"\n📊 Evaluation Summary:")
        print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Max Population: {np.mean(episode_populations):.2f} ± {np.std(episode_populations):.2f}")
        print(f"Best Episode Reward: {max(episode_rewards):.2f}")
        print(f"Worst Episode Reward: {min(episode_rewards):.2f}")


def compare_algorithms():
    print("🏆 Comparing Different RL Algorithms")
    print("=" * 50)
    
    algorithms = ['PPO', 'DQN', 'A2C']
    timesteps = 50000
    
    results = {}
    
    for algo in algorithms:
        print(f"\n{'='*20} Training {algo} {'='*20}")
        try:
            model = train_agent(algorithm=algo, total_timesteps=timesteps, n_envs=2)
            
            # Quick evaluation
            env = PixelLifeEnv(grid_size=16, max_steps=200)
            rewards = []
            
            for _ in range(5):
                obs, _ = env.reset()
                total_reward = 0
                
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break
                
                rewards.append(total_reward)
            
            env.close()
            results[algo] = np.mean(rewards)
            print(f"{algo} average reward: {results[algo]:.2f}")
            
        except Exception as e:
            print(f"Error training {algo}: {e}")
            results[algo] = 0
    
    # Print comparison
    print(f"\n🎯 Algorithm Comparison Results:")
    for algo, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{algo}: {score:.2f}")


def main():
    print("🧬 Pixel Life RL Training Suite")
    print("=" * 40)
    print("1. Train PPO Agent")
    print("2. Train DQN Agent")
    print("3. Train A2C Agent")
    print("4. Compare All Algorithms")
    print("5. Evaluate Existing Model")
    print("6. Quick Training Test")
    
    try:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            train_agent('PPO', total_timesteps=100000, n_envs=4)
        elif choice == "2":
            train_agent('DQN', total_timesteps=100000, n_envs=1)  # DQN doesn't support multi-env
        elif choice == "3":
            train_agent('A2C', total_timesteps=100000, n_envs=4)
        elif choice == "4":
            compare_algorithms()
        elif choice == "5":
            model_path = input("Enter model path: ").strip()
            if os.path.exists(model_path):
                evaluate_agent(model_path, episodes=5)
            else:
                print("❌ Model file not found!")
        elif choice == "6":
            print("Running quick training test...")
            train_agent('PPO', total_timesteps=10000, n_envs=2)
        else:
            print("Invalid choice. Running quick training test...")
            train_agent('PPO', total_timesteps=10000, n_envs=2)
            
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()