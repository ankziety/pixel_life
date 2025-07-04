"""Training script for Pixel Life environment using PPO agents."""

import os
import sys
import time
import numpy as np
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from env import PixelLifeEnv


class PixelLifeWrapper:
    """Wrapper to handle the dual-agent nature of PixelLifeEnv for SB3."""
    
    def __init__(self, env, agent_type='main'):
        self.env = env
        self.agent_type = agent_type
        self.other_model = None  # Will be set during training
        
        # Copy relevant attributes
        self.action_space = env.pixel_action_space if agent_type == 'main' else env.spice_action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata if hasattr(env, 'metadata') else {}
        
    def reset(self):
        obs_main, obs_spice = self.env.reset()
        return obs_main if self.agent_type == 'main' else obs_spice
    
    def step(self, action):
        # Get action from the other agent
        if self.other_model is not None:
            other_obs = self._get_other_obs()
            other_action, _ = self.other_model.predict(other_obs, deterministic=False)
        else:
            # Random actions if other model not set
            if self.agent_type == 'main':
                # We are main, need spice action
                other_action = self.env.spice_action_space.sample()
            else:
                # We are spice, need pixel actions
                other_action = {}
                for coord in self.env.pixel_to_org.keys():
                    other_action[coord] = (
                        np.random.randint(0, 5),
                        np.random.randint(0, 4)
                    )
        
        # Execute step based on agent type
        if self.agent_type == 'main':
            # Convert main action to pixel actions dict
            pixel_actions = self._convert_main_action(action)
            (obs_main, obs_spice), (r_main, r_spice), done, info = self.env.step(other_action, pixel_actions)
            obs = obs_main
            reward = r_main
        else:
            # action is spice action, other_action is pixel actions
            (obs_main, obs_spice), (r_main, r_spice), done, info = self.env.step(action, other_action)
            obs = obs_spice
            reward = r_spice
            
        return obs, reward, done, info
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def _get_other_obs(self):
        """Get observation for the other agent."""
        if self.agent_type == 'main':
            return self.env._get_spice_observation()
        else:
            return self.env._get_main_observation()
    
    def _convert_main_action(self, action):
        """Convert main agent's action to pixel actions dict."""
        pixel_actions = {}
        
        # For simplicity, all pixels take the same action initially
        # In a more sophisticated approach, you'd have per-pixel policies
        for coord in self.env.pixel_to_org.keys():
            if isinstance(action, (list, np.ndarray)) and len(action) >= 2:
                pixel_actions[coord] = (int(action[0]), int(action[1]))
            else:
                # Fallback to no-op
                pixel_actions[coord] = (0, 0)
                
        return pixel_actions


def make_env(agent_type='main', env_kwargs=None):
    """Create environment factory for vectorized environments."""
    if env_kwargs is None:
        env_kwargs = {}
        
    def _init():
        env = PixelLifeEnv(**env_kwargs)
        wrapped = PixelLifeWrapper(env, agent_type)
        return Monitor(wrapped)
    
    return _init


def train_pixel_life(
    total_timesteps=1_000_000,
    n_envs=4,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    checkpoint_freq=10000,
    eval_freq=5000,
    device='cpu',
    log_dir='./logs',
):
    """Train both main and spice agents in the Pixel Life environment.
    
    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        n_steps: Number of steps to collect before update
        batch_size: Minibatch size for PPO
        n_epochs: Number of epochs for PPO
        gamma: Discount factor
        checkpoint_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        device: 'cpu' or 'cuda'
        log_dir: Directory for logs and checkpoints
    """
    
    # Create log directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    main_dir = os.path.join(run_dir, "main_agent")
    spice_dir = os.path.join(run_dir, "spice_agent")
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(spice_dir, exist_ok=True)
    
    print(f"Starting training run: {run_dir}")
    print(f"Device: {device}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    
    # Environment kwargs
    env_kwargs = {'H': 30, 'W': 30, 'max_size': 100}
    
    # Create vectorized environments
    print("\nCreating environments...")
    main_vec_env = make_vec_env(
        make_env('main', env_kwargs),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv
    )
    
    spice_vec_env = make_vec_env(
        make_env('spice', env_kwargs),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv
    )
    
    # Create eval environments
    eval_main_env = make_vec_env(make_env('main', env_kwargs), n_envs=1)
    eval_spice_env = make_vec_env(make_env('spice', env_kwargs), n_envs=1)
    
    # Create PPO agents
    print("\nCreating PPO agents...")
    
    # Main agent (controls pixels)
    main_model = PPO(
        "MlpPolicy",  # Using MLP for now, can switch to MlpLstmPolicy
        main_vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        tensorboard_log=main_dir,
        device=device
    )
    
    # Spice agent (adversarial)
    spice_model = PPO(
        "MlpPolicy",
        spice_vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps // 2,  # Spice updates less frequently
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        tensorboard_log=spice_dir,
        device=device
    )
    
    # Set models in environment wrappers for co-evolution
    for env_idx in range(n_envs):
        main_vec_env.envs[env_idx].other_model = spice_model
        spice_vec_env.envs[env_idx].other_model = main_model
    
    # Callbacks
    main_checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(main_dir, "checkpoints"),
        name_prefix="main_model"
    )
    
    spice_checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(spice_dir, "checkpoints"),
        name_prefix="spice_model"
    )
    
    # Training loop with alternating updates
    print("\nStarting training...")
    steps_per_update = n_steps * n_envs
    n_updates = total_timesteps // steps_per_update
    
    for update in range(n_updates):
        current_step = update * steps_per_update
        
        print(f"\nUpdate {update+1}/{n_updates} (Step {current_step:,}/{total_timesteps:,})")
        
        # Train main agent
        print("  Training main agent...")
        main_model.learn(
            total_timesteps=steps_per_update,
            reset_num_timesteps=False,
            callback=main_checkpoint_cb,
            progress_bar=True
        )
        
        # Train spice agent (less frequently)
        if update % 2 == 0:
            print("  Training spice agent...")
            spice_model.learn(
                total_timesteps=steps_per_update // 2,
                reset_num_timesteps=False,
                callback=spice_checkpoint_cb,
                progress_bar=True
            )
        
        # Periodic evaluation
        if (update + 1) % (eval_freq // steps_per_update) == 0:
            print("\n  Evaluating agents...")
            eval_env = PixelLifeEnv(**env_kwargs)
            
            obs = eval_env.reset()
            total_main_reward = 0
            total_spice_reward = 0
            
            for eval_step in range(100):
                # Get actions from both models
                main_obs = obs[0]
                spice_obs = obs[1]
                
                spice_action, _ = spice_model.predict(spice_obs, deterministic=True)
                
                # Simple pixel actions for evaluation
                pixel_actions = {}
                for coord in eval_env.pixel_to_org.keys():
                    pixel_actions[coord] = (1, eval_step % 4)  # Split in different directions
                
                obs, rewards, done, _ = eval_env.step(int(spice_action), pixel_actions)
                total_main_reward += rewards[0]
                total_spice_reward += rewards[1]
                
                if done:
                    break
            
            print(f"    Eval rewards - Main: {total_main_reward:.1f}, Spice: {total_spice_reward:.1f}")
            print(f"    Episode length: {eval_step + 1}")
    
    # Save final models
    print("\nSaving final models...")
    main_model.save(os.path.join(main_dir, "final_model"))
    spice_model.save(os.path.join(spice_dir, "final_model"))
    
    print(f"\nTraining complete! Models saved to: {run_dir}")
    
    return main_model, spice_model, run_dir


if __name__ == "__main__":
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Hyperparameters
    hyperparams = {
        'total_timesteps': 100_000,  # Reduced for testing
        'n_envs': 2,  # Reduced for CPU
        'learning_rate': 3e-4,
        'n_steps': 1024,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'checkpoint_freq': 10000,
        'eval_freq': 5000,
        'device': device,
        'log_dir': './pixel_life_logs'
    }
    
    print("Pixel Life RL Training")
    print("=" * 50)
    print("Hyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    
    # Run training
    try:
        main_model, spice_model, run_dir = train_pixel_life(**hyperparams)
        
        # Quick test of trained models
        print("\nTesting trained models...")
        env = PixelLifeEnv(H=30, W=30)
        obs = env.reset()
        
        for step in range(50):
            main_obs, spice_obs = obs
            spice_action, _ = spice_model.predict(spice_obs, deterministic=True)
            
            # Get pixel actions (simplified for now)
            pixel_actions = {}
            for coord in env.pixel_to_org.keys():
                # In a full implementation, you'd predict per-pixel actions
                pixel_actions[coord] = (1, step % 4)
            
            obs, rewards, done, info = env.step(int(spice_action), pixel_actions)
            
            if step % 10 == 0:
                print(f"Step {step}: Organisms={info['organisms']}, Pixels={info['live_pixels']}")
            
            if done:
                print(f"Episode ended at step {step}")
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()