"""Training script for Pixel Life environment using PPO agents."""

import os
import sys
import time
import numpy as np
from datetime import datetime
import argparse

import gymnasium as gym
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from env import PixelLifeEnv


class PixelLifeWrapper(gym.Env):
    """Wrapper to handle the dual-agent nature of PixelLifeEnv for SB3."""
    
    def __init__(self, env, agent_type='main'):
        self.env = env
        self.agent_type = agent_type
        self.other_model = None  # Will be set during training
        
        # Copy relevant attributes and flatten action space for SB3 compatibility
        if agent_type == 'main':
            # Flatten MultiDiscrete([5, 4]) to Discrete(20) for pixel actions
            # action_type * 4 + direction
            self.action_space = gym.spaces.Discrete(5 * 4)  # 20 possible actions
        else:
            self.action_space = env.spice_action_space
        
        # Flatten observation space for SB3 compatibility
        # Original: Dict with 'grid', 'params', 'tick'
        # Flattened: concatenated array [grid_flattened, params, tick]
        grid_size = env.observation_space['grid'].shape[0] * env.observation_space['grid'].shape[1]
        params_size = env.observation_space['params'].shape[0]
        tick_size = env.observation_space['tick'].shape[0]
        total_size = grid_size + params_size + tick_size
        
        self.observation_space = gym.spaces.Box(
            low=-1, 
            high=10000, 
            shape=(total_size,), 
            dtype=np.float32
        )
        self.metadata = env.metadata if hasattr(env, 'metadata') else {}
        
    def reset(self, seed=None, options=None):
        (obs_main, obs_spice), info = self.env.reset(seed=seed, options=options)
        obs = obs_main if self.agent_type == 'main' else obs_spice
        return self._flatten_observation(obs), info
    
    def _flatten_observation(self, obs_dict):
        """Flatten dict observation to array for SB3 compatibility."""
        grid_flat = obs_dict['grid'].flatten().astype(np.float32)
        params = obs_dict['params'].astype(np.float32)
        tick = obs_dict['tick'].astype(np.float32)
        return np.concatenate([grid_flat, params, tick])
    
    def step(self, action):
        # Get action from the other agent
        if self.other_model is not None:
            other_obs = self._get_other_obs()
            prediction = self.other_model.predict(other_obs, deterministic=False)
            other_action = prediction[0]
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
            
        return self._flatten_observation(obs), reward, done, info
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def _get_other_obs(self):
        """Get observation for the other agent."""
        if self.agent_type == 'main':
            obs_dict = self.env._get_spice_observation()
        else:
            obs_dict = self.env._get_main_observation()
        return self._flatten_observation(obs_dict)
    
    def _convert_main_action(self, action):
        """Convert main agent's flattened action to pixel actions dict."""
        pixel_actions = {}
        
        # Convert flattened action (0-19) back to (action_type, direction)
        action_type = action // 4  # 0-4
        direction = action % 4     # 0-3
        
        # For simplicity, all pixels take the same action initially
        # In a more sophisticated approach, you'd have per-pixel policies
        for coord in self.env.pixel_to_org.keys():
            pixel_actions[coord] = (action_type, direction)
                
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
    no_tensorboard=False,
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
    
    # Create vectorized environments for main agent
    print("\nCreating environments...")
    main_vec_env = make_vec_env(
        make_env('main', env_kwargs),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv
    )
    
    # For spice agent, use the raw PixelLifeEnv (not wrapped in Monitor) for DQN
    spice_env = PixelLifeEnv(**env_kwargs)
    spice_env = PixelLifeWrapper(spice_env, agent_type='spice')

    # Debug: Print action and observation spaces
    print("\nMain agent action space:", main_vec_env.action_space)
    print("Main agent observation space:", main_vec_env.observation_space)
    print("Spice agent action space:", spice_env.action_space)
    print("Spice agent observation space:", spice_env.observation_space)

    # Create eval environments
    eval_main_env = make_vec_env(make_env('main', env_kwargs), n_envs=1)
    eval_spice_env = make_vec_env(make_env('spice', env_kwargs), n_envs=1)

    # Create PPO agent for main agent (controls pixels)
    main_model = PPO(
        "MlpPolicy",  # Using MLP for now, can switch to MlpLstmPolicy
        main_vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        tensorboard_log=None if no_tensorboard else main_dir,
        device=device
    )

    # Create DQN agent for spice agent (adversarial)
    spice_model = DQN(
        "MlpPolicy",
        spice_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        verbose=1,
        tensorboard_log=None if no_tensorboard else spice_dir,
        device=device
    )

    # Set models in environment wrappers for co-evolution
    for env_idx in range(n_envs):
        # Access the wrapped environment inside the Monitor wrapper
        main_vec_env.envs[env_idx].env.other_model = spice_model
    # For spice_env, set the main_model as other_model
    spice_env.other_model = main_model
    
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
            eval_wrapped = PixelLifeWrapper(eval_env, agent_type='spice')
            
            obs, _ = eval_wrapped.reset()
            total_main_reward = 0
            total_spice_reward = 0
            
            for eval_step in range(100):
                # Get actions from spice model
                spice_action, _ = spice_model.predict(obs, deterministic=True)
                
                # Simple pixel actions for evaluation
                pixel_actions = {}
                for coord in eval_env.pixel_to_org.keys():
                    pixel_actions[coord] = (1, eval_step % 4)  # Split in different directions
                
                # Convert spice action and step
                obs, reward, done, info = eval_wrapped.step(int(spice_action))
                total_spice_reward += reward
                
                if done:
                    break
            
            print(f"    Eval rewards - Spice: {total_spice_reward:.1f}")
            print(f"    Episode length: {eval_step + 1}")
    
    # Save final models
    print("\nSaving final models...")
    main_model.save(os.path.join(main_dir, "final_model"))
    spice_model.save(os.path.join(spice_dir, "final_model"))
    
    print(f"\nTraining complete! Models saved to: {run_dir}")
    
    return main_model, spice_model, run_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RL agents in Pixel Life environment')
    
    parser.add_argument('--total-timesteps', type=int, default=100_000,
                        help='Total training timesteps (default: 100,000)')
    parser.add_argument('--n-envs', type=int, default=2,
                        help='Number of parallel environments (default: 2)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--n-steps', type=int, default=1024,
                        help='Number of steps per environment per update (default: 1024)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Number of epochs for PPO (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--checkpoint-freq', type=int, default=10000,
                        help='Save checkpoint every N steps (default: 10000)')
    parser.add_argument('--eval-freq', type=int, default=5000,
                        help='Evaluate every N steps (default: 5000)')
    parser.add_argument('--log-dir', type=str, default='./pixel_life_logs',
                        help='Directory for logs and checkpoints (default: ./pixel_life_logs)')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable tensorboard logging')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage (default: auto-detect GPU)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Check if CUDA is available
    if args.cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Hyperparameters from args
    hyperparams = {
        'total_timesteps': args.total_timesteps,
        'n_envs': args.n_envs,
        'learning_rate': args.learning_rate,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'checkpoint_freq': args.checkpoint_freq,
        'eval_freq': args.eval_freq,
        'device': device,
        'log_dir': args.log_dir,
        'no_tensorboard': args.no_tensorboard
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
        test_wrapped = PixelLifeWrapper(env, agent_type='spice')
        obs, _ = test_wrapped.reset()
        
        for step in range(50):
            spice_action, _ = spice_model.predict(obs, deterministic=True)
            
            # Get pixel actions (simplified for now)
            pixel_actions = {}
            for coord in env.pixel_to_org.keys():
                # In a full implementation, you'd predict per-pixel actions
                pixel_actions[coord] = (1, step % 4)
            
            obs, reward, done, info = test_wrapped.step(int(spice_action))
            
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