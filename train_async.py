"""Asynchronous training pipeline for Pixel Life inspired by Sample Factory and APEX.

This implements a high-throughput training system with:
- Rollout workers: Run environment simulations
- Policy workers: Handle model inference on GPU
- Learner: Trains models asynchronously
- Replay buffer: Stores and samples experience
"""

import os
import sys
import time
import queue
import threading
import multiprocessing as mp
from multiprocessing import Queue, Process, Value, Array
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
import pickle

from env_optimized import PixelLifeEnvOptimized


class PixelLifeNet(nn.Module):
    """Shared network architecture for both agents."""
    
    def __init__(self, grid_size, hidden_size=256):
        super().__init__()
        
        # Convolutional encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate conv output size
        def conv_out_size(size, kernel_size, stride, padding):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        size = grid_size
        size = conv_out_size(size, 8, 4, 2)
        size = conv_out_size(size, 4, 2, 1)
        size = conv_out_size(size, 3, 1, 1)
        
        self.conv_output_size = size * size * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        """Forward pass through the network."""
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        
        return features


class MainAgent(nn.Module):
    """Policy network for the main agent controlling pixels."""
    
    def __init__(self, grid_size, num_actions=5, num_directions=4):
        super().__init__()
        self.encoder = PixelLifeNet(grid_size)
        
        # Output heads for each pixel
        self.action_head = nn.Linear(256, num_actions)
        self.direction_head = nn.Linear(256, num_directions)
        self.value_head = nn.Linear(256, 1)
        
    def forward(self, obs, coords=None):
        """Forward pass. If coords provided, output actions for those pixels."""
        features = self.encoder(obs)
        
        # For now, output global policy
        action_logits = self.action_head(features)
        direction_logits = self.direction_head(features)
        value = self.value_head(features)
        
        return action_logits, direction_logits, value


class SpiceAgent(nn.Module):
    """Policy network for the spice agent."""
    
    def __init__(self, grid_size, num_actions=3):
        super().__init__()
        self.encoder = PixelLifeNet(grid_size)
        
        # Single action head
        self.action_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)
        
    def forward(self, obs):
        """Forward pass."""
        features = self.encoder(obs)
        
        action_logits = self.action_head(features)
        value = self.value_head(features)
        
        return action_logits, value


class SharedMemoryBuffer:
    """Fast shared memory buffer for experience replay."""
    
    def __init__(self, capacity, obs_shape, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate arrays
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        
        # Thread-safe lock
        self.lock = threading.Lock()
        
    def add(self, obs, action, reward, next_obs, done):
        """Add experience to buffer."""
        with self.lock:
            self.obs[self.ptr] = obs
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_obs[self.ptr] = next_obs
            self.dones[self.ptr] = done
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample batch of experiences."""
        with self.lock:
            indices = np.random.choice(self.size, batch_size, replace=False)
            
            batch = {
                'obs': torch.FloatTensor(self.obs[indices]).to(self.device),
                'actions': torch.LongTensor(self.actions[indices]).to(self.device),
                'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
                'next_obs': torch.FloatTensor(self.next_obs[indices]).to(self.device),
                'dones': torch.FloatTensor(self.dones[indices]).to(self.device)
            }
            
        return batch


def rollout_worker(worker_id, env_fn, obs_queue, action_queue, exp_queue, 
                   stats_queue, stop_flag):
    """Worker process that runs environment rollouts."""
    
    # Create environments
    num_envs = 4  # Environments per worker
    envs = [env_fn() for _ in range(num_envs)]
    
    # Reset all environments
    observations = []
    for i, env in enumerate(envs):
        obs = env.reset()
        observations.append(obs)
        obs_queue.put((worker_id, i, obs))
    
    # Rollout loop
    episode_rewards = [0.0] * num_envs
    episode_lengths = [0] * num_envs
    
    while not stop_flag.value:
        try:
            # Get actions from policy worker
            actions = {}
            for _ in range(num_envs):
                try:
                    env_idx, spice_action, pixel_actions = action_queue.get(timeout=0.1)
                    actions[env_idx] = (spice_action, pixel_actions)
                except queue.Empty:
                    continue
            
            # Step environments
            for i, env in enumerate(envs):
                if i in actions:
                    spice_action, pixel_actions = actions[i]
                    
                    # Execute step
                    obs, rewards, done, info = env.step(spice_action, pixel_actions)
                    
                    # Track stats
                    episode_rewards[i] += rewards[0]  # Main agent reward
                    episode_lengths[i] += 1
                    
                    # Store experience
                    exp_queue.put({
                        'obs': observations[i],
                        'actions': (spice_action, pixel_actions),
                        'rewards': rewards,
                        'next_obs': obs,
                        'done': done,
                        'info': info
                    })
                    
                    # Handle episode end
                    if done:
                        stats_queue.put({
                            'episode_reward': episode_rewards[i],
                            'episode_length': episode_lengths[i],
                            'final_pixels': info.get('live_pixels', 0)
                        })
                        
                        episode_rewards[i] = 0.0
                        episode_lengths[i] = 0
                        
                        obs = env.reset()
                    
                    # Update observation
                    observations[i] = obs
                    
                    # Send new observation for inference
                    obs_queue.put((worker_id, i, obs))
                    
        except Exception as e:
            print(f"Rollout worker {worker_id} error: {e}")
            break
    
    # Cleanup
    for env in envs:
        env.close()


def policy_worker(main_model, spice_model, obs_queue, action_queue, 
                  param_queue, device, stop_flag):
    """Worker that handles model inference on GPU."""
    
    main_model = main_model.to(device)
    spice_model = spice_model.to(device)
    main_model.eval()
    spice_model.eval()
    
    # Batch inference
    batch_size = 32
    obs_batch = []
    metadata = []
    
    while not stop_flag.value:
        try:
            # Check for model updates
            try:
                new_params = param_queue.get_nowait()
                main_model.load_state_dict(new_params['main'])
                spice_model.load_state_dict(new_params['spice'])
            except queue.Empty:
                pass
            
            # Collect observations for batching
            while len(obs_batch) < batch_size:
                try:
                    worker_id, env_idx, obs = obs_queue.get(timeout=0.01)
                    obs_batch.append(obs)
                    metadata.append((worker_id, env_idx))
                except queue.Empty:
                    break
            
            if obs_batch:
                # Prepare batch
                obs_main_batch = torch.FloatTensor([o[0] for o in obs_batch]).to(device)
                obs_spice_batch = torch.FloatTensor([o[1] for o in obs_batch]).to(device)
                
                with torch.no_grad():
                    # Main agent inference
                    action_logits, dir_logits, _ = main_model(obs_main_batch)
                    
                    # Sample actions
                    action_probs = F.softmax(action_logits, dim=-1)
                    dir_probs = F.softmax(dir_logits, dim=-1)
                    
                    main_actions = torch.multinomial(action_probs, 1).squeeze(-1)
                    main_dirs = torch.multinomial(dir_probs, 1).squeeze(-1)
                    
                    # Spice agent inference
                    spice_logits, _ = spice_model(obs_spice_batch)
                    spice_probs = F.softmax(spice_logits, dim=-1)
                    spice_actions = torch.multinomial(spice_probs, 1).squeeze(-1)
                
                # Send actions back
                for i, (worker_id, env_idx) in enumerate(metadata):
                    # For now, apply same action to all pixels (simplified)
                    pixel_actions = {}
                    
                    # Get live pixels from observation
                    obs_grid = obs_batch[i][0]
                    live_coords = np.argwhere(obs_grid > 0)
                    
                    for y, x in live_coords[:10]:  # Limit for performance
                        pixel_actions[(int(y), int(x))] = (
                            main_actions[i].item(),
                            main_dirs[i].item()
                        )
                    
                    action_queue.put((
                        env_idx,
                        spice_actions[i].item(),
                        pixel_actions
                    ))
                
                # Clear batch
                obs_batch = []
                metadata = []
                
        except Exception as e:
            print(f"Policy worker error: {e}")
            break


def learner_process(main_model, spice_model, exp_queue, param_queue, 
                   stats_queue, device, stop_flag):
    """Process that trains the models asynchronously."""
    
    main_model = main_model.to(device)
    spice_model = spice_model.to(device)
    
    # Optimizers
    main_optimizer = torch.optim.Adam(main_model.parameters(), lr=3e-4)
    spice_optimizer = torch.optim.Adam(spice_model.parameters(), lr=3e-4)
    
    # Replay buffers
    main_buffer = SharedMemoryBuffer(100000, (50, 50), device)
    spice_buffer = SharedMemoryBuffer(100000, (200, 200), device)
    
    # Training stats
    update_count = 0
    param_update_freq = 100
    
    while not stop_flag.value:
        try:
            # Collect experience
            exp_count = 0
            while exp_count < 1000:  # Collect batch of experience
                try:
                    exp = exp_queue.get(timeout=0.1)
                    
                    # Add to buffers
                    obs_main, obs_spice = exp['obs']
                    next_obs_main, next_obs_spice = exp['next_obs']
                    rewards = exp['rewards']
                    done = exp['done']
                    
                    # Simplified: store single action
                    main_buffer.add(obs_main, 0, rewards[0], next_obs_main, done)
                    spice_buffer.add(obs_spice, 0, rewards[1], next_obs_spice, done)
                    
                    exp_count += 1
                    
                except queue.Empty:
                    continue
            
            # Train if enough experience
            if main_buffer.size > 1000 and spice_buffer.size > 1000:
                # Train main agent
                for _ in range(10):
                    batch = main_buffer.sample(256)
                    
                    # Compute loss (simplified DQN)
                    obs = batch['obs']
                    rewards = batch['rewards']
                    
                    action_logits, _, values = main_model(obs)
                    
                    # Value loss
                    value_loss = F.mse_loss(values.squeeze(), rewards)
                    
                    # Policy loss (simplified)
                    policy_loss = -action_logits.mean()
                    
                    # Total loss
                    loss = value_loss + 0.01 * policy_loss
                    
                    main_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(main_model.parameters(), 0.5)
                    main_optimizer.step()
                
                # Train spice agent
                for _ in range(10):
                    batch = spice_buffer.sample(256)
                    
                    obs = batch['obs']
                    rewards = batch['rewards']
                    
                    action_logits, values = spice_model(obs)
                    
                    # Value loss
                    value_loss = F.mse_loss(values.squeeze(), rewards)
                    
                    # Policy loss
                    policy_loss = -action_logits.mean()
                    
                    # Total loss
                    loss = value_loss + 0.01 * policy_loss
                    
                    spice_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(spice_model.parameters(), 0.5)
                    spice_optimizer.step()
                
                update_count += 1
                
                # Send parameter updates
                if update_count % param_update_freq == 0:
                    param_queue.put({
                        'main': main_model.state_dict(),
                        'spice': spice_model.state_dict()
                    })
                    
                    # Log stats
                    stats_queue.put({
                        'update_count': update_count,
                        'main_buffer_size': main_buffer.size,
                        'spice_buffer_size': spice_buffer.size
                    })
                
        except Exception as e:
            print(f"Learner error: {e}")
            break


def train_async(num_workers=4, device='cuda'):
    """Main asynchronous training loop."""
    
    # Create models
    main_model = MainAgent(grid_size=50)
    spice_model = SpiceAgent(grid_size=200)
    
    # Create queues
    obs_queue = mp.Queue(maxsize=1000)
    action_queue = mp.Queue(maxsize=1000)
    exp_queue = mp.Queue(maxsize=10000)
    param_queue = mp.Queue(maxsize=10)
    stats_queue = mp.Queue(maxsize=1000)
    
    # Stop flag
    stop_flag = mp.Value('b', False)
    
    # Environment factory
    def env_fn():
        return PixelLifeEnvOptimized(H=50, W=50)
    
    # Start processes
    processes = []
    
    # Rollout workers
    for i in range(num_workers):
        p = mp.Process(
            target=rollout_worker,
            args=(i, env_fn, obs_queue, action_queue, exp_queue, 
                  stats_queue, stop_flag)
        )
        p.start()
        processes.append(p)
    
    # Policy worker
    p = mp.Process(
        target=policy_worker,
        args=(main_model, spice_model, obs_queue, action_queue,
              param_queue, device, stop_flag)
    )
    p.start()
    processes.append(p)
    
    # Learner
    p = mp.Process(
        target=learner_process,
        args=(main_model, spice_model, exp_queue, param_queue,
              stats_queue, device, stop_flag)
    )
    p.start()
    processes.append(p)
    
    # Main monitoring loop
    print("Starting asynchronous training...")
    print(f"Workers: {num_workers}, Device: {device}")
    
    start_time = time.time()
    total_episodes = 0
    total_updates = 0
    
    try:
        while True:
            # Collect stats
            while not stats_queue.empty():
                stat = stats_queue.get()
                
                if 'episode_reward' in stat:
                    total_episodes += 1
                    if total_episodes % 100 == 0:
                        elapsed = time.time() - start_time
                        eps_per_sec = total_episodes / elapsed
                        print(f"Episodes: {total_episodes}, "
                              f"Reward: {stat['episode_reward']:.1f}, "
                              f"FPS: {eps_per_sec:.1f}")
                
                elif 'update_count' in stat:
                    total_updates = stat['update_count']
                    print(f"Updates: {total_updates}, "
                          f"Buffer sizes: {stat['main_buffer_size']}/{stat['spice_buffer_size']}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping training...")
        stop_flag.value = True
        
        # Wait for processes to finish
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        print("Training stopped.")


if __name__ == "__main__":
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Start training
    train_async(num_workers=mp.cpu_count(), device=device)