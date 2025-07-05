#!/usr/bin/env python3
"""
Pixel Life: Unified Command Line Interface
A comprehensive tool for running all Pixel Life environment modes.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json
import platform
import psutil

# Pygame import for rendering
try:
    import pygame
except ImportError:
    pygame = None

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import PixelLifeEnv
from train import train_pixel_life, PixelLifeWrapper, make_env
from per_pixel_ai import PerPixelAISystem, PerPixelObservationWrapper
from continual_learning import ContinualLearningSystem
from basic_renderer import PixelLifeRenderer
from enhanced_renderer import EnhancedPixelLifeRenderer
from stable_baselines3 import PPO, DQN
from src.model_manager import ModelManager
from src.experiment_manager import ExperimentManager
from src.system_monitor import SystemMonitor

# Apple Silicon acceleration imports
try:
    from apple_acceleration import create_accelerated_env, benchmark_acceleration
    from accelerated_training import train_accelerated_pixel_life, train_custom_model
    from accelerated_renderer import create_renderer, BenchmarkRenderer
    APPLE_ACCELERATION_AVAILABLE = True
except ImportError:
    APPLE_ACCELERATION_AVAILABLE = False
    print("⚠️  Apple Silicon acceleration not available. Install required packages.")


def run_basic_demo(args):
    """Run basic environment demonstration."""
    print("Running Basic Environment Demo")
    print("=" * 40)
    
    # Use accelerated environment if requested
    if hasattr(args, 'accelerated') and args.accelerated and APPLE_ACCELERATION_AVAILABLE:
        print("🚀 Using Apple Silicon acceleration")
        env = create_accelerated_env(H=args.size, W=args.size)
    else:
        env = PixelLifeEnv(H=args.size, W=args.size)
    obs = env.reset()
    
    total_steps = args.steps
    for step in range(total_steps):
        # Random spice action
        spice_action = env.spice_action_space.sample()
        
        # Random pixel actions
        pixel_actions = {}
        for coord in env.live_pixels:
            action_type = np.random.randint(0, 4)
            direction = np.random.randint(0, 4)
            pixel_actions[coord] = (action_type, direction)
        
        obs, rewards, done, truncated, info = env.step(spice_action, pixel_actions)
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Pixels={len(env.live_pixels):2d}, "
                  f"Main reward={rewards[0]:6.2f}, Spice reward={rewards[1]:6.2f}")
        
        if args.render and step % 10 == 0:
            env.render()
            plt.pause(0.1)
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Final state: {len(env.live_pixels)} pixels alive")
    if args.render:
        plt.show()


def run_ai_demo(args):
    """Run AI agent demonstration."""
    print("Running AI Agent Demo")
    print("=" * 40)
    
    # Use accelerated environment if requested
    if hasattr(args, 'accelerated') and args.accelerated and APPLE_ACCELERATION_AVAILABLE:
        print("🚀 Using Apple Silicon acceleration")
        env = create_accelerated_env(H=args.size, W=args.size)
    else:
        env = PixelLifeEnv(H=args.size, W=args.size)
    renderer = PixelLifeRenderer(env)
    
    # Create simple AI models
    main_env = PixelLifeWrapper(env, 'main')
    spice_env = PixelLifeWrapper(env, 'spice')
    
    main_model = PPO("MlpPolicy", main_env, verbose=0)
    spice_model = PPO("MlpPolicy", spice_env, verbose=0)
    
    # Quick training
    print("Training AI agents...")
    try:
        main_model.learn(total_timesteps=10000)
        spice_model.learn(total_timesteps=10000)
    except Exception as e:
        print(f"Training failed: {e}")
        print("Running demo with random actions instead...")
        main_model = None
        spice_model = None
    
    # Run demo
    obs, _ = env.reset()
    for step in range(args.steps):
        obs_main, obs_spice = obs
        
        # Get AI actions or use random actions if training failed
        if spice_model is not None:
            try:
                spice_action, _ = spice_model.predict(spice_env._flatten_observation(obs_spice), deterministic=True)
            except:
                spice_action = env.spice_action_space.sample()
        else:
            spice_action = env.spice_action_space.sample()
            
        if main_model is not None:
            try:
                main_action, _ = main_model.predict(main_env._flatten_observation(obs_main), deterministic=True)
            except:
                main_action = np.random.randint(0, 20)
        else:
            main_action = np.random.randint(0, 20)
        
        # Convert main action to pixel actions
        pixel_actions = {}
        for coord in env.live_pixels:
            pixel_actions[coord] = (int(main_action % 4), int(main_action // 4))
        
        obs, rewards, done, truncated, info = env.step(int(spice_action), pixel_actions)
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Pixels={len(env.live_pixels):2d}, "
                  f"Main reward={rewards[0]:6.2f}, Spice reward={rewards[1]:6.2f}")
        
        if args.render:
            renderer.render()
            time.sleep(0.1)
        
        if done or truncated:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Final state: {len(env.live_pixels)} pixels alive")


def run_per_pixel_demo(args):
    """Run per-pixel AI system demonstration."""
    print("Running Per-Pixel AI Demo")
    print("=" * 40)
    
    ai_system = PerPixelAISystem(env_size=args.size, log_dir="./logs/per_pixel_demo")
    
    if args.train:
        print("Training per-pixel AI system...")
        ai_system.run_continual_learning(total_episodes=args.generations)
    
    print("Running per-pixel AI demo...")
    episode_info = ai_system.run_episode(max_steps=args.steps)
    print(f"Episode completed: {episode_info}")


def run_continual_learning_demo(args):
    """Run continual learning demonstration."""
    print("Running Continual Learning Demo")
    print("=" * 40)
    
    cl_system = ContinualLearningSystem(
        env_kwargs={'H': args.size, 'W': args.size},
        log_dir="./logs/continual_learning_demo"
    )
    
    if args.train:
        print("Training continual learning system...")
        cl_system.run_continual_learning(total_episodes=args.episodes)
    
    print("Running continual learning demo...")
    episode_info = cl_system.run_episode(max_steps=args.steps, render=args.render)
    print(f"Episode completed: {episode_info}")


def run_pygame_demo(args):
    """Run Pygame-based visualization demo."""
    print("Running Pygame Demo")
    print("=" * 40)
    
    try:
        import pygame
    except ImportError:
        print("Pygame not installed. Install with: pip install pygame")
        return
    
    env = PixelLifeEnv(H=args.size, W=args.size)
    renderer = PixelLifeRenderer(env)
    
    obs = env.reset()
    try:
        import pygame
        clock = pygame.time.Clock()
    except ImportError:
        print("Pygame not available for rendering")
        return
    
    for step in range(args.steps):
        # Random actions
        spice_action = env.spice_action_space.sample()
        pixel_actions = {}
        for coord in env.live_pixels:
            action_type = np.random.randint(0, 4)
            direction = np.random.randint(0, 4)
            pixel_actions[coord] = (action_type, direction)
        
        obs, rewards, done, truncated, info = env.step(spice_action, pixel_actions)
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Pixels={len(env.live_pixels):2d}")
        
        renderer.render()
        clock.tick(10)  # 10 FPS
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Final state: {len(env.live_pixels)} pixels alive")


def run_accelerated_demo(args):
    """Run Apple Silicon accelerated demo."""
    if not APPLE_ACCELERATION_AVAILABLE:
        print("❌ Apple Silicon acceleration not available")
        print("Install required packages: pip install torch-mps coremltools")
        return
    
    if pygame is None:
        print("❌ Pygame not available")
        print("Install with: pip install pygame")
        return
    
    print("Running Apple Silicon Accelerated Demo")
    print("=" * 40)
    
    # Create accelerated environment
    env = create_accelerated_env(H=args.size, W=args.size)
    
    # Create accelerated renderer
    renderer = create_renderer(env, window_size=(800, 600), cell_size=8)
    
    obs = env.reset()
    clock = pygame.time.Clock()
    
    for step in range(args.steps):
        # Random actions
        pixel_actions = {}
        for coord in env.live_pixels:
            action_type = np.random.randint(0, 4)
            direction = np.random.randint(0, 4)
            pixel_actions[coord] = (action_type, direction)
        
        obs, rewards, done, truncated, info = env.step(0, pixel_actions)
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Pixels={len(env.live_pixels):2d}")
        
        renderer.render()
        if 'clock' in locals():
            clock.tick(30)  # 30 FPS
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Final state: {len(env.live_pixels)} pixels alive")

def run_benchmark_demo(args):
    """Run performance benchmark demo."""
    if not APPLE_ACCELERATION_AVAILABLE:
        print("❌ Apple Silicon acceleration not available")
        return
    
    print("Running Performance Benchmark Demo")
    print("=" * 40)
    
    # Benchmark environment acceleration
    print("🏃‍♂️ Benchmarking environment acceleration...")
    benchmark_acceleration(env_size=args.size, steps=1000)
    
    # Benchmark rendering
    print("\n🎨 Benchmarking rendering acceleration...")
    benchmark = BenchmarkRenderer(env_size=args.size)
    benchmark.benchmark_rendering(frames=500)

def run_enhanced_demo(args):
    """Run Enhanced Pygame-based visualization demo with zoom and resizable window."""
    print("Running Enhanced Pygame Demo")
    print("=" * 40)
    
    try:
        import pygame
    except ImportError:
        print("Pygame not installed. Install with: pip install pygame")
        return
    
    env = PixelLifeEnv(H=args.size, W=args.size)
    renderer = EnhancedPixelLifeRenderer(
        env=env,
        width=1200,
        height=800,
        initial_zoom=args.initial_zoom
    )
    
    print(f"Enhanced renderer started with {args.initial_zoom}x zoom")
    print("Controls: Mouse wheel zoom, middle drag pan, F fullscreen, R reset view")
    
    # Run the enhanced renderer
    renderer.run_with_env()


def run_training(args):
    """Run full training session."""
    print("Running Full Training Session")
    print("=" * 40)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/training_run_{timestamp}"
    
    print(f"Training log directory: {log_dir}")
    print(f"Environment size: {args.size}x{args.size}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Device: {args.device}")
    
    # Training parameters
    hyperparams = {
        'total_timesteps': args.timesteps,
        'n_envs': args.n_envs,
        'learning_rate': args.learning_rate,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'device': args.device,
        'log_dir': log_dir,
        'no_tensorboard': args.no_tensorboard,
    }
    
    # Use accelerated training if requested
    if hasattr(args, 'accelerated') and args.accelerated and APPLE_ACCELERATION_AVAILABLE:
        print("🚀 Using Apple Silicon accelerated training")
        from accelerated_training import train_accelerated_pixel_life
        # Remove device parameter as it's handled internally by the accelerated training
        accelerated_hyperparams = {k: v for k, v in hyperparams.items() if k != 'device'}
        # Add environment kwargs
        accelerated_hyperparams['env_kwargs'] = {'H': args.size, 'W': args.size, 'max_size': args.size * 2}
        main_model, spice_model = train_accelerated_pixel_life(**accelerated_hyperparams)
        run_dir = log_dir
    else:
        # Run standard training
        main_model, spice_model, run_dir = train_pixel_life(**hyperparams)
    
    print(f"\nTraining complete!")
    print(f"Models saved to: {run_dir}")
    print(f"View logs with: tensorboard --logdir {log_dir}")


def run_evaluation(args):
    """Run model evaluation."""
    print("Running Model Evaluation")
    print("=" * 40)
    
    if not args.model_path:
        print("Error: --model-path is required for evaluation")
        sys.exit(1)
    
    # Load model
    try:
        model = PPO.load(args.model_path)
        print(f"Loaded model from: {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create environment
    env = PixelLifeEnv(H=args.size, W=args.size)
    wrapped_env = PixelLifeWrapper(env, 'main')
    
    # Run evaluation
    total_rewards = []
    for episode in range(args.episodes):
        reset_result = wrapped_env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        episode_reward = 0
        
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            step_result = wrapped_env.step(action)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                truncated = False
            episode_reward += reward
            
            if args.render and step % 10 == 0:
                env.render()
                plt.pause(0.1)
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nEvaluation Results:")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {min(total_rewards):.2f}")
    print(f"Max reward: {max(total_rewards):.2f}")
    
    if args.render:
        plt.show()


def run_info(args):
    """Display system information and available models."""
    print("Pixel Life System Information")
    print("=" * 40)
    
    # Environment info
    print(f"Environment size: {args.size}x{args.size}")
    env = PixelLifeEnv(H=args.size, W=args.size)
    print(f"Action spaces:")
    print(f"  Main agent: {env.action_space}")
    print(f"  Spice agent: {env.spice_action_space}")
    print(f"  Pixel actions: {env.pixel_action_space}")
    print(f"  Observation spaces:")
    print(f"    Main: {env.observation_space}")
    print(f"    Spice: {env.observation_space}")
    
    # Check for saved models
    print(f"\nSaved models:")
    if os.path.exists("./logs"):
        for root, dirs, files in os.walk("./logs"):
            for file in files:
                if file.endswith(".zip"):
                    model_path = os.path.join(root, file)
                    print(f"  {model_path}")
    
    if not os.path.exists("./logs") or not any(f.endswith(".zip") for _, _, files in os.walk("./logs") for f in files):
        print("  No saved models found")
    
    # System info
    print(f"\nSystem information:")
    print(f"  Python version: {sys.version}")
    print(f"  NumPy version: {np.__version__}")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("  PyTorch: Not installed")
    
    try:
        import pygame
        print(f"  Pygame version: {pygame.version.ver}")
    except ImportError:
        print("  Pygame: Not installed")


def run_benchmark(args):
    """Run performance benchmark."""
    print("Running Performance Benchmark")
    print("=" * 40)
    
    env = PixelLifeEnv(H=args.size, W=args.size)
    
    # Warm up
    obs = env.reset()
    for _ in range(100):
        spice_action = env.spice_action_space.sample()
        pixel_actions = {}
        for coord in env.live_pixels:
            pixel_actions[coord] = (np.random.randint(0, 4), np.random.randint(0, 4))
        obs, _, _, _, _ = env.step(spice_action, pixel_actions)
    
    # Benchmark
    start_time = time.time()
    steps = 0
    
    for _ in range(args.steps):
        spice_action = env.spice_action_space.sample()
        pixel_actions = {}
        for coord in env.live_pixels:
            pixel_actions[coord] = (np.random.randint(0, 4), np.random.randint(0, 4))
        obs, _, _, _, _ = env.step(spice_action, pixel_actions)
        steps += 1
    
    end_time = time.time()
    elapsed = end_time - start_time
    steps_per_second = steps / elapsed
    
    print(f"Benchmark Results:")
    print(f"  Environment size: {args.size}x{args.size}")
    print(f"  Steps executed: {steps:,}")
    print(f"  Time elapsed: {elapsed:.2f} seconds")
    print(f"  Steps per second: {steps_per_second:.1f}")
    print(f"  Average pixels per step: {np.mean([len(env.live_pixels) for _ in range(10)]):.1f}")


def run_config(args):
    """Generate or load configuration file."""
    if args.generate:
        config = {
            "environment": {
                "size": 30,
                "max_pixels": 1000,
                "spice_expansion_rate": 0.1
            },
            "training": {
                "total_timesteps": 1000000,
                "n_envs": 4,
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99
            },
            "rendering": {
                "enable": True,
                "fps": 10,
                "window_size": [1200, 800]
            }
        }
        
        with open("pixel_life_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("Generated pixel_life_config.json")
        
    elif args.load and os.path.exists(args.load):
        with open(args.load, "r") as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.load}:")
        print(json.dumps(config, indent=2))
    else:
        print("Use --generate to create a config file or --load <file> to load one")

def run_logs_overview(args):
    from src.logging_tools import PixelLogManager
    log_manager = PixelLogManager()
    log_manager.show_overview()

def run_logs_search(args):
    from src.logging_tools import PixelLogManager
    log_manager = PixelLogManager()
    log_manager.search_logs(args.query, level=args.level, limit=args.limit)

def run_logs_performance(args):
    from src.logging_tools import PixelLogManager
    log_manager = PixelLogManager()
    log_manager.show_performance(args.limit)

def run_logs_errors(args):
    from src.logging_tools import PixelLogManager
    log_manager = PixelLogManager()
    log_manager.show_errors(args.limit)

def run_logs_cleanup(args):
    from src.logging_tools import PixelLogManager
    log_manager = PixelLogManager()
    log_manager.cleanup_logs(args.days, dry_run=not args.execute)


# New CLI command functions
def run_models_list(args):
    """List all registered models."""
    model_manager = ModelManager()
    models = model_manager.list_models(tags=args.tags.split(",") if args.tags else None)
    
    print("📦 Registered Models")
    print("=" * 80)
    print(f"{'ID':<8} {'Name':<20} {'Version':<10} {'Size(MB)':<10} {'Author':<15} {'Tags':<20}")
    print("-" * 80)
    
    for model in models:
        tags_str = ", ".join(model.tags[:3]) if model.tags else ""
        if len(model.tags) > 3:
            tags_str += "..."
        print(f"{model.model_id:<8} {model.name:<20} {model.version:<10} "
              f"{model.file_size_mb:<10.1f} {model.author:<15} {tags_str:<20}")
    
    if not models:
        print("No models registered. Use 'pixel_life models register' to add models.")


def run_models_register(args):
    """Register a new model."""
    model_manager = ModelManager()
    
    try:
        model_id = model_manager.register_model(
            model_path=args.model_path,
            name=args.name,
            description=args.description,
            author=args.author,
            tags=args.tags.split(",") if args.tags else []
        )
        print(f"✅ Model registered successfully with ID: {model_id}")
        print(f"📁 Model stored at: {model_manager.models_dir}")
    except Exception as e:
        print(f"❌ Failed to register model: {e}")


def run_models_info(args):
    """Show detailed information about a model."""
    model_manager = ModelManager()
    model = model_manager.get_model(args.model_id)
    
    if not model:
        print(f"❌ Model {args.model_id} not found")
        return
    
    print(f"📦 Model Information: {model.name}")
    print("=" * 60)
    print(f"ID: {model.model_id}")
    print(f"Name: {model.name}")
    print(f"Version: {model.version}")
    print(f"Description: {model.description}")
    print(f"Author: {model.author}")
    print(f"Created: {model.created_at}")
    print(f"Updated: {model.updated_at}")
    print(f"File size: {model.file_size_mb:.2f} MB")
    print(f"Checksum: {model.checksum}")
    
    if model.tags:
        print(f"Tags: {', '.join(model.tags)}")
    
    if model.performance_metrics:
        print("\n📊 Performance Metrics:")
        for metric, value in model.performance_metrics.items():
            print(f"  {metric}: {value}")
    
    if model.hyperparameters:
        print("\n⚙️  Hyperparameters:")
        for param, value in model.hyperparameters.items():
            print(f"  {param}: {value}")


def run_models_delete(args):
    """Delete a registered model."""
    model_manager = ModelManager()
    
    try:
        model_manager.delete_model(args.model_id)
        print(f"✅ Model {args.model_id} deleted successfully")
    except Exception as e:
        print(f"❌ Failed to delete model: {e}")


def run_experiments_list(args):
    """List all experiments."""
    exp_manager = ExperimentManager()
    experiments = exp_manager.list_experiments(status=args.status, tags=args.tags.split(",") if args.tags else None)
    
    print("🧪 Experiments")
    print("=" * 80)
    print(f"{'ID':<8} {'Name':<25} {'Status':<12} {'Created':<20} {'Tags':<20}")
    print("-" * 80)
    
    for exp in experiments:
        tags_str = ", ".join(exp.tags[:3]) if exp.tags else ""
        if len(exp.tags) > 3:
            tags_str += "..."
        print(f"{exp.experiment_id:<8} {exp.name:<25} {exp.status:<12} "
              f"{exp.created_at[:19]:<20} {tags_str:<20}")
    
    if not experiments:
        print("No experiments found. Use 'pixel_life experiments create' to create experiments.")


def run_experiments_create(args):
    """Create a new experiment."""
    exp_manager = ExperimentManager()
    
    # Parse hyperparameters from command line
    hyperparameters = {}
    if args.hyperparams:
        for param in args.hyperparams:
            if "=" in param:
                key, value = param.split("=", 1)
                try:
                    # Try to convert to appropriate type
                    if value.lower() in ['true', 'false']:
                        hyperparameters[key] = value.lower() == 'true'
                    elif '.' in value:
                        hyperparameters[key] = float(value)
                    else:
                        hyperparameters[key] = int(value)
                except ValueError:
                    hyperparameters[key] = value
    
    try:
        experiment_id = exp_manager.create_experiment(
            name=args.name,
            description=args.description,
            hyperparameters=hyperparameters,
            tags=args.tags.split(",") if args.tags else [],
            parent_experiment=args.parent
        )
        print(f"✅ Experiment created successfully with ID: {experiment_id}")
        print(f"📁 Experiment config stored at: {exp_manager.experiments_dir}")
    except Exception as e:
        print(f"❌ Failed to create experiment: {e}")


def run_experiments_info(args):
    """Show detailed information about an experiment."""
    exp_manager = ExperimentManager()
    exp = exp_manager.get_experiment(args.experiment_id)
    
    if not exp:
        print(f"❌ Experiment {args.experiment_id} not found")
        return
    
    print(f"🧪 Experiment Information: {exp.name}")
    print("=" * 60)
    print(f"ID: {exp.experiment_id}")
    print(f"Name: {exp.name}")
    print(f"Description: {exp.description}")
    print(f"Status: {exp.status}")
    print(f"Created: {exp.created_at}")
    
    if exp.parent_experiment:
        print(f"Parent experiment: {exp.parent_experiment}")
    
    if exp.tags:
        print(f"Tags: {', '.join(exp.tags)}")
    
    if exp.hyperparameters:
        print("\n⚙️  Hyperparameters:")
        for param, value in exp.hyperparameters.items():
            print(f"  {param}: {value}")
    
    if exp.results:
        print("\n📊 Results:")
        for metric, value in exp.results.items():
            print(f"  {metric}: {value}")


def run_monitor_start(args):
    """Start system monitoring."""
    monitor = SystemMonitor()
    monitor.start_monitoring(interval=args.interval)
    print(f"🔍 System monitoring started (interval: {args.interval}s)")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            metrics = monitor.get_current_metrics()
            print(f"\r🖥️  CPU: {metrics.cpu_percent:5.1f}% | "
                  f"💾 RAM: {metrics.memory_percent:5.1f}% | "
                  f"💿 Disk: {metrics.disk_usage_percent:5.1f}% | "
                  f"🔄 Processes: {metrics.active_processes}", end="", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("\n✅ Monitoring stopped")


def run_monitor_status(args):
    """Show current system status."""
    monitor = SystemMonitor()
    metrics = monitor.get_current_metrics()
    
    print("🖥️  System Status")
    print("=" * 50)
    print(f"CPU Usage: {metrics.cpu_percent:.1f}%")
    print(f"Memory Usage: {metrics.memory_percent:.1f}%")
    print(f"Disk Usage: {metrics.disk_usage_percent:.1f}%")
    if metrics.gpu_usage_percent:
        print(f"GPU Usage: {metrics.gpu_usage_percent:.1f}%")
    print(f"Active Processes: {metrics.active_processes}")
    print(f"Timestamp: {metrics.timestamp}")
    
    # System information
    print(f"\n💻 System Information")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Architecture: {platform.machine()}")
    
    # Memory details
    memory = psutil.virtual_memory()
    print(f"\n💾 Memory Details")
    print(f"Total: {memory.total / (1024**3):.1f} GB")
    print(f"Available: {memory.available / (1024**3):.1f} GB")
    print(f"Used: {memory.used / (1024**3):.1f} GB")


def run_visualize_compare(args):
    """Compare multiple models or experiments visually."""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Load data from different sources
    data_sources = []
    
    # Load model performance data
    model_manager = ModelManager()
    models = model_manager.list_models()
    if models:
        model_data = []
        for model in models:
            if model.performance_metrics:
                model_data.append({
                    'name': model.name,
                    'type': 'model',
                    **model.performance_metrics
                })
        if model_data:
            data_sources.append(('Models', pd.DataFrame(model_data)))
    
    # Load experiment results
    exp_manager = ExperimentManager()
    experiments = exp_manager.list_experiments()
    if experiments:
        exp_data = []
        for exp in experiments:
            if exp.results:
                exp_data.append({
                    'name': exp.name,
                    'type': 'experiment',
                    **exp.results
                })
        if exp_data:
            data_sources.append(('Experiments', pd.DataFrame(exp_data)))
    
    if not data_sources:
        print("❌ No data available for comparison")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model and Experiment Comparison', fontsize=16)
    
    plot_idx = 0
    for title, df in data_sources:
        if plot_idx >= 4:
            break
        
        ax = axes[plot_idx // 2, plot_idx % 2]
        
        # Plot different metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols[:3]].plot(kind='bar', ax=ax)
            ax.set_title(f'{title} - Top Metrics')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
        
        plot_idx += 1
    
    plt.tight_layout()
    
    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison saved to {args.save}")
    
    if args.show:
        plt.show()


def run_workflow_batch(args):
    """Run batch processing workflow."""
    print(f"🔄 Starting batch processing: {args.workflow}")
    
    if args.workflow == "train-multiple":
        # Train multiple models with different configurations
        configs = [
            {"size": 20, "timesteps": 50000, "learning_rate": 3e-4},
            {"size": 30, "timesteps": 100000, "learning_rate": 1e-4},
            {"size": 40, "timesteps": 150000, "learning_rate": 5e-4}
        ]
        
        for i, config in enumerate(configs):
            print(f"\n📦 Training model {i+1}/{len(configs)} with config: {config}")
            
            # Create experiment
            exp_manager = ExperimentManager()
            exp_id = exp_manager.create_experiment(
                name=f"batch-train-{i+1}",
                description=f"Batch training run {i+1}",
                hyperparameters=config,
                tags=["batch", "training"]
            )
            
            # Run training
            try:
                env = PixelLifeEnv(H=config["size"], W=config["size"])
                model = PPO("MlpPolicy", env, verbose=0, learning_rate=config["learning_rate"])
                model.learn(total_timesteps=config["timesteps"])
                
                # Save model
                model_path = f"./models/batch_model_{i+1}.zip"
                model.save(model_path)
                
                # Register model
                model_manager = ModelManager()
                model_id = model_manager.register_model(
                    model_path=model_path,
                    name=f"batch-model-{i+1}",
                    description=f"Batch trained model {i+1}",
                    tags=["batch", "trained"]
                )
                
                # Update experiment
                exp_manager.update_experiment(exp_id, status="completed", results={
                    "model_id": model_id,
                    "final_reward": 0.0,  # Would need to evaluate
                    "training_steps": config["timesteps"]
                })
                
                print(f"✅ Model {i+1} completed successfully")
                
            except Exception as e:
                print(f"❌ Model {i+1} failed: {e}")
                exp_manager.update_experiment(exp_id, status="failed", results={"error": str(e)})
    
    elif args.workflow == "evaluate-all":
        # Evaluate all registered models
        model_manager = ModelManager()
        models = model_manager.list_models()
        
        print(f"🔍 Evaluating {len(models)} models...")
        
        for model in models:
            print(f"\n📊 Evaluating {model.name}...")
            
            try:
                # Load and evaluate model
                model_path = f"./models/{model.model_id}_{model.name}.zip"
                if os.path.exists(model_path):
                    env = PixelLifeEnv(H=30, W=30)
                    model_obj = PPO.load(model_path)
                    
                    # Run evaluation
                    obs = env.reset()
                    total_reward = 0
                    for step in range(100):
                        action, _ = model_obj.predict(obs, deterministic=True)
                        obs, reward, done, info = env.step(action)
                        total_reward += reward
                        if done:
                            break
                    
                    # Update model performance
                    model_manager.update_model(
                        model.model_id,
                        performance_metrics={"evaluation_reward": total_reward}
                    )
                    
                    print(f"✅ {model.name}: Reward = {total_reward:.2f}")
                else:
                    print(f"⚠️  Model file not found: {model_path}")
                    
            except Exception as e:
                print(f"❌ Failed to evaluate {model.name}: {e}")
    
    else:
        print(f"❌ Unknown workflow: {args.workflow}")
        print("Available workflows: train-multiple, evaluate-all")


def run_debug_profile(args):
    """Run performance profiling and debugging."""
    import cProfile
    import pstats
    import io
    
    print(f"🔍 Profiling: {args.command}")
    
    # Create profiler
    pr = cProfile.Profile()
    pr.enable()
    
    try:
        # Run the specified command
        if args.command == "basic":
            env = PixelLifeEnv(H=args.size, W=args.size)
            obs = env.reset()
            for _ in range(args.steps):
                spice_action = env.spice_action_space.sample()
                pixel_actions = {}
                for coord in env.live_pixels:
                    action_type = np.random.randint(0, 4)
                    direction = np.random.randint(0, 4)
                    pixel_actions[coord] = (action_type, direction)
                obs, rewards, done, truncated, info = env.step(spice_action, pixel_actions)
                if done:
                    break
        
        elif args.command == "training":
            env = PixelLifeEnv(H=args.size, W=args.size)
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=args.steps)
        
        else:
            print(f"❌ Unknown debug command: {args.command}")
            return
            
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
    finally:
        pr.disable()
    
    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    print("\n📊 Profiling Results (Top 20 functions):")
    print("=" * 60)
    print(s.getvalue())
    
    if args.save:
        ps.dump_stats(args.save)
        print(f"📁 Profile data saved to {args.save}")


def run_help_tutorial(args):
    """Show interactive tutorial."""
    tutorials = {
        "basic": """
🎯 Basic Tutorial: Getting Started with Pixel Life

1. Run a basic demo:
   pixel_life basic --render

2. Try different sizes:
   pixel_life basic --size 50 --steps 300

3. Run with AI agents:
   pixel_life ai --render

4. Check system info:
   pixel_life info

5. Run benchmarks:
   pixel_life benchmark --steps 1000
        """,
        
        "training": """
🎯 Training Tutorial: Training AI Models

1. Start a training session:
   pixel_life train --timesteps 100000

2. Train with custom parameters:
   pixel_life train --size 40 --timesteps 500000 --n-envs 8

3. Monitor training progress:
   pixel_life logs performance

4. Evaluate a trained model:
   pixel_life evaluate --model-path ./logs/your_model.zip

5. Compare models:
   pixel_life visualize compare --show
        """,
        
        "experiments": """
🎯 Experiments Tutorial: Managing Experiments

1. Create an experiment:
   pixel_life experiments create --name "my-experiment" --description "Testing new params"

2. List experiments:
   pixel_life experiments list

3. View experiment details:
   pixel_life experiments info --experiment-id abc12345

4. Create experiment with hyperparameters:
   pixel_life experiments create --name "lr-test" --hyperparams learning_rate=0.001 batch_size=64

5. Run batch workflow:
   pixel_life workflow batch --workflow train-multiple
        """,
        
        "monitoring": """
🎯 Monitoring Tutorial: System Monitoring

1. Check current system status:
   pixel_life monitor status

2. Start real-time monitoring:
   pixel_life monitor start --interval 2

3. Monitor during training:
   pixel_life monitor start &
   pixel_life train --timesteps 100000

4. Profile performance:
   pixel_life debug profile --command basic --steps 1000

5. View system logs:
   pixel_life logs overview
        """
    }
    
    tutorial = tutorials.get(args.topic, "basic")
    print(tutorial)
    
    if args.interactive:
        print("\n❓ Would you like to try this tutorial? (y/n): ", end="")
        response = input().lower().strip()
        if response == 'y':
            print("🚀 Let's get started!")
            # Could implement interactive guided execution here


def main():
    parser = argparse.ArgumentParser(
        prog='pixel_life',
        description='Pixel Life: A 2D Artificial Life Environment',
        epilog='Run "pixel_life <mode> --help" for mode-specific options'
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Available modes')
    
    # Basic demo parser
    basic_parser = subparsers.add_parser('basic', help='Basic environment demonstration')
    basic_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    basic_parser.add_argument('--steps', type=int, default=200, help='Number of steps (default: 200)')
    basic_parser.add_argument('--render', action='store_true', help='Enable rendering')
    if APPLE_ACCELERATION_AVAILABLE:
        basic_parser.add_argument('--accelerated', action='store_true', help='Use Apple Silicon acceleration')
    basic_parser.set_defaults(func=run_basic_demo)
    
    # AI demo parser
    ai_parser = subparsers.add_parser('ai', help='AI agent demonstration')
    ai_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    ai_parser.add_argument('--steps', type=int, default=200, help='Number of steps (default: 200)')
    ai_parser.add_argument('--render', action='store_true', help='Enable rendering')
    if APPLE_ACCELERATION_AVAILABLE:
        ai_parser.add_argument('--accelerated', action='store_true', help='Use Apple Silicon acceleration')
    ai_parser.set_defaults(func=run_ai_demo)
    
    # Per-pixel demo parser
    per_pixel_parser = subparsers.add_parser('per-pixel', help='Per-pixel AI system demonstration')
    per_pixel_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    per_pixel_parser.add_argument('--steps', type=int, default=200, help='Number of steps (default: 200)')
    per_pixel_parser.add_argument('--render', action='store_true', help='Enable rendering')
    per_pixel_parser.add_argument('--train', action='store_true', help='Train the system first')
    per_pixel_parser.add_argument('--generations', type=int, default=5, help='Number of generations (default: 5)')
    per_pixel_parser.add_argument('--steps-per-gen', type=int, default=1000, help='Steps per generation (default: 1000)')
    per_pixel_parser.set_defaults(func=run_per_pixel_demo)
    
    # Continual learning demo parser
    cl_parser = subparsers.add_parser('continual', help='Continual learning demonstration')
    cl_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    cl_parser.add_argument('--steps', type=int, default=200, help='Number of steps (default: 200)')
    cl_parser.add_argument('--render', action='store_true', help='Enable rendering')
    cl_parser.add_argument('--train', action='store_true', help='Train the system first')
    cl_parser.add_argument('--episodes', type=int, default=10, help='Number of episodes (default: 10)')
    cl_parser.add_argument('--steps-per-episode', type=int, default=500, help='Steps per episode (default: 500)')
    cl_parser.set_defaults(func=run_continual_learning_demo)
    
    # Pygame demo parser
    pygame_parser = subparsers.add_parser('pygame', help='Pygame-based visualization')
    pygame_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    pygame_parser.add_argument('--steps', type=int, default=200, help='Number of steps (default: 200)')
    pygame_parser.set_defaults(func=run_pygame_demo)
    
    # Enhanced pygame demo parser
    enhanced_parser = subparsers.add_parser('enhanced', help='Enhanced Pygame visualization with zoom and resizable window')
    enhanced_parser.add_argument('--size', type=int, default=100, help='Environment size (default: 100)')
    enhanced_parser.add_argument('--initial-zoom', type=float, default=0.01, help='Initial zoom level (default: 0.01 = 100x smaller pixels)')
    enhanced_parser.set_defaults(func=run_enhanced_demo)
    
    # Accelerated demo parser (Apple Silicon)
    if APPLE_ACCELERATION_AVAILABLE:
        accelerated_parser = subparsers.add_parser('accelerated', help='Apple Silicon accelerated demo')
        accelerated_parser.add_argument('--size', type=int, default=50, help='Environment size (default: 50)')
        accelerated_parser.add_argument('--steps', type=int, default=200, help='Number of steps (default: 200)')
        accelerated_parser.set_defaults(func=run_accelerated_demo)
        
        # Benchmark demo parser
        benchmark_demo_parser = subparsers.add_parser('benchmark-demo', help='Performance benchmark demo')
        benchmark_demo_parser.add_argument('--size', type=int, default=50, help='Environment size (default: 50)')
        benchmark_demo_parser.set_defaults(func=run_benchmark_demo)
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Full training session')
    train_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    train_parser.add_argument('--timesteps', type=int, default=1000000, help='Total timesteps (default: 1000000)')
    train_parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments (default: 4)')
    train_parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    train_parser.add_argument('--n-steps', type=int, default=2048, help='Steps before update (default: 2048)')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    train_parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs (default: 10)')
    train_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    train_parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu', help='Device (default: cpu)')
    train_parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard logging')
    if APPLE_ACCELERATION_AVAILABLE:
        train_parser.add_argument('--accelerated', action='store_true', help='Use Apple Silicon acceleration')
    train_parser.set_defaults(func=run_training)
    
    # Evaluation parser
    eval_parser = subparsers.add_parser('evaluate', help='Model evaluation')
    eval_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    eval_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    eval_parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes (default: 10)')
    eval_parser.add_argument('--steps', type=int, default=500, help='Steps per episode (default: 500)')
    eval_parser.add_argument('--render', action='store_true', help='Enable rendering')
    eval_parser.set_defaults(func=run_evaluation)
    
    # Info parser
    info_parser = subparsers.add_parser('info', help='Display system information')
    info_parser.add_argument('--size', type=int, default=30, help='Environment size for testing (default: 30)')
    info_parser.set_defaults(func=run_info)
    
    # Benchmark parser
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    benchmark_parser.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    benchmark_parser.add_argument('--steps', type=int, default=10000, help='Number of steps to benchmark (default: 10000)')
    benchmark_parser.set_defaults(func=run_benchmark)
    
    # Config parser
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--generate', action='store_true', help='Generate default config file')
    config_parser.add_argument('--load', type=str, help='Load and display config file')
    config_parser.set_defaults(func=run_config)
    
    # Logs management parser
    logs_parser = subparsers.add_parser('logs', help='Log management and inspection')
    logs_subparsers = logs_parser.add_subparsers(dest='logs_cmd', help='Log commands')

    # Overview
    logs_overview = logs_subparsers.add_parser('overview', help='Show comprehensive log overview')
    logs_overview.set_defaults(func=run_logs_overview)

    # Search
    logs_search = logs_subparsers.add_parser('search', help='Search logs')
    logs_search.add_argument('query', help='Search query')
    logs_search.add_argument('--level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Filter by log level')
    logs_search.add_argument('--limit', type=int, default=20, help='Max results')
    logs_search.set_defaults(func=run_logs_search)

    # Performance
    logs_perf = logs_subparsers.add_parser('performance', help='Show model performance')
    logs_perf.add_argument('--limit', type=int, default=10, help='Number of models to show')
    logs_perf.set_defaults(func=run_logs_performance)

    # Errors
    logs_errors = logs_subparsers.add_parser('errors', help='Show recent errors and warnings')
    logs_errors.add_argument('--limit', type=int, default=20, help='Number of errors to show')
    logs_errors.set_defaults(func=run_logs_errors)

    # Cleanup
    logs_cleanup = logs_subparsers.add_parser('cleanup', help='Clean up old log files')
    logs_cleanup.add_argument('--days', type=int, default=30, help='Days to keep')
    logs_cleanup.add_argument('--execute', action='store_true', help='Actually delete files (dry run by default)')
    logs_cleanup.set_defaults(func=run_logs_cleanup)

    # Models management parser
    models_parser = subparsers.add_parser('models', help='Model management')
    models_subparsers = models_parser.add_subparsers(dest='models_cmd', help='Model commands')

    # List models
    models_list = models_subparsers.add_parser('list', help='List all registered models')
    models_list.add_argument('--tags', type=str, help='Filter by tags (comma-separated)')
    models_list.set_defaults(func=run_models_list)

    # Register model
    models_register = models_subparsers.add_parser('register', help='Register a new model')
    models_register.add_argument('--model-path', type=str, required=True, help='Path to the model file (e.g., .zip)')
    models_register.add_argument('--name', type=str, required=True, help='Model name')
    models_register.add_argument('--description', type=str, help='Model description')
    models_register.add_argument('--author', type=str, help='Model author')
    models_register.add_argument('--tags', type=str, help='Tags for the model (comma-separated)')
    models_register.set_defaults(func=run_models_register)

    # Get model info
    models_info = models_subparsers.add_parser('info', help='Show detailed information about a model')
    models_info.add_argument('model_id', type=str, help='ID of the model to show')
    models_info.set_defaults(func=run_models_info)

    # Delete model
    models_delete = models_subparsers.add_parser('delete', help='Delete a registered model')
    models_delete.add_argument('model_id', type=str, help='ID of the model to delete')
    models_delete.set_defaults(func=run_models_delete)

    # Experiments management parser
    experiments_parser = subparsers.add_parser('experiments', help='Experiment management')
    experiments_subparsers = experiments_parser.add_subparsers(dest='experiments_cmd', help='Experiment commands')

    # List experiments
    experiments_list = experiments_subparsers.add_parser('list', help='List all experiments')
    experiments_list.add_argument('--status', choices=['running', 'completed', 'failed'], help='Filter by status')
    experiments_list.add_argument('--tags', type=str, help='Filter by tags (comma-separated)')
    experiments_list.set_defaults(func=run_experiments_list)

    # Create experiment
    experiments_create = experiments_subparsers.add_parser('create', help='Create a new experiment')
    experiments_create.add_argument('--name', type=str, required=True, help='Experiment name')
    experiments_create.add_argument('--description', type=str, help='Experiment description')
    experiments_create.add_argument('--hyperparams', nargs='*', help='Hyperparameters in key=value format (e.g., learning_rate=0.001 batch_size=64)')
    experiments_create.add_argument('--parent', type=str, help='ID of the parent experiment')
    experiments_create.add_argument('--tags', type=str, help='Tags for the experiment (comma-separated)')
    experiments_create.set_defaults(func=run_experiments_create)

    # Get experiment info
    experiments_info = experiments_subparsers.add_parser('info', help='Show detailed information about an experiment')
    experiments_info.add_argument('experiment_id', type=str, help='ID of the experiment to show')
    experiments_info.set_defaults(func=run_experiments_info)

    # Monitoring parser
    monitor_parser = subparsers.add_parser('monitor', help='System monitoring')
    monitor_subparsers = monitor_parser.add_subparsers(dest='monitor_cmd', help='Monitor commands')

    # Start monitoring
    monitor_start = monitor_subparsers.add_parser('start', help='Start system monitoring')
    monitor_start.add_argument('--interval', type=int, default=1, help='Interval in seconds for metrics (default: 1)')
    monitor_start.set_defaults(func=run_monitor_start)

    # Show status
    monitor_status = monitor_subparsers.add_parser('status', help='Show current system status')
    monitor_status.set_defaults(func=run_monitor_status)

    # Visualization parser
    visualize_parser = subparsers.add_parser('visualize', help='Visualization tools')
    visualize_subparsers = visualize_parser.add_subparsers(dest='visualize_cmd', help='Visualization commands')

    # Compare models/experiments
    visualize_compare = visualize_subparsers.add_parser('compare', help='Compare multiple models or experiments visually')
    visualize_compare.add_argument('--show', action='store_true', help='Show the plot after saving')
    visualize_compare.add_argument('--save', type=str, help='Save the plot to a file (e.g., plot.png)')
    visualize_compare.set_defaults(func=run_visualize_compare)

    # Workflow parser
    workflow_parser = subparsers.add_parser('workflow', help='Run predefined workflows')
    workflow_subparsers = workflow_parser.add_subparsers(dest='workflow', help='Workflow commands')

    # Batch training
    batch_train = workflow_subparsers.add_parser('batch', help='Run a batch training workflow')
    batch_train.add_argument('--workflow', choices=['train-multiple', 'evaluate-all'], required=True, help='Workflow to run')
    batch_train.set_defaults(func=run_workflow_batch)

    # Debug parser
    debug_parser = subparsers.add_parser('debug', help='Performance profiling and debugging')
    debug_subparsers = debug_parser.add_subparsers(dest='command', help='Debug commands')

    # Basic demo for profiling
    debug_basic = debug_subparsers.add_parser('basic', help='Run a basic demo for profiling')
    debug_basic.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    debug_basic.add_argument('--steps', type=int, default=100, help='Number of steps (default: 100)')
    debug_basic.set_defaults(func=run_debug_profile)

    # Training for profiling
    debug_training = debug_subparsers.add_parser('training', help='Run a training session for profiling')
    debug_training.add_argument('--size', type=int, default=30, help='Environment size (default: 30)')
    debug_training.add_argument('--steps', type=int, default=10000, help='Total timesteps (default: 10000)')
    debug_training.set_defaults(func=run_debug_profile)

    # Help/Tutorial parser
    help_parser = subparsers.add_parser('help', help='Show tutorials and help')
    help_subparsers = help_parser.add_subparsers(dest='topic', help='Help topics')

    # Basic help
    help_basic = help_subparsers.add_parser('basic', help='Show basic tutorial')
    help_basic.add_argument('--interactive', action='store_true', help='Run interactive tutorial')
    help_basic.set_defaults(func=run_help_tutorial)

    # Training help
    help_training = help_subparsers.add_parser('training', help='Show training tutorial')
    help_training.add_argument('--interactive', action='store_true', help='Run interactive training tutorial')
    help_training.set_defaults(func=run_help_tutorial)

    # Experiments help
    help_experiments = help_subparsers.add_parser('experiments', help='Show experiments tutorial')
    help_experiments.add_argument('--interactive', action='store_true', help='Run interactive experiments tutorial')
    help_experiments.set_defaults(func=run_help_tutorial)

    # Monitoring help
    help_monitoring = help_subparsers.add_parser('monitoring', help='Show monitoring tutorial')
    help_monitoring.add_argument('--interactive', action='store_true', help='Run interactive monitoring tutorial')
    help_monitoring.set_defaults(func=run_help_tutorial)

    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # Create logs directory if it doesn't exist
    os.makedirs('./logs', exist_ok=True)
    
    # Initialize logging manager for the session
    try:
        from log_manager import LogManager
        log_manager = LogManager('./logs')
        session_id = log_manager.start_request()
        log_manager.log("INFO", f"Starting Pixel Life session: {args.mode}", 
                       module="pixel_life", function="main")
    except ImportError:
        log_manager = None
        session_id = None
    
    # Run the selected mode
    try:
        args.func(args)
        if log_manager:
            log_manager.log("INFO", f"Session completed successfully: {args.mode}", 
                           module="pixel_life", function="main")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if log_manager:
            log_manager.log("WARNING", "Session interrupted by user", 
                           module="pixel_life", function="main")
    except Exception as e:
        print(f"Error: {e}")
        if log_manager:
            log_manager.log("ERROR", f"Session failed: {e}", 
                           module="pixel_life", function="main")
        if args.mode in ['train', 'per-pixel', 'continual']:
            print("This might be due to missing dependencies or insufficient resources.")
            print("Try running with --size 20 for a smaller environment.")


if __name__ == "__main__":
    main() 