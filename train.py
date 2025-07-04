#!/usr/bin/env python3
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Pixel Life Training Script')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--env_size', type=int, default=64, help='Environment grid size')
    parser.add_argument('--render', action='store_true', help='Enable rendering during training')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Training configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Environment size: {args.env_size}")
    print(f"  Rendering: {args.render}")
    
    # Import here to avoid crashes if dependencies aren't installed
    try:
        from env import PixelLifeEnv
        env = PixelLifeEnv(size=args.env_size)
        print("Environment created successfully")
        
        if args.render:
            from renderer import BasicRenderer
            renderer = BasicRenderer(env)
            renderer.run()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()