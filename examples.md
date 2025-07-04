# Pixel Life Examples

This file contains examples of how to use the unified Pixel Life command-line interface.

## Quick Examples

### Basic Environment Demo
```bash
# Simple demo with rendering
python pixel_life.py basic --render

# Larger environment, more steps
python pixel_life.py basic --size 50 --steps 500 --render
```

### AI Agent Demo
```bash
# Quick AI demo with training
python pixel_life.py ai --render

# Custom environment size
python pixel_life.py ai --size 40 --steps 300 --render
```

### Per-Pixel AI System
```bash
# Run with training
python pixel_life.py per-pixel --train --generations 5 --render

# Just demo without training
python pixel_life.py per-pixel --steps 200 --render
```

### Continual Learning
```bash
# Run continual learning with training
python pixel_life.py continual --train --episodes 20 --render

# Quick demo
python pixel_life.py continual --steps 150 --render
```

### Pygame Visualization
```bash
# Pygame demo (requires pygame)
python pixel_life.py pygame --size 30 --steps 200
```

## Training Examples

### Quick Training
```bash
# Train for 100k timesteps
python pixel_life.py train --timesteps 100000 --size 30

# Train with more environments
python pixel_life.py train --timesteps 500000 --n-envs 8 --size 40
```

### Advanced Training
```bash
# GPU training with custom parameters
python pixel_life.py train \
  --timesteps 1000000 \
  --n-envs 16 \
  --learning-rate 1e-4 \
  --batch-size 128 \
  --device cuda \
  --size 50
```

### Training Without TensorBoard
```bash
python pixel_life.py train --timesteps 100000 --no-tensorboard
```

## Evaluation Examples

### Evaluate a Trained Model
```bash
# Evaluate with rendering
python pixel_life.py evaluate \
  --model-path ./logs/training_run_20240101_120000/main_agent/final_model \
  --episodes 10 \
  --render

# Quick evaluation
python pixel_life.py evaluate \
  --model-path ./logs/training_run_20240101_120000/main_agent/final_model \
  --episodes 5
```

## Common Use Cases

### Development and Testing
```bash
# Quick test with small environment
python pixel_life.py basic --size 20 --steps 100

# Test AI functionality
python pixel_life.py ai --size 25 --steps 150
```

### Research and Experimentation
```bash
# Compare different AI approaches
python pixel_life.py per-pixel --train --generations 10 --steps 500
python pixel_life.py continual --train --episodes 50 --steps 500

# Long-term training
python pixel_life.py train --timesteps 5000000 --n-envs 32 --size 60
```

### Visualization and Demos
```bash
# Matplotlib rendering
python pixel_life.py basic --render --steps 300

# Pygame rendering
python pixel_life.py pygame --steps 500

# AI visualization
python pixel_life.py ai --render --steps 400
```

## Performance Tips

### For Faster Training
```bash
# Use more parallel environments
python pixel_life.py train --n-envs 16 --timesteps 1000000

# Use GPU if available
python pixel_life.py train --device cuda --n-envs 8

# Smaller environment for quick experiments
python pixel_life.py train --size 25 --timesteps 100000
```

### For Better Visualization
```bash
# Slower rendering for better observation
python pixel_life.py basic --render --steps 200

# Larger environment for more interesting patterns
python pixel_life.py basic --size 60 --steps 1000 --render
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--n-envs` or `--size`
   ```bash
   python pixel_life.py train --n-envs 2 --size 20
   ```

2. **Slow Training**: Use smaller environment or fewer environments
   ```bash
   python pixel_life.py train --size 25 --n-envs 4
   ```

3. **Pygame Not Working**: Install pygame or use matplotlib rendering
   ```bash
   pip install pygame
   # or use
   python pixel_life.py basic --render
   ```

4. **CUDA Issues**: Use CPU instead
   ```bash
   python pixel_life.py train --device cpu
   ```

### Getting Help
```bash
# General help
python pixel_life.py --help

# Mode-specific help
python pixel_life.py train --help
python pixel_life.py per-pixel --help
``` 