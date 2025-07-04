# Pixel Life Environment - Fixes and Usage

## What Was Fixed

### 1. ✅ Fixed `python train.py --help` Command
**Problem**: The train.py script had no command line argument parsing, so `--help` would just run the training instead of showing help.

**Solution**: 
- Added `argparse` module with comprehensive command line options
- Added `--help` support with detailed descriptions
- Added options like `--no-tensorboard`, `--cpu`, `--total-timesteps`, etc.

**Usage**:
```bash
source pixel_life_env/bin/activate
python train.py --help
```

### 2. ✅ Fixed Missing Dependencies
**Problem**: Tensorboard was missing despite being in requirements.txt

**Solution**:
- Installed tensorboard in the virtual environment
- Added `--no-tensorboard` option to disable it if needed
- Updated requirements.txt to use modern `gymnasium` instead of deprecated `gym`

### 3. ✅ Created Basic Pixel Renderer
**Problem**: No simple renderer for visualization

**Solution**: Created a pygame-based renderer with:
- **White pixels on black background** (as requested)
- **Simple controls**: SPACE (pause), ESC/Q (quit)
- Click to toggle pixels
- R for random pixels, C to clear
- Grid-based display with configurable size

**Files Created**:
- `basic_renderer.py` - Main renderer classes
- `demo_renderer.py` - Demo with example patterns  
- `test_renderer.py` - Headless testing

### 4. ✅ Virtual Environment Setup
**Problem**: Needed to ensure .venv usage

**Solution**: 
- Virtual environment is already at `pixel_life_env/` 
- All commands use `source pixel_life_env/bin/activate`
- Updated requirements to include pygame

## Usage Instructions

### 1. Activate Virtual Environment
```bash
source pixel_life_env/bin/activate
```

### 2. Run Basic Renderer (Interactive)
```bash
# Simple demo
python basic_renderer.py

# Demo with patterns
python demo_renderer.py
```

**Controls**:
- `SPACE` - Pause/Unpause
- `ESC` or `Q` - Quit  
- `Click` - Toggle pixels
- `R` - Random pixels
- `C` - Clear all pixels

### 3. Test Renderer (Headless)
```bash
python test_renderer.py
```

### 4. Training with Fixed Command Line
```bash
# Show help
python train.py --help

# Quick test run (no tensorboard)
python train.py --total-timesteps 1000 --no-tensorboard --cpu

# Full training
python train.py --total-timesteps 100000 --n-envs 4
```

### 5. Install Missing Dependencies (if needed)
```bash
pip install -r requirements.txt
```

## Renderer Features

The basic renderer provides exactly what was requested:

✅ **White pixels in a window** - Simple grid of white squares on black background
✅ **Stop and pause controls** - ESC to stop, SPACE to pause
✅ **Super basic** - No complex UI, just pixels and minimal controls
✅ **Self-rolled** - Custom pygame implementation, not a heavy framework

### Renderer Classes

1. **BasicPixelRenderer** - Standalone pixel display
2. **PixelLifeRenderer** - Can integrate with environment (extends Basic)

### Example Usage in Code
```python
from basic_renderer import BasicPixelRenderer

# Create renderer
renderer = BasicPixelRenderer(width=800, height=600, grid_size=20)

# Set some pixels
renderer.set_pixel(10, 10, True)
renderer.set_pixels([(5,5), (6,6), (7,7)], True)

# Run interactive display
renderer.run()
```

## Technical Details

- **Graphics**: pygame for cross-platform compatibility
- **Grid**: Boolean numpy array (True = white pixel, False = black)
- **Performance**: 30 FPS for smooth interaction
- **Size**: Configurable grid size and window dimensions
- **Headless**: Can run tests without display using SDL dummy driver

## Error Resolution

All major issues have been resolved:
- ✅ argparse import and command line parsing
- ✅ tensorboard dependency issues  
- ✅ Virtual environment activation
- ✅ Basic renderer implementation
- ✅ Proper help documentation

The system now works with the virtual environment and provides the simple white pixel renderer as requested.