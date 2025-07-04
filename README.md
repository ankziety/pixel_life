# Pixel Life

A simple pixel-based life simulation with basic rendering.

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the training script:
```bash
python train.py --help
python train.py --render  # Run with visual rendering
```

### Run renderer only:
```bash
python renderer.py
```

### Test the setup:
```bash
python test_basic.py
```

## Controls

When renderer is running:
- **SPACE** - Pause/Resume simulation
- **ESC** - Quit

## Features

- Simple cellular automata (Conway's Game of Life rules)
- Basic pygame renderer showing white pixels on black background
- Minimal dependencies (numpy, pygame)
- Clean argument parsing that doesn't crash