#!/bin/bash

# Pixel Life Installation Script

echo "Installing Pixel Life..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install pixel_life as a command
echo "Installing pixel_life command..."
pip install -e .

echo ""
echo "Installation complete!"
echo ""
echo "You can now use the pixel_life command:"
echo "  pixel_life --help                    # Show all available commands"
echo "  pixel_life basic --render            # Run basic demo"
echo "  pixel_life ai --render               # Run AI demo"
echo "  pixel_life train --timesteps 100000  # Run training"
echo "  pixel_life info                      # Show system information"
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate" 