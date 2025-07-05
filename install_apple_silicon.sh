#!/bin/bash

# Apple Silicon Installation Script for Pixel Life
# This script installs all necessary packages for Apple Silicon acceleration

set -e

echo "🚀 Installing Pixel Life for Apple Silicon Macs..."
echo "=================================================="

# Check if we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ This script is designed for Apple Silicon Macs (arm64)"
    echo "   Detected architecture: $(uname -m)"
    exit 1
fi

# Check macOS version
MACOS_VERSION=$(sw_vers -productVersion)
echo "📱 macOS Version: $MACOS_VERSION"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    echo "   You can download it from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "🐍 Python Version: $PYTHON_VERSION"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install system dependencies
echo "🔧 Installing system dependencies..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# Install system packages
echo "📚 Installing system packages..."
brew install cmake pkg-config

# Create virtual environment
echo "🏗️  Creating virtual environment..."
python3 -m venv pixel_life_env
source pixel_life_env/bin/activate

# Upgrade pip in virtual environment
pip install --upgrade pip

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install numpy matplotlib gymnasium stable-baselines3 numba tensorboard psutil pygame

# Install PyTorch with MPS support
echo "🔥 Installing PyTorch with Apple Silicon support..."
pip install torch torchvision torchaudio

# Install Apple Silicon specific packages
echo "🍎 Installing Apple Silicon acceleration packages..."
pip install coremltools

# Install optional packages for advanced features
echo "🔬 Installing optional packages..."
pip install jupyterlab ipywidgets

# Install development tools
echo "🛠️  Installing development tools..."
pip install black flake8 pytest

# Test PyTorch MPS
echo "🧪 Testing PyTorch MPS support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.randn(3, 3, device=device)
    print(f'MPS test successful: {x.device}')
else:
    print('MPS not available - using CPU')
"

# Test Numba
echo "🧪 Testing Numba..."
python3 -c "
import numba
print(f'Numba version: {numba.__version__}')
from numba import jit
@jit(nopython=True)
def test_func(x):
    return x * 2
result = test_func(5)
print(f'Numba test successful: {result}')
"

# Create activation script
echo "📝 Creating activation script..."
cat > activate_pixel_life.sh << 'EOF'
#!/bin/bash
echo "🚀 Activating Pixel Life environment..."
source pixel_life_env/bin/activate
echo "✅ Environment activated!"
echo "   Run 'python pixel_life.py --help' to see available commands"
echo "   Run 'python pixel_life.py accelerated --help' for Apple Silicon acceleration"
EOF

chmod +x activate_pixel_life.sh

# Create quick start script
echo "📝 Creating quick start script..."
cat > quick_start.sh << 'EOF'
#!/bin/bash
echo "🚀 Quick Start for Pixel Life on Apple Silicon"
echo "=============================================="
echo ""
echo "1. Activate environment:"
echo "   source activate_pixel_life.sh"
echo ""
echo "2. Run basic demo:"
echo "   python pixel_life.py basic --size 50 --steps 200 --render"
echo ""
echo "3. Run accelerated demo (Apple Silicon):"
echo "   python pixel_life.py accelerated --size 50 --steps 200"
echo ""
echo "4. Run performance benchmark:"
echo "   python pixel_life.py benchmark-demo --size 50"
echo ""
echo "5. Train AI models:"
echo "   python pixel_life.py train --timesteps 100000 --device cpu"
echo ""
echo "For more options, run: python pixel_life.py --help"
EOF

chmod +x quick_start.sh

echo ""
echo "✅ Installation completed successfully!"
echo "======================================"
echo ""
echo "🎯 Next steps:"
echo "1. Activate the environment:"
echo "   source activate_pixel_life.sh"
echo ""
echo "2. Run the quick start guide:"
echo "   ./quick_start.sh"
echo ""
echo "3. Test Apple Silicon acceleration:"
echo "   python pixel_life.py accelerated --size 50 --steps 200"
echo ""
echo "4. Run performance benchmarks:"
echo "   python pixel_life.py benchmark-demo --size 50"
echo ""
echo "📚 Available demos:"
echo "   basic      - Basic environment demo"
echo "   ai         - AI agent demo"
echo "   per-pixel  - Per-pixel AI system"
echo "   continual  - Continual learning"
echo "   pygame     - Pygame visualization"
echo "   enhanced   - Enhanced pygame with zoom"
echo "   accelerated - Apple Silicon accelerated (NEW!)"
echo "   benchmark-demo - Performance benchmarks (NEW!)"
echo ""
echo "🔧 Development:"
echo "   python pixel_life.py train --help"
echo "   python pixel_life.py evaluate --help"
echo "   python pixel_life.py info"
echo ""
echo "🚀 Happy coding with Apple Silicon acceleration!" 