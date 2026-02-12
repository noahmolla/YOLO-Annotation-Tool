#!/bin/bash
# ============================================
# YOLO Annotator - Auto-Setup & Launch Script
# ============================================

echo ""
echo "========================================"
echo "  YOLO Annotator"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed!"
    echo ""
    echo "Please install Python 3.8+ from:"
    echo "  https://www.python.org/downloads/"
    echo ""
    echo "Or use your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "  macOS: brew install python3"
    echo ""
    exit 1
fi

# Determine python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "First time setup detected!"
    echo ""
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment!"
        echo "Try: sudo apt install python3-venv"
        exit 1
    fi
    
    echo "Virtual environment created."
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if packages are installed
python -c "import ttkbootstrap" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "Installing required packages..."
    echo "This may take a few minutes on first run."
    echo ""
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to install packages!"
        echo "Try running manually: pip install -r requirements.txt"
        exit 1
    fi
    
    echo ""
    echo "Packages installed successfully!"
    echo ""
fi

# Run the application
echo "Starting application..."
echo ""
python main.py

# Check for errors
if [ $? -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "ERROR: Application crashed or failed to start."
    echo ""
    echo "Try these fixes:"
    echo "  1. Delete the 'venv' folder and run this script again"
    echo "  2. Check README.md for troubleshooting"
    echo "========================================"
fi
