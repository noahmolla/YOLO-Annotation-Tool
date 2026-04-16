#!/bin/bash
# ============================================
# YOLO Annotator - Auto-Setup & Launch Script
# ============================================

echo ""
echo "========================================"
echo "  YOLO Annotator"
echo "========================================"
echo ""

if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
    echo "ERROR: Python is not installed."
    echo ""
    echo "Install Python 3.12 for the smoothest setup:"
    echo "  https://www.python.org/downloads/"
    echo ""
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

if [ ! -f "venv/bin/activate" ]; then
    echo "First time setup detected."
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv venv

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create the virtual environment."
        exit 1
    fi
fi

echo "Activating virtual environment..."
source venv/bin/activate

python -c "import ttkbootstrap" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo ""
    echo "Installing base application requirements..."
    echo ""
    python -m pip install --upgrade pip
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to upgrade pip."
        exit 1
    fi

    python -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to install base requirements."
        echo "Try running manually:"
        echo "  python -m pip install --upgrade pip"
        echo "  python -m pip install -r requirements.txt"
        exit 1
    fi

    echo ""
    echo "Base app installed."
    echo "Optional backends:"
    echo "  .pt models:     python -m pip install -r requirements-pt.txt"
    echo "  .tflite models: python -m pip install -r requirements-tflite.txt"
    echo ""
fi

echo "Starting application..."
echo ""
python main.py

if [ $? -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "ERROR: Application crashed or failed to start."
    echo "Check README.md or INSTALL.md for troubleshooting."
    echo "========================================"
fi
