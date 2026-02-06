#!/bin/bash
# ============================================
# YOLO Annotator - Quick Launch Script
# ============================================

echo "Starting YOLO Annotator..."

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Using system Python."
    echo "TIP: Create a venv with: python3 -m venv venv"
fi

# Run the application
python main.py || python3 main.py

# Check for errors
if [ $? -ne 0 ]; then
    echo ""
    echo "============================================"
    echo "ERROR: Failed to start the application."
    echo ""
    echo "Make sure you have:"
    echo "  1. Python 3.8+ installed"
    echo "  2. Run: pip install -r requirements.txt"
    echo "============================================"
fi
