@echo off
REM ============================================
REM YOLO Annotator - Quick Launch Script
REM ============================================

echo Starting YOLO Annotator...

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
    echo TIP: Create a venv with: python -m venv venv
)

REM Run the application
python main.py

REM If python fails, try python3
if errorlevel 1 (
    echo Trying python3...
    python3 main.py
)

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo ============================================
    echo ERROR: Failed to start the application.
    echo.
    echo Make sure you have:
    echo   1. Python 3.8+ installed
    echo   2. Run: pip install -r requirements.txt
    echo ============================================
    pause
)
