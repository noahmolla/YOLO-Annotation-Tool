@echo off
REM ============================================
REM YOLO Annotator - Auto-Setup & Launch Script
REM ============================================

echo.
echo ========================================
echo   YOLO Annotator
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.8+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo First time setup detected!
    echo.
    echo Creating virtual environment...
    python -m venv venv
    
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    
    echo Virtual environment created.
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if packages are installed (check for ttkbootstrap as indicator)
python -c "import ttkbootstrap" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Installing required packages...
    echo This may take a few minutes on first run.
    echo.
    pip install -r requirements.txt
    
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install packages!
        echo Try running manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
    
    echo.
    echo Packages installed successfully!
    echo.
)

REM Run the application
echo Starting application...
echo.
python main.py

REM If there was an error, show message
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Application crashed or failed to start.
    echo.
    echo Try these fixes:
    echo   1. Delete the 'venv' folder and run this script again
    echo   2. Check README.md for troubleshooting
    echo ========================================
    pause
)
