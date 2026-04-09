@echo off
REM ============================================
REM YOLO Annotator - Auto-Setup & Launch Script
REM ============================================
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON_EXE="
set "REG_PYTHON_EXE="
set "REQ_FILE=requirements.txt"
set "PYTHON_MM="

for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.13\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if not defined REG_PYTHON_EXE for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.12\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if not defined REG_PYTHON_EXE for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.11\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if not defined REG_PYTHON_EXE for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.10\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if defined REG_PYTHON_EXE set "PYTHON_EXE=%REG_PYTHON_EXE%"

REM Prefer a real Python install over the Microsoft Store alias
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python313\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python310\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python310\python.exe"
if not defined PYTHON_EXE if exist "C:\Program Files\Python313\python.exe" set "PYTHON_EXE=C:\Program Files\Python313\python.exe"
if not defined PYTHON_EXE if exist "C:\Program Files\Python312\python.exe" set "PYTHON_EXE=C:\Program Files\Python312\python.exe"
if not defined PYTHON_EXE if exist "C:\Program Files\Python311\python.exe" set "PYTHON_EXE=C:\Program Files\Python311\python.exe"
if not defined PYTHON_EXE if exist "C:\Program Files\Python310\python.exe" set "PYTHON_EXE=C:\Program Files\Python310\python.exe"
if not defined PYTHON_EXE (
    for %%P in (python.exe) do (
        if /I not "%%~$PATH:P"=="%LocalAppData%\Microsoft\WindowsApps\python.exe" if not "%%~$PATH:P"=="" set "PYTHON_EXE=%%~$PATH:P"
    )
)

echo.
echo ========================================
echo   YOLO Annotator
echo ========================================
echo.

REM Check if Python is installed
if not defined PYTHON_EXE (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.10, 3.11, 3.12, or 3.13 from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Found Python, but it could not be started:
    echo   %PYTHON_EXE%
    echo.
    pause
    exit /b 1
)

for /f %%V in ('"%PYTHON_EXE%" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"' ) do set "PYTHON_MM=%%V"
if "%PYTHON_MM%"=="3.13" set "REQ_FILE=requirements-py313.txt"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo First time setup detected!
    echo.
    echo Creating virtual environment...
    "%PYTHON_EXE%" -m venv venv
    
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
    if "%REQ_FILE%"=="requirements-py313.txt" (
        echo Python 3.13 detected. Installing the compatible package set.
        echo TFLite/TensorFlow support will be skipped, but PyTorch .pt models will work.
        echo.
    )
    pip install -r "%REQ_FILE%"
    
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install packages!
        echo Try running manually: pip install -r "%REQ_FILE%"
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
