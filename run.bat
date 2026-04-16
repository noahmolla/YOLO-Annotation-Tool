@echo off
REM ============================================
REM YOLO Annotator - Auto-Setup & Launch Script
REM ============================================
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON_EXE="
set "REG_PYTHON_EXE="

for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.14\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if not defined REG_PYTHON_EXE for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.13\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if not defined REG_PYTHON_EXE for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.12\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if not defined REG_PYTHON_EXE for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.11\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if not defined REG_PYTHON_EXE for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.10\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if defined REG_PYTHON_EXE set "PYTHON_EXE=%REG_PYTHON_EXE%"

if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python314\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python314\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python313\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python310\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python310\python.exe"
if not defined PYTHON_EXE if exist "C:\Program Files\Python314\python.exe" set "PYTHON_EXE=C:\Program Files\Python314\python.exe"
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

if not defined PYTHON_EXE (
    echo ERROR: Python is not installed or not in PATH.
    echo.
    echo Install Python 3.12 for the smoothest setup:
    echo   https://www.python.org/downloads/
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

if not exist "venv\Scripts\activate.bat" (
    echo First time setup detected!
    echo.
    echo Creating virtual environment...
    "%PYTHON_EXE%" -m venv venv

    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )

    echo Virtual environment created.
    echo.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

python -c "import ttkbootstrap" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Installing base application requirements...
    echo.
    python -m pip install --upgrade pip
    if errorlevel 1 (
        echo ERROR: Failed to upgrade pip.
        pause
        exit /b 1
    )

    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install base requirements.
        echo Try running manually:
        echo   python -m pip install --upgrade pip
        echo   python -m pip install -r requirements.txt
        pause
        exit /b 1
    )

    echo.
    echo Base app installed.
    echo Optional backends:
    echo   .pt models:     python -m pip install -r requirements-pt.txt
    echo   .tflite models: python -m pip install -r requirements-tflite.txt
    echo.
)

echo Starting application...
echo.
python main.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Application crashed or failed to start.
    echo Check README.md or INSTALL.md for troubleshooting.
    echo ========================================
    pause
)
