@echo off
REM ============================================
REM YOLO Image Triage Sorter - Auto-Setup
REM ============================================
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON_EXE="
set "REG_PYTHON_EXE="
set "REQ_FILE=requirements.txt"
set "PYTHON_MM="
set "ENTRY_POINT=pre_annotation_sorter\main.py"

for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.13\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if not defined REG_PYTHON_EXE for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.12\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if not defined REG_PYTHON_EXE for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.11\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if not defined REG_PYTHON_EXE for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.10\InstallPath" /v ExecutablePath 2^>nul ^| find "ExecutablePath"') do set "REG_PYTHON_EXE=%%B"
if defined REG_PYTHON_EXE set "PYTHON_EXE=%REG_PYTHON_EXE%"

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
echo   YOLO Image Triage Sorter
echo ========================================
echo.

if not defined PYTHON_EXE (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Install Python 3.10, 3.11, 3.12, or 3.13 first:
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

for /f %%V in ('"%PYTHON_EXE%" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"' ) do set "PYTHON_MM=%%V"
if "%PYTHON_MM%"=="3.13" set "REQ_FILE=requirements-py313.txt"

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

echo Activating virtual environment...
call venv\Scripts\activate.bat

python -c "import ttkbootstrap" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Installing required packages...
    echo.
    if "%REQ_FILE%"=="requirements-py313.txt" (
        echo Python 3.13 detected. Installing the compatible package set.
        echo.
    )
    pip install -r "%REQ_FILE%"
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install packages!
        pause
        exit /b 1
    )
    echo.
    echo Packages installed successfully!
    echo.
)

echo Starting application...
echo.
python "%ENTRY_POINT%"

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Application crashed or failed to start.
    echo ========================================
    pause
)
