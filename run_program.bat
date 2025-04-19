@echo off
echo Starting Molecular Network Viewer...

:: Change to the directory containing this batch file
cd /d "%~dp0"

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python first.
    echo You can download Python from: https://www.python.org/downloads/
    pause
    exit /b
)

:: Check if app.py exists
if not exist "app.py" (
    echo Error: app.py not found in the current directory!
    echo Please make sure app.py is in the same folder as this batch file.
    pause
    exit /b
)

:: Check if requirements.txt exists
if not exist "requirements.txt" (
    echo Error: requirements.txt not found in the current directory!
    echo Please make sure requirements.txt is in the same folder as this batch file.
    pause
    exit /b
)

:: Check if required packages are installed
echo Checking and installing required packages...
pip install -r requirements.txt

:: Run the program
python app.py

pause