@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo === batchresize: install ===
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.10+ not found in PATH.
    echo Install it from https://www.python.org/downloads/windows/ and re-run.
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.10 or newer is required.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create the virtual environment.
        exit /b 1
    )
)

echo Installing dependencies ^(this downloads ~250 MB the first time^) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: dependency installation failed.
    echo If you have no NVIDIA GPU you can still use the CPU backend:
    echo     .venv\Scripts\python.exe -m pip install numpy pillow opencv-python-headless piexif pytest
    exit /b 1
)

echo.
echo === environment check ===
".venv\Scripts\python.exe" -m batchresize --doctor
echo.
echo Done. Try:
echo     run.bat --doctor
echo     run.bat -i "D:\Photos" -o "D:\Photos_Resized" --size web_1024 --quality balanced
endlocal
