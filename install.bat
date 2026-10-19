@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0" || exit /b 1
where python >nul 2>&1
if errorlevel 1 goto no_python
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 goto no_python
if exist ".venv\Scripts\python.exe" goto dependencies
python -m venv .venv
if errorlevel 1 exit /b 1
:dependencies
set "REQ=requirements.txt"
if "%~1"=="--cpu" set "REQ=requirements-cpu.txt"
".venv\Scripts\python.exe" -X utf8 -m pip install -r "%REQ%"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -X utf8 -m batchresize --doctor
if errorlevel 1 goto failed
echo Installation and available-engine diagnostics completed.
echo CPU-only diagnostics do not certify GPU availability. Use run.bat --doctor.
exit /b 0
:no_python
echo ERROR: Python 3.10+ is required. Install a supported python.org build.
exit /b 1
:failed
echo ERROR: Installation or diagnostics failed. Nothing was installed globally.
echo For CPU-only installation: install.bat --cpu
exit /b 1
