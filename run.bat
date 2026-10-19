@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0" || exit /b 1
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
if "%~1"=="" goto menu
"%PY%" -X utf8 -m batchresize %*
exit /b %errorlevel%
:menu
"%PY%" -X utf8 -m batchresize.launcher
exit /b %errorlevel%
