@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ===========================================================================
REM  batchresize launcher (Windows)
REM
REM    run.bat                 -> process .\input into .\output
REM    run.bat <arguments>     -> same, arguments passed to the CLI
REM
REM  Folders are fixed: put photos into "input", results appear in "output".
REM  Prompts are ASCII-only on purpose: .bat files are read with the console
REM  OEM codepage, so non-ASCII text would show as garbage.
REM ===========================================================================

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

if not "%~1"=="" (
    "%PY%" -m batchresize %*
    exit /b %errorlevel%
)

if not exist "input\" (
    echo.
    echo ERROR: folder "input" not found next to run.bat.
    echo Create it and put your photos inside.
    echo.
    pause
    exit /b 1
)

:menu
cls
echo ======================================================================
echo   batchresize - GPU batch image resizer
echo ======================================================================
echo   Input : %CD%\input
echo   Output: %CD%\output
echo ----------------------------------------------------------------------
echo     1  Light      q=90  1024 px, 8 threads
echo     2  Medium     q=80  1024 px, 8 threads  (recommended)
echo     3  Strong     q=65  1024 px, 8 threads
echo     4  Extreme    q=50  1024 px, 8 threads
echo     5  Thumbnails q=65  256 px, cover crop
echo     6  Custom     choose quality, size, threads, backend
echo     7  Benchmark  GPU vs CPU
echo     8  Environment check
echo     9  Desktop UI (all settings, presets, live log)
echo     0  Exit
echo ======================================================================
choice /c 1234567890 /n /m "Select: "
if errorlevel 10 goto :eof
if errorlevel 9 goto gui
if errorlevel 8 goto doctor
if errorlevel 7 goto bench
if errorlevel 6 goto custom
if errorlevel 5 goto thumbs
if errorlevel 4 goto extreme
if errorlevel 3 goto strong
if errorlevel 2 goto medium
if errorlevel 1 goto light


:light
set "CMD=--quality light"
goto ask_threads


:medium
set "CMD=--quality medium"
goto ask_threads


:strong
set "CMD=--quality strong"
goto ask_threads


:extreme
set "CMD=--quality extreme"
goto ask_threads


:thumbs
set "CMD=--size thumb_256 --mode cover --quality strong"
goto ask_threads


:ask_threads
call :num WK "Threads (1-64)" 8
set "CMD=!CMD! --workers !WK!"
goto run_it


:custom
set "CMD="
call :pick QP "Compression preset" "medium" "light medium strong extreme"
set "CMD=!CMD! --quality !QP!"
call :pick SZC "Target size" "web_1024" "thumb_256 small_512 web_1024 large_1600 custom"
if "!SZC!"=="custom" (
    call :num CW "Max width" 1600
    call :num CH "Max height" 1600
    set "CMD=!CMD! --size custom --width !CW! --height !CH!"
) else (
    set "CMD=!CMD! --size !SZC!"
)
call :pick MODE "Fit mode" "contain" "contain cover"
set "CMD=!CMD! --mode !MODE!"
call :num WK "Threads (1-64)" 8
set "CMD=!CMD! --workers !WK!"
call :pick BKND "Backend" "auto" "auto gpu cpu"
set "CMD=!CMD! --backend !BKND!"
call :num BSZ "GPU batch size" 8
set "CMD=!CMD! --batch-size !BSZ!"
goto run_it


:bench
set "CMD=--benchmark --size web_1024 --quality medium"
call :num REP "Repeat runs per backend" 3
set "CMD=!CMD! --benchmark-repeats !REP!"
goto run_it


:doctor
"%PY%" -m batchresize --doctor
echo.
pause
goto menu


:gui
"%PY%" -m batchresize --gui
if errorlevel 1 (
    echo.
    echo The GUI needs tkinter. Install a python.org build of Python
    echo or use the CLI menu instead.
    pause
)
goto menu


:run_it
echo.
echo ----------------------------------------------------------------------
echo  Running:
echo      "%PY%" -m batchresize !CMD!
echo ----------------------------------------------------------------------
echo   IN  = %CD%\input
echo   OUT = %CD%\output
choice /c YNQ /n /m "Run now? [Y]es / back to me[N]u / [Q]uit"
if errorlevel 3 goto :eof
if errorlevel 2 goto menu
"%PY%" -m batchresize !CMD!
set "RC=%errorlevel%"
echo.
echo Exit code: %RC%
pause
goto menu


REM ===========================================================================
REM  helpers
REM ===========================================================================

REM :pick VAR "prompt" "default" "item1 item2 ..."  (items without spaces)
:pick
setlocal EnableDelayedExpansion
set "PROMPT=%~2"
set "DEF=%~3"
set "ITEMS=%~4"
set "N=0"
for %%I in (%ITEMS%) do (
    set /a N+=1
    echo     !N!  %%I
)
set /p "ANS=!PROMPT! (1..!N!, default !DEF!): "
set "CHOSEN=!DEF!"
if not "!ANS!"=="" (
    set "IDX=0"
    for %%I in (%ITEMS%) do (
        set /a IDX+=1
        if "!ANS!"=="!IDX!" set "CHOSEN=%%I"
    )
)
endlocal & set "%~1=%CHOSEN%"
goto :eof


REM :num VAR "prompt" default
:num
set "ANS="
set /p "ANS=%~2 (default %~3): "
if "%ANS%"=="" goto num_use_default
echo %ANS%| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 goto num_bad
set "%~1=%ANS%"
goto :eof
:num_use_default
set "%~1=%~3"
goto :eof
:num_bad
echo   Not a number - using default %~3
set "%~1=%~3"
goto :eof
