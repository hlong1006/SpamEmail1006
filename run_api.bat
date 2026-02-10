@echo off
REM Batch file để chạy Flask API với UTF-8 encoding trên Windows
setlocal enabledelayedexpansion

set PYTHONIOENCODING=utf-8

set PORT=%1
if "%PORT%"=="" set PORT=5000

set DEBUG=%2
if "%DEBUG%"=="" set DEBUG=0

if "%1"==""  (
    echo Usage: run_api.bat [PORT] [DEBUG]
    echo.
    echo Examples:
    echo   run_api.bat 5000 0
    echo   run_api.bat 5000 1
    echo.
    echo Default: PORT=5000, DEBUG=0
)

echo Starting API server on http://localhost:%PORT% (DEBUG=%DEBUG%)
python api/app.py

endlocal
