@echo off
REM Batch file để chạy main.py với UTF-8 encoding trên Windows
setlocal enabledelayedexpansion

set PYTHONIOENCODING=utf-8

if "%1"=="" (
    echo Usage: run_main.bat [options]
    echo.
    echo Examples:
    echo   run_main.bat --data data/raw/spam_or_not_spam.csv
    echo   run_main.bat --skip-preprocessing
    echo.
    python main.py --help
) else (
    python main.py %*
)

endlocal
pause
