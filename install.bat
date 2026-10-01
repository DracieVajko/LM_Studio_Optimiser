@echo off
REM =====================================================================
REM  LM Studio Optimizer - Windows INSTALL
REM  Creates .venv, installs the package, copies .env.example to .env
REM =====================================================================
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
if not exist ".venv" python -m venv .venv

echo [2/3] Installing package...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -e ".[dev]"

echo [3/3] Configuration...
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Created .env from example - adjust if needed.
) else (
    echo .env already exists - kept.
)

echo.
echo Done. Next steps:
echo   1. Start LM Studio with Developer server on 127.0.0.1:1234
echo   2. run-web.bat  (web UI)  or  run-optimizer.bat is not needed - use:
echo      .venv\Scripts\activate.bat  then  python -m lm_optimizer --help
pause
