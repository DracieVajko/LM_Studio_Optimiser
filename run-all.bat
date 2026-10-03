@echo off
REM =====================================================================
REM  LM Studio Optimizer - OPTIMIZE EVERYTHING (overnight)
REM  Small models: full tuning. Large models (>=5GB): fit ceilings.
REM  Results: results\*.md   Logs: logs\
REM =====================================================================
cd /d "%~dp0"

echo [1/4] Checking LM Studio (127.0.0.1:1234)...
python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:1234/api/v1/models', timeout=10); print('LM Studio is running')" 2>nul
if errorlevel 1 (
    echo ERROR: LM Studio is not running or Developer server is off.
    echo        Open LM Studio - Settings - Developer - Start server.
    pause
    exit /b 1
)

echo [2/4] Quick tests...
python -m pytest -q
if errorlevel 1 (
    echo ERROR: tests failed, aborting.
    pause
    exit /b 1
)

echo [3/4] AUTO pipeline - small models up to 6GB, full tuning...
python -m lm_optimizer auto --max-size-gb 6 --max-context 8192
if errorlevel 1 (
    echo WARNING: auto finished with an error, continuing to fit...
)

echo [4/4] FIT ladder - large models, max context ceilings...
python -m lm_optimizer fit

echo.
echo DONE. See results:
echo   - per-model reports: results\*-best.md and results\*-fit.md
echo   - summaries: results\auto-summary-*.md
pause
