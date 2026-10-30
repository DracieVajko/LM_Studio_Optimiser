@echo off
REM =====================================================================
REM  LM Studio Optimizer - CLEANUP LOGS + CACHES ONLY
REM  Deleted: data\logs contents, __pycache__/.pytest_cache/.
REM  .ruff_cache/.mypy_cache.
REM  NOT touched: optimizer.db (runs/benchmarks stay), data\reports,
REM  data\checkpoints, models, presets, results\.
REM  Use full cleanup.bat when you also want runs/benchmarks cleaned.
REM =====================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto venvpy
where python >nul 2>nul
if errorlevel 1 goto nopython
set PY=python
goto havepython

:venvpy
set PY=.venv\Scripts\python.exe
goto havepython

:nopython
echo [ERROR] Neither .venv\Scripts\python.exe nor python on PATH found.
pause
exit /b 1

:havepython
echo This will delete data\logs contents and Python/test caches only.
set /p CONFIRM=Type YES to continue:
if /i not "%CONFIRM%"=="YES" goto cancelled

"%PY%" tools\cleanup.py --yes --logs-only
echo.
echo Done.
pause
exit /b 0

:cancelled
echo Cancelled - nothing was changed.
pause
exit /b 0
