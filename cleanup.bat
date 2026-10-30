@echo off
REM =====================================================================
REM  LM Studio Optimizer - CLEANUP
REM  Archives runs + benchmarks (DB snapshot to data\archive\), then
REM  deletes: run/benchmark DB rows, data\logs contents, data\reports,
REM  data\checkpoints, Python/test caches.
REM  NEVER touches: LM Studio models, data\presets, results\ reports,
REM  the tuned presets in the DB, the archive files themselves.
REM  Refuses to run while an optimizer run holds the job lock.
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
echo This will:
echo   1. archive: optimizer.db snapshot + reports/ + checkpoints/ into
echo      data\archive\cleanup_TIMESTAMP
echo   2. delete:  all run/benchmark rows from optimizer.db
echo   3. delete:  data\logs\ contents
echo   4. delete:  __pycache__ / .pytest_cache / .ruff_cache / .mypy_cache
echo Kept: LM Studio models, data\presets, results\ reports,
echo       tuned presets and settings in the DB, archive files.
echo.
echo Inspect first:  "%PY%" tools\cleanup.py --dry-run
echo.
set /p CONFIRM=Type YES to continue:
if /i not "%CONFIRM%"=="YES" goto cancelled

"%PY%" tools\cleanup.py --yes
set RC=%errorlevel%
if "%RC%"=="1" goto refused
if "%RC%"=="2" goto needdryrun

echo.
echo Cleanup finished.
pause
exit /b 0

:needdryrun
echo [HINT] Missing parameter - run  "%PY%" tools\cleanup.py --dry-run  first.
pause
exit /b 2

:refused
echo [REFUSED] An optimizer run is active. Nothing was changed.
pause
exit /b 1

:cancelled
echo Cancelled - nothing was changed.
pause
exit /b 0
