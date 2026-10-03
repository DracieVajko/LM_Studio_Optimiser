@echo off
REM =====================================================================
REM  LM Studio Optimizer - OPTIMIZE ONE MODEL
REM  Usage:   run-model.bat <model-key> [max-context] [profile]
REM  Example: run-model.bat qwen3.5-4b 16384
REM            run-model.bat qwen3.5-4b 8192 context
REM =====================================================================
cd /d "%~dp0"

if "%~1"=="" (
    echo Usage: run-model.bat MODEL [MAX-CONTEXT] [PROFILE]
    echo         run-model.bat ALL [MAX-CONTEXT] [PROFILE]   = all models one by one
    echo.
    echo Model list:
    python -m lm_optimizer models
    pause
    exit /b 1
)

if /i "%~1"=="ALL" goto run_all
set MODEL=%~1
if "%~2"=="" (set MAXCTX=8192) else (set MAXCTX=%~2)
if "%~3"=="" (set PROFIL=balanced) else (set PROFIL=%~3)

echo Model: %MODEL%  max-context: %MAXCTX%  profile: %PROFIL%
python -m lm_optimizer auto %MODEL% --max-context %MAXCTX% --profile %PROFIL%
echo.
echo Done. Result: results\%MODEL%-best.md
pause
goto :eof

:run_all
if "%~2"=="" (set MAXCTX=8192) else (set MAXCTX=%~2)
if "%~3"=="" (set PROFIL=balanced) else (set PROFIL=%~3)
echo All models one by one:  max-context: %MAXCTX%  profile: %PROFIL%
python -m lm_optimizer auto --max-context %MAXCTX% --profile %PROFIL%
echo.
echo Done. Results: results\campaign\ + individual -best.md
pause
