@echo off
REM =====================================================================
REM  LM Studio Optimizer - OPTIMALIZUJ JEDEN MODEL
REM  Pouzitie:  run-model.bat <model-key> [max-context] [profil]
REM  Priklad:   run-model.bat qwen3.5-4b 16384
REM              run-model.bat qwen3.5-4b 8192 context
REM =====================================================================
cd /d "%~dp0"

if "%~1"=="" (
    echo Pouzitie: run-model.bat MODEL [MAX-CONTEXT] [PROFIL]
    echo          run-model.bat ALL [MAX-CONTEXT] [PROFIL]   = vsetky modely po jednom
    echo.
    echo Zoznam modelov:
    python -m lm_optimizer models
    pause
    exit /b 1
)

if /i "%~1"=="ALL" goto run_all
set MODEL=%~1
if "%~2"=="" (set MAXCTX=8192) else (set MAXCTX=%~2)
if "%~3"=="" (set PROFIL=balanced) else (set PROFIL=%~3)

echo Model: %MODEL%  max-context: %MAXCTX%  profil: %PROFIL%
python -m lm_optimizer auto %MODEL% --max-context %MAXCTX% --profile %PROFIL%
echo.
echo Hotovo. Vysledok: results\%MODEL%-best.md
pause
goto :eof

:run_all
if "%~2"=="" (set MAXCTX=8192) else (set MAXCTX=%~2)
if "%~3"=="" (set PROFIL=balanced) else (set PROFIL=%~3)
echo Vsetky modely po jednom:  max-context: %MAXCTX%  profil: %PROFIL%
python -m lm_optimizer auto --max-context %MAXCTX% --profile %PROFIL%
echo.
echo Hotovo. Vysledky: results\campaign\ + jednotlive -best.md
pause
