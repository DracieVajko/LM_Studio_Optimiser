@echo off
REM LM Studio Auto Optimizer - primary Windows launcher.
REM run-model.bat / run-all.bat / run-web.bat remain as utilities - see README.
REM Batch parser rules used here: no parentheses inside blocks, no if-else
REM blocks, linear goto flow. Every failure prints reason + log + waits.
setlocal EnableDelayedExpansion

cd /d "%~dp0"
set LOGFILE=data\logs\startup.log
if not exist "data\logs" mkdir "data\logs" 2>nul

call :log "=== LM Studio Auto Optimizer startup ==="
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
call :log "Python: %PYVER%"
call :log "WorkDir: %CD%"

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 goto fail_python
echo       Python OK - %PYVER%
call :log "command: python --version = %PYVER%"

echo [2/4] Checking virtual environment and dependencies...
if exist ".venv\Scripts\python.exe" goto deps_check
echo       Creating .venv - one-time step, may take a few minutes...
call :log "creating .venv"
python -m venv .venv >> "%LOGFILE%" 2>&1
if errorlevel 1 goto fail_venv
echo       Installing dependencies - one-time step, may take a few minutes...
call :log "command: pip install -e .[dev]"
".venv\Scripts\python.exe" -m pip install -e ".[dev]" >> "%LOGFILE%" 2>&1
if errorlevel 1 goto fail_deps
goto deps_done

:deps_check
".venv\Scripts\python.exe" -c "import lm_optimizer" >> "%LOGFILE%" 2>&1
if errorlevel 1 goto deps_repair
goto deps_done

:deps_repair
echo       Dependencies missing, installing - may take a few minutes...
call :log "command: pip install -e .[dev] - repair"
".venv\Scripts\python.exe" -m pip install -e ".[dev]" >> "%LOGFILE%" 2>&1
if errorlevel 1 goto fail_deps
goto deps_done

:fail_python
call :fail "Python not found on PATH. Install Python 3.11 or newer." "python --version"
goto end
:fail_venv
call :fail "Could not create .venv. See log for the python error." "python -m venv .venv"
goto end
:fail_deps
call :fail "Dependency install failed. See log for the pip error." "pip install -e .[dev]"
goto end

:deps_done
set PYBIN=.venv\Scripts\python.exe
echo       Dependencies OK

:menu
echo.
echo ========================================
echo  LM Studio Auto Optimizer
echo ========================================
echo  [1] Start Web UI - server plus browser
echo  [2] Optimize one model
echo  [3] Check LM Studio connection
echo  [4] List models
echo  [5] Exit
echo.
set /p choice="Select 1-5: "
if "%choice%"=="1" goto webui
if "%choice%"=="2" goto optimize_one
if "%choice%"=="3" goto check
if "%choice%"=="4" goto models
if "%choice%"=="5" goto end
echo Invalid choice.
goto menu

:webui
echo [3/4] Starting LM Optimizer...
call :log "starting web server in visible console"
start "LM Studio Auto Optimizer - server" "%PYBIN%" -m lm_optimizer.web_main
if errorlevel 1 goto fail_start
echo [4/4] Waiting for http://127.0.0.1:8080 ...
call :log "polling /api/status - 30s timeout"
set READY=0
for /L %%i in (1,1,30) do call :poll_once
if "%READY%"=="1" goto ready
goto fail_timeout

:poll_once
curl.exe -sf http://127.0.0.1:8080/api/status >nul 2>&1
if errorlevel 1 goto poll_wait
set READY=1
goto :eof
:poll_wait
timeout /t 1 >nul
goto :eof

:ready
echo.
echo LM Studio Auto Optimizer is running:
echo http://127.0.0.1:8080
echo.
echo Opening browser...
call :log "ready - opening browser"
start http://127.0.0.1:8080/
echo.
echo Server log streams live in the server window.
echo Launcher stages: %CD%\%LOGFILE%
echo.
echo Press any key to STOP the server and return to the menu.
pause >nul
call :log "stopping server"
taskkill /FI "WINDOWTITLE eq LM Studio Auto Optimizer - server*" >nul 2>&1
goto menu

:optimize_one
"%PYBIN%" -m lm_optimizer models --full-ids > "%TEMP%\lmo_models.txt" 2>nul
type "%TEMP%\lmo_models.txt"
echo.
set /p osel="Model number or full ID - empty returns to menu: "
if "%osel%"=="" goto menu
set omodel=%osel%
for /f "tokens=1,2 delims=|" %%a in ('type "%TEMP%\lmo_models.txt"') do if "%%a"=="%osel%" set omodel=%%b
set /p oprofile="Profile speed/balanced/context/quality - default balanced: "
if "%oprofile%"=="" set oprofile=balanced
echo Selected model: %omodel%
echo Profile: %oprofile%
set /p oconfirm="Start optimization? Y/n: "
if /i not "%oconfirm%"=="Y" if not "%oconfirm%"=="" goto menu
call :log "optimize %omodel% profile %oprofile%"
"%PYBIN%" -m lm_optimizer optimize "%omodel%" --profile "%oprofile%"
echo.
echo Finished with exit code %errorlevel%. Press any key for menu.
pause >nul
goto menu

:check
"%PYBIN%" -m lm_optimizer status --quick
echo.
echo Press any key for menu.
pause >nul
goto menu

:models
"%PYBIN%" -m lm_optimizer models
echo.
"%PYBIN%" -m lm_optimizer models --full-ids
echo.
echo Copy a full ID from the numbered list above.
echo Press any key for menu.
pause >nul
goto menu

:fail_start
call :fail "Could not start the server process." "web_main"
goto end

:fail_timeout
echo.
echo ====================================
echo LM Studio Auto Optimizer FAILED
echo ====================================
echo Reason: server did not answer /api/status within 30 seconds.
echo.
echo Last log lines:
powershell -NoProfile -Command "Get-Content '%LOGFILE%' -Tail 20"
echo.
echo Log:
echo %CD%\%LOGFILE%
echo.
echo Look at the server window for live output. Press any key to close.
pause >nul
call :log "FAILED: readiness timeout"
goto end

:fail
echo.
echo ====================================
echo LM Studio Auto Optimizer FAILED
echo ====================================
echo Reason: %~1
echo Command: %~2
echo.
echo Log:
echo %CD%\%LOGFILE%
echo.
echo Press any key to close.
call :log "FAILED: %~1 [%~2]"
pause >nul
goto :eof

:log
echo %date% %time% %~1>> "%LOGFILE%"
goto :eof

:end
endlocal
