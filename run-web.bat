@echo off
REM =====================================================================
REM  LM Studio Optimizer - WEB UI na http://127.0.0.1:8080
REM  Ukoncis klavesou CTRL+C
REM =====================================================================
cd /d "%~dp0"
echo Web UI: http://127.0.0.1:8080
echo (ukoncis klavesou CTRL+C)
python -m lm_optimizer.web_main
pause
