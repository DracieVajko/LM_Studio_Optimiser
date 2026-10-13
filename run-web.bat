@echo off
REM =====================================================================
REM  LM Studio Optimizer - WEB UI on http://127.0.0.1:8080
REM  Stop with CTRL+C
REM =====================================================================
cd /d "%~dp0"
echo Web UI: http://127.0.0.1:8080
echo (stop with CTRL+C)
python -m lm_optimizer.web_main
pause
