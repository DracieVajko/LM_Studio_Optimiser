@echo off
REM =====================================================================
REM  LM Studio Optimizer - OPTIMALIZUJ UPLNE VSETKO (cez noc)
REM  Malé modely: plný tuning. Veľké modely (>=5GB): fit-stropy.
REM  Výsledky: results\*.md   Logy: logs\
REM =====================================================================
cd /d "%~dp0"

echo [1/4] Kontrola LM Studio (127.0.0.1:1234)...
python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:1234/api/v1/models', timeout=10); print('LM Studio bezi')" 2>nul
if errorlevel 1 (
    echo CHYBA: LM Studio nebezi alebo nema zapnuty Developer server.
    echo        Zapni LM Studio - Settings - Developer - Start server.
    pause
    exit /b 1
)

echo [2/4] Rýchle testy...
python -m pytest -q
if errorlevel 1 (
    echo CHYBA: testy nepresli, koncim.
    pause
    exit /b 1
)

echo [3/4] AUTO pipeline - male modely do 6GB, plny tuning...
python -m lm_optimizer auto --max-size-gb 6 --max-context 8192
if errorlevel 1 (
    echo VAROVANIE: auto skoncilo s chybou, pokracujem na fit...
)

echo [4/4] FIT ladder - velke modely, max kontext stropy...
python -m lm_optimizer fit

echo.
echo HOTOVO. Pozri vysledky:
echo   - per-model reporty: results\*-best.md a results\*-fit.md
echo   - suhrny: results\auto-summary-*.md
pause
