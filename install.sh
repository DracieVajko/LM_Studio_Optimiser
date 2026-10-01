#!/usr/bin/env bash
# =====================================================================
#  LM Studio Optimizer - Linux/macOS INSTALL
#  Creates .venv, installs the package, copies .env.example to .env
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 not found. Install Python 3.11+ first."
    exit 1
fi

echo "[1/3] Creating virtual environment..."
[ -d ".venv" ] || python3 -m venv .venv

echo "[2/3] Installing package..."
# shellcheck disable=SC1091
source ".venv/bin/activate"
python -m pip install --upgrade pip
pip install -e ".[dev]"

echo "[3/3] Configuration..."
if [ ! -f ".env" ]; then
    cp ".env.example" ".env"
    echo "Created .env from example - adjust if needed."
else
    echo ".env already exists - kept."
fi

echo ""
echo "Done. Next steps:"
echo "  1. Start LM Studio with Developer server on 127.0.0.1:1234"
echo "  2. ./run-web.sh  (web UI)  or  source .venv/bin/activate, then  python -m lm_optimizer --help"
