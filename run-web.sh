#!/usr/bin/env bash
# =====================================================================
#  LM Studio Optimizer - WEB UI on http://127.0.0.1:8080
#  Stop with CTRL+C
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"
echo "Web UI: http://127.0.0.1:8080"
echo "(stop with CTRL+C)"
exec python -m lm_optimizer.web_main
