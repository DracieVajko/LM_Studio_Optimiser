#!/usr/bin/env bash
# =====================================================================
#  LM Studio Optimizer - CLI passthrough
#  Usage:  ./run-optimizer.sh <command> [args...]
#  Example: ./run-optimizer.sh status
#           ./run-optimizer.sh models
#           ./run-optimizer.sh auto qwen3.8-9b-distill --max-context 8192
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"
if [ "$#" -eq 0 ]; then
    echo "Usage: ./run-optimizer.sh <command> [args...]"
    echo ""
    echo "Available models:"
    exec python -m lm_optimizer models
fi
exec python -m lm_optimizer "$@"
