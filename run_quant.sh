#!/usr/bin/env bash
# Quantize then immediately KL-eval the result against the bf16 baseline.
set -euo pipefail
cd "$(dirname "$0")"
[[ -d .venv ]] && source .venv/bin/activate

SCHEME="${1:?usage: run_quant.sh <fp8|nvfp4>}"

python llm_quant.py "$SCHEME"
python eval_kl.py "$SCHEME"
