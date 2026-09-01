#!/usr/bin/env bash
# Tool-calling eval with BFCL (Berkeley Function Calling Leaderboard).
# Generates responses then scores them; the accuracy table prints at the end.
set -euo pipefail
cd "$(dirname "$0")"
[[ -d .venv ]] && source .venv/bin/activate

MODELS=(
  "Qwen3.5-0.8B-fp8"
  "Qwen3.5-4B-fp8"
)
# non_live  = single-turn core (simple_python/java/js, parallel, multiple,
#             parallel_multiple, irrelevance)
# multi_turn = stateful multi-call (base, miss_func, miss_param, long_context)
# agentic    = memory_* + web_search_*
CATEGORIES=non_live,single_turn,multi_turn,agentic

for model in "${MODELS[@]}"; do
  echo "================ $model ================"
  bfcl generate \
    --model "$model" \
    --backend hf \
    --test-category "$CATEGORIES"
  bfcl evaluate --model "$model" --test-category "$CATEGORIES"
done

bfcl results