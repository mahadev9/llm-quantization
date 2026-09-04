#!/usr/bin/env bash
# Benchmark each model with lm-eval-harness (HF backend). Run once per model,
# then eyeball the scores lm-eval prints at the end of each run.
set -euo pipefail
cd "$(dirname "$0")"
[[ -d .venv ]] && source .venv/bin/activate

MODELS=(
  "Qwen3.5-0.8B-fp8"
  "Qwen3.5-4B-fp8"
)
TASKS=gsm8k,mmlu,ifeval
LIMIT=200

for model in "${MODELS[@]}"; do
  echo "================ $model ================"
  lm_eval --model hf \
    --model_args "pretrained=$model,dtype=auto,device_map=auto" \
    --tasks "$TASKS" \
    --apply_chat_template --fewshot_as_multiturn \
    --batch_size 8 \
    --limit "$LIMIT" \
    --output_path "eval_results/$model"
done
