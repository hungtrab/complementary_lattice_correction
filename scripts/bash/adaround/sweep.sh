#!/usr/bin/env bash
# AdaRound: learned rounding decisions followed by the same CLC correction.
# Set ADAROUND_ITERATIONS to a small value for a smoke test.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

MODELS="${MODELS:-mistralai/Mistral-7B-v0.3 meta-llama/Llama-3.1-8B}"
BITS="${BITS:-4}"

mkdir -p "$RESULTS_DIR"
for model in $MODELS; do
  for bits in $BITS; do
    run_variant "$model" adaround "$bits" none
    for budget in $BUDGETS; do
      run_variant "$model" adaround "$bits" clc "$budget"
    done
  done
done
