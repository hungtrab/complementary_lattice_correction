#!/usr/bin/env bash
# GPTQ: base quantizer vs +BC vs +CLC, sweeping the flip budget p, at 3 and 4 bits.
# Reproduces the GPTQ rows of the main tables.
#
#   MODELS="mistralai/Mistral-7B-v0.3" DO_EXPORT=1 ./scripts/bash/gptq/sweep.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

MODELS="${MODELS:-mistralai/Mistral-7B-v0.3 meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B meta-llama/Meta-Llama-3-8B}"
BITS="${BITS:-4 3}"

mkdir -p "$RESULTS_DIR"
for model in $MODELS; do
  for bits in $BITS; do
    run_variant "$model" gptq "$bits" none
    run_variant "$model" gptq "$bits" bias_correction
    for budget in $BUDGETS; do
      run_variant "$model" gptq "$bits" clc "$budget"
    done
  done
done
