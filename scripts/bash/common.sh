#!/usr/bin/env bash
# Shared settings for the sweep scripts. Source this, do not run it.
set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
EXPORT_ROOT="${EXPORT_ROOT:-./out}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
N_CALIB="${N_CALIB:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
GROUP_SIZE="${GROUP_SIZE:-128}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
BUDGETS="${BUDGETS:-0.01 0.02 0.03 0.04 0.05}"
PERPLEXITY="${PERPLEXITY:-wikitext2 c4}"
CLC_CMD="${CLC_CMD:-clc}"
EXPORT_FORMAT="${EXPORT_FORMAT:-auto}"

# The AWQ kernels are 4-bit only; 3-bit checkpoints have to use the GPTQ layout.
export_format_for_bits() {
  local quantizer="$1" bits="$2"
  if [[ "$quantizer" == "awq" && "$bits" == "4" ]]; then echo "awq"; else echo "gptq"; fi
}

run_clc() {
  if [[ "$CLC_CMD" == "clc" ]] && ! command -v clc >/dev/null 2>&1; then
    local package_root
    package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
    PYTHONPATH="${package_root}${PYTHONPATH:+:${PYTHONPATH}}" python3 -m clc.cli "$@"
  else
    "$CLC_CMD" "$@"
  fi
}

run_variant() {
  local model="$1" quantizer="$2" bits="$3" correction="$4" budget="${5:-}"
  local slug="${model##*/}"
  local name="${quantizer}_${correction}_${slug}_w${bits}"
  local args=(
    --model "$model" --quantizer "$quantizer" --bits "$bits"
    --group-size "$GROUP_SIZE" --post-correction "$correction"
    --calib-dataset "$CALIB_DATASET" --n-calib "$N_CALIB"
    --calib-seqlen "$CALIB_SEQLEN" --seed "$SEED"
    --device "$DEVICE"
    --perplexity $PERPLEXITY
    --stats-out "${RESULTS_DIR}/${name}${budget:+_p${budget}}.json"
  )
  if [[ "$correction" == "clc" && -n "$budget" ]]; then
    args+=(--budget "$budget")
    name="${name}_p${budget}"
  fi
  if [[ "${DO_EXPORT:-0}" == "1" && "$correction" != "bias_correction" ]]; then
    local export_format="$EXPORT_FORMAT"
    if [[ "$export_format" == "auto" ]]; then
      export_format="$(export_format_for_bits "$quantizer" "$bits")"
    fi
    args+=(--export "${EXPORT_ROOT}/${name}"
           --export-format "$export_format")
  fi
  echo "=== ${name} ==="
  run_clc quantize "${args[@]}"
}
