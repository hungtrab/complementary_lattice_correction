#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

MODEL="${MODEL:-${1:?usage: MODEL=/path/to/model $0}}"
for bits in ${BITS_VALUES:-4}; do
  if [[ "${RUN_RAW:-1}" == "1" ]]; then
    run_variant "$MODEL" adaround "$bits" none
  fi
  if [[ "${RUN_CLC:-1}" == "1" ]]; then
    for budget in $BUDGETS; do
      run_variant "$MODEL" adaround "$bits" clc "$budget"
    done
  fi
done
