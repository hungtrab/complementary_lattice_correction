#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [[ "$FORMAT" == compressed-tensors ]]; then
  exec vllm serve "$MODEL" "$@"
else
  exec vllm serve "$MODEL" --quantization "$FORMAT" "$@"
fi
