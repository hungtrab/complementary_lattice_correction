#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [[ "$FORMAT" == compressed-tensors ]]; then
  echo "TGI does not directly consume compressed-tensors checkpoints; use AWQ or GPTQ." >&2
  exit 2
fi
exec text-generation-launcher \
  --model-id "$MODEL" \
  --quantize "$FORMAT" \
  "$@"
