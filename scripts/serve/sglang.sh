#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
source "$SCRIPT_DIR/common.sh"

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --quantization "$FORMAT" \
  "$@"
