#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [[ -v BACKEND ]]; then
  :
else
  BACKEND=turbomind
fi

exec lmdeploy serve api_server "$MODEL" \
  --backend "$BACKEND" \
  --model-format "$FORMAT" \
  "$@"
