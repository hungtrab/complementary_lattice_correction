#!/usr/bin/env bash
# Run a GGUF produced by clc convert-gguf with llama.cpp's CLI.
set -euo pipefail

if [[ -v MODEL ]]; then
  :
else
  if [[ "$#" -lt 1 ]]; then
    echo "usage: $0 MODEL.gguf [llama-cli arguments...]" >&2
    exit 2
  fi
  MODEL="$1"
  shift
fi

if [[ -v LLAMA_CLI_BIN ]]; then
  :
else
  LLAMA_CLI_BIN=llama-cli
fi

exec "$LLAMA_CLI_BIN" -m "$MODEL" "$@"
