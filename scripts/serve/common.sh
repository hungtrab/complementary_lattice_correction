#!/usr/bin/env bash
# Shared argument handling for the engine launch wrappers.
set -euo pipefail

if [[ -v MODEL ]]; then
  :
else
  MODEL="$1"
  shift
fi

if [[ -v FORMAT ]]; then
  :
elif [[ -v MODEL_FORMAT ]]; then
  FORMAT="$MODEL_FORMAT"
else
  FORMAT=awq
fi
