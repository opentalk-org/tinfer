#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

cd "$ROOT"
uv run --package tinfer --extra inference --with matplotlib --with phonemizer \
  python -m test_speed.investigations.duration_training.train "$@"
uv run --package tinfer --extra inference --with matplotlib --with phonemizer \
  python -m test_speed.investigations.duration_training.audit
