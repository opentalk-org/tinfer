#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

uv run --package tinfer --extra inference --with phonemizer \
  python -m test_speed.solution.aligned_training
uv run --package tinfer --extra inference --with phonemizer \
  python -m test_speed.solution.rate_regularized_training
uv run --package tinfer --extra inference --with matplotlib --with phonemizer \
  python -m test_speed.solution.run_benchmark
