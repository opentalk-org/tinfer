#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WHEEL_DIR=/tmp/tinfer-espeak-wheel

cd "$ROOT"
uv sync --package tinfer --extra inference
rm -rf "$WHEEL_DIR"
mkdir -p "$WHEEL_DIR"
(
  cd tinfer/espeak_align/espeak_align
  uv run --with 'maturin[patchelf]' maturin build --release --out "$WHEEL_DIR"
)
uv pip install --python "$ROOT/.venv/bin/python" "$WHEEL_DIR"/espeak_align-*.whl
uv run --package tinfer --extra inference --with matplotlib --with phonemizer \
  python -m test_speed.duration_padding.genuine_length
