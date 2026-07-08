#!/usr/bin/env bash
SP=/workspace/tinfer/.venv/lib/python3.11/site-packages
NV=$(find $SP/nvidia -name '*.so*' -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="$SP/tensorrt_libs:$NV:$LD_LIBRARY_PATH"
/workspace/tinfer/cpp/runner "$@"
