#!/usr/bin/env bash
set -e
SP=/workspace/tinfer/.venv/lib/python3.11/site-packages
TRTLIB=$SP/tensorrt_libs
CUDAINC=$SP/nvidia/cuda_runtime/include
CUDALIB=$SP/nvidia/cuda_runtime/lib
[ -e "$TRTLIB/libnvinfer.so" ] || ln -sf libnvinfer.so.11 "$TRTLIB/libnvinfer.so"
[ -e "$CUDALIB/libcudart.so" ] || ln -sf libcudart.so.12 "$CUDALIB/libcudart.so"
nvcc -std=c++17 -O3 --gpu-architecture=sm_120 --cudart shared \
  -I/tmp/trt_include -I"$CUDAINC" -I/tmp/cccl/nvidia/cuda_cccl/include \
  /workspace/tinfer/cpp/runner.cu -o /workspace/tinfer/cpp/runner \
  -L"$TRTLIB" -L"$CUDALIB" -L/tmp/culibs -lnvinfer -lcudart
echo "BUILD_OK"
