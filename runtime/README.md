# StyleTTS2 → 3 TensorRT engines + CUDA glue (C++ runtime)

The full StyleTTS2 (istftnet, Polish) inference pipeline compiled into **3 dynamic,
weight-input TensorRT engines** glued by hand-written **CUDA kernels**, driven by a
standalone **C++/CUDA runtime** (no Python/torch at runtime).

Target met: **6.66 ms / 150 chars (~11 s audio) at batch 16** on an RTX 5090 (fp16).

## Pipeline

```
tokens ─▶ [Engine A] ─▶ durations,d,t_en,s,ref
                          │
                   glue1 (CUDA): duration round/speed/clamp + alignment
                   repeat-interleave  t_en→asr, d→en  (per-item frame count)
                          │
                     [Engine B] ─▶ F0, N            (F0Ntrain)
                          │
                   glue2 (CUDA): SourceModuleHnNSF  F0→har
                   f02sine (phase cumsum + interp) + forward STFT-20 (atan2)
                          │
                     [Engine C] ─▶ audio            (decoder backbone + generator)
```

- **A** = text_encoder + PLBERT + bert_encoder + ADPM2 diffusion + style blend + duration predictor.
- **B** = predictor.F0Ntrain.
- **C** = decoder backbone + generator (`forward_with_har`).
- **glue1/glue2** = `runtime/src/kernels.cu` (the ops that don't compile to TensorRT).

The A→B split lets B/C run at the *actual* predicted frame count (not padded to max).
The B→C split isolates the f02sine (cumsum) + forward-STFT that TensorRT can't build.

## Weights are engine INPUTS (not baked, not refit)

Every model weight is promoted from an ONNX initializer to a graph **input**
(`trt3/core.py: promote_initializers_to_inputs`). The engine plan is weight-agnostic,
so swapping voice/model = binding different weight tensors — proven: agnieszka's
engine C running magda's weights reproduces magda's audio (0.68% mean err). All 3
converted voices share one architecture (779 identical param shapes).

## Build (offline, python/torch)

```
cd scripts
.venv/bin/python build_c.py            # engine C + C.weights + C.ref
.venv/bin/python build_b.py            # engine B
.venv/bin/python build_a.py            # engine A  (add --cfg for dynamic embedding_scale)
.venv/bin/python dump_glue.py          # glue reference; glue.weights dumped separately
```

## Runtime (C++/CUDA)

```
cd runtime
nvcc -c -O2 -arch=sm_120 src/kernels.cu -o kernels.o -I include
g++ -std=c++17 -O2 src/pipeline.cpp kernels.o -o pipeline \
    -I include -I /usr/local/cuda/include \
    -L <venv>/lib/python3.11/site-packages/tensorrt_libs -l:libnvinfer.so.11 \
    -L /usr/local/cuda/lib64 -lcudart -Wl,-rpath,...
./pipeline engines out.wav          # full pipeline -> wav
./bench_pipeline engines            # benchmark batch 1/2/4/8/16
./bench_engine engines C audio      # per-engine validate + bench
./test_glue engines                 # glue2 kernel vs pytorch
```

TensorRT v11.1 C++ headers are vendored in `runtime/include/` (fetched from the
matching OSS tag; ABI-identical to the pip `libnvinfer.so.11`).

## Results (RTX 5090, fp16, 150 chars ≈ 11 s audio)

| batch | full pipeline ms/item |
|------:|----------------------:|
| 1  | 19.5 |
| 2  | 13.3 |
| 4  | 9.3  |
| 8  | 7.5  |
| 16 | **6.66** |

Per engine at batch 16: A 1.47 · B 0.32 · C 4.30 ms/item.
TRT-vs-PyTorch audio log-spectrum correlation **0.9928**.
PyTorch eager baseline: ~245 ms (batch 1).

## Dynamic per-inference params

`alpha`, `beta` (engine A inputs), `speed` (glue1), `diffusion_steps` (engine A
`sigmas` schedule input — any K ≤ compiled max via an identity-padded schedule),
batch size, char count, and asr-frame count are all dynamic. `embedding_scale` is
dynamic in the `--cfg` build (classifier-free guidance doubles the diffusion cost,
~7.1 ms/item); the default scale=1.0 build stays under 7 ms.

## Key techniques

- **Native InstanceNorm AdaIN** instead of manual mean/var → decoder 10.4→6.2 ms.
- **Virtual output splits** at conv/matmul fusion boundaries to defeat a TensorRT
  `enqueueV3` crash that only appears with runtime weights on both sides of a fusion.
- **fp32 sampler arithmetic** around an fp16 denoise transformer (small-sigma
  divisions overflow fp16 → NaN).
- **Reshape-based nearest upsample** in F0Ntrain to avoid a Myelin resize-lowering bug.
