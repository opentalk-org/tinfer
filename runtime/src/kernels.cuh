// CUDA glue kernels between the TensorRT engines (the ops that don't compile to
// TensorRT): alignment expansion (A->B) and the harmonic source + forward STFT
// (B->C, the f02sine part). All tensors are fp16 device pointers unless noted.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace glue {

// glue1: durations -> per-token frame counts + prefix sums.
// dur[B,T] fp16 (raw sigmoid-sum), lengths[B] int32, speed[B] f32.
// Writes predDur[B,T] int32, startFrame[B,T] int32, totalFrames[B] int32.
void duration_prefix(const __half* dur, const int32_t* lengths, const float* speed,
                     int32_t* predDur, int32_t* startFrame, int32_t* totalFrames,
                     int B, int T, cudaStream_t stream);

// glue1: expand token-resolution t_en/d to frame resolution via the alignment
// (repeat_interleave by predDur). tokenOfFrame[B,Fmax] built from prefix sums.
// asr[B,512,Fmax] = t_en[B,512,tok]; en[B,Cd,Fmax] = d[B,tok,Cd].
void build_token_of_frame(const int32_t* predDur, const int32_t* startFrame,
                          const int32_t* totalFrames, int32_t* tokenOfFrame,
                          int B, int T, int Fmax, cudaStream_t stream);
void align_expand(const __half* t_en, const __half* d, const int32_t* tokenOfFrame,
                  const int32_t* totalFrames, __half* asr, __half* en,
                  int B, int T, int Cd, int Fmax, cudaStream_t stream);

// glue2: F0[B,2F] -> har[B,22,120F+1]. Implements SourceModuleHnNSF (f02sine via
// downsample-cumsum-upsample) + linear/tanh + forward STFT-20 (hann, hop 5).
// weights: l_linear W[9], b[1] (fp16). randScale in {0,1} toggles rand_ini/noise.
void source_to_har(const __half* F0, const __half* linW, const __half* linB,
                   __half* har, int B, int F, uint64_t seed, int randScale,
                   cudaStream_t stream);

}  // namespace glue
