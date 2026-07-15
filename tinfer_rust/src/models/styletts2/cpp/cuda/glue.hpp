#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tinfer::styletts2::cuda {

void duration_prefix(const __half* durations, const std::int32_t* lengths,
                     const float* speeds, std::int32_t* predicted,
                     std::int32_t* starts, std::int32_t* totals,
                     std::int32_t batch, std::int32_t tokens,
                     cudaStream_t stream);
void build_tokens(const std::int32_t* predicted, const std::int32_t* starts,
                  const std::int32_t* totals, std::int32_t* token_of_frame,
                  std::int32_t batch, std::int32_t tokens,
                  std::int32_t frames, cudaStream_t stream);
void align_expand(const __half* text, const __half* encoding,
                  const std::int32_t* token_of_frame,
                  const std::int32_t* totals, __half* asr, __half* expanded,
                  std::int32_t batch, std::int32_t tokens,
                  std::int32_t channels, std::int32_t frames,
                  cudaStream_t stream);
void source_to_har(const __half* f0, const __half* weights,
                   const __half* bias, const std::uint64_t* seeds, __half* har,
                   float* phases, float* phase_state,
                   const std::int32_t* advances, float* source,
                   std::int32_t batch, std::int32_t frames, bool randomize,
                   cudaStream_t stream);

}  // namespace tinfer::styletts2::cuda
