#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tinfer::styletts2::cuda {

struct WindowSource {
  const __half* text;
  const __half* encoding;
  const std::int32_t* starts;
  const __half* style;
  const __half* reference;
  const __half* tail_text;
  const __half* tail_encoding;
  std::int32_t tokens;
  std::int32_t text_stride;
  std::int32_t channels;
  std::int32_t frames;
  std::int32_t cursor;
  std::int32_t tail_frames;
};

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
void fill_noise(__half* output, const std::uint64_t* seeds,
                std::int32_t batch, std::int32_t samples,
                cudaStream_t stream);
void expand_windows(const WindowSource* sources, __half* asr,
                    __half* encoding, __half* styles, __half* references,
                    std::int32_t batch, std::int32_t channels,
                    std::int32_t frames,
                    cudaStream_t stream);

}  // namespace tinfer::styletts2::cuda
