#pragma once

#include "rust/cxx.h"

#include <cstdint>
#include <vector>

namespace tinfer::styletts2::cpu {

struct Alignment {
  std::vector<std::int32_t> durations;
  std::vector<std::int32_t> starts;
  std::vector<std::int32_t> totals;
  std::vector<std::int32_t> tokens;
  std::int32_t frames = 0;
};

Alignment duration_prefix(const float* durations, const std::int32_t* lengths,
                          const float* speeds, std::int32_t batch,
                          std::int32_t token_count);
void align_expand(const float* text, const float* encoding,
                  const Alignment& alignment, std::int32_t batch,
                  std::int32_t token_count, std::int32_t encoding_channels,
                  std::vector<float>& asr, std::vector<float>& expanded);
std::vector<float> source_to_har(const float* f0, const float* weights,
                                 float bias, std::int32_t batch,
                                 std::int32_t frames, const std::uint64_t* seeds,
                                 bool randomize);

}  // namespace tinfer::styletts2::cpu

namespace tinfer::native {

rust::Vec<std::int32_t> cpu_duration_prefix(rust::Slice<const float> durations,
                                            rust::Slice<const std::int32_t> lengths,
                                            rust::Slice<const float> speeds,
                                            std::int32_t batch, std::int32_t tokens);

}  // namespace tinfer::native
