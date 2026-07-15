#include "tinfer_rust/src/models/styletts2/cpp/cpu/glue.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tinfer::styletts2::cpu {
Alignment duration_prefix(const float* durations, const std::int32_t* lengths,
                          const float* speeds, std::int32_t batch,
                          std::int32_t token_count) {
  if (batch < 1 || token_count < 1) throw std::invalid_argument("invalid alignment dimensions");
  Alignment result;
  result.durations.resize(static_cast<std::size_t>(batch) * token_count);
  result.starts.resize(result.durations.size());
  result.totals.resize(batch);
  for (std::int32_t item = 0; item < batch; ++item) {
    if (lengths[item] < 0 || lengths[item] > token_count || speeds[item] <= 0.0F) {
      throw std::invalid_argument("invalid alignment length or speed");
    }
    std::int32_t total = 0;
    for (std::int32_t token = 0; token < token_count; ++token) {
      const auto index = static_cast<std::size_t>(item) * token_count + token;
      const auto duration = token < lengths[item]
                                ? std::max<std::int32_t>(1, std::lround(durations[index] / speeds[item]))
                                : 0;
      result.durations[index] = duration;
      result.starts[index] = total;
      total += duration;
    }
    result.totals[item] = total;
    result.frames = std::max(result.frames, total);
  }
  result.tokens.assign(static_cast<std::size_t>(batch) * result.frames, 0);
  for (std::int32_t item = 0; item < batch; ++item) {
    for (std::int32_t token = 0; token < token_count; ++token) {
      const auto index = static_cast<std::size_t>(item) * token_count + token;
      for (std::int32_t frame = result.starts[index]; frame < result.starts[index] + result.durations[index]; ++frame) {
        result.tokens[static_cast<std::size_t>(item) * result.frames + frame] = token;
      }
    }
  }
  return result;
}

void align_expand(const float* text, const float* encoding,
                  const Alignment& alignment, std::int32_t batch,
                  std::int32_t token_count, std::int32_t encoding_channels,
                  std::vector<float>& asr, std::vector<float>& expanded) {
  asr.assign(static_cast<std::size_t>(batch) * 512 * alignment.frames, 0.0F);
  expanded.assign(static_cast<std::size_t>(batch) * encoding_channels * alignment.frames, 0.0F);
  for (std::int32_t item = 0; item < batch; ++item) {
    for (std::int32_t frame = 0; frame < alignment.totals[item]; ++frame) {
      const auto token = alignment.tokens[static_cast<std::size_t>(item) * alignment.frames + frame];
      for (std::int32_t channel = 0; channel < 512; ++channel) {
        asr[(static_cast<std::size_t>(item) * 512 + channel) * alignment.frames + frame] =
            text[(static_cast<std::size_t>(item) * 512 + channel) * token_count + token];
      }
      for (std::int32_t channel = 0; channel < encoding_channels; ++channel) {
        expanded[(static_cast<std::size_t>(item) * encoding_channels + channel) * alignment.frames + frame] =
            encoding[(static_cast<std::size_t>(item) * token_count + token) * encoding_channels + channel];
      }
    }
  }
}

}  // namespace tinfer::styletts2::cpu

namespace tinfer::native {

rust::Vec<std::int32_t> cpu_duration_prefix(rust::Slice<const float> durations,
                                            rust::Slice<const std::int32_t> lengths,
                                            rust::Slice<const float> speeds,
                                            std::int32_t batch, std::int32_t tokens) {
  if (durations.size() != static_cast<std::size_t>(batch) * tokens ||
      lengths.size() != static_cast<std::size_t>(batch) || speeds.size() != lengths.size()) {
    throw std::invalid_argument("CPU duration inputs differ from dimensions");
  }
  const auto result = styletts2::cpu::duration_prefix(durations.data(), lengths.data(), speeds.data(), batch, tokens);
  rust::Vec<std::int32_t> values;
  values.reserve(result.durations.size());
  for (const auto value : result.durations) values.push_back(value);
  return values;
}

}  // namespace tinfer::native
