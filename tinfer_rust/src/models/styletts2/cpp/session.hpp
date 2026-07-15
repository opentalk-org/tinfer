#pragma once

#include "tinfer_rust/src/models/base/cpp/engine.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace tinfer::native {

struct StyleTts2Session {
  std::vector<float> text;
  std::vector<float> encoding;
  std::vector<float> style;
  std::vector<float> reference;
  std::vector<std::int32_t> durations;
  std::vector<std::int32_t> starts;
  std::vector<float> tail_text;
  std::vector<float> tail_encoding;
  std::optional<Buffer> device_text;
  std::optional<Buffer> device_encoding;
  std::optional<Buffer> device_starts;
  std::optional<Buffer> device_style;
  std::optional<Buffer> device_reference;
  std::optional<Buffer> device_tail_text;
  std::optional<Buffer> device_tail_encoding;
  std::int32_t tail_frames = 0;
  std::int32_t tokens = 0;
  std::int32_t channels = 0;
  std::int32_t frames = 0;
  std::int32_t cursor = 0;
  std::uint64_t seed = 0;
  std::uint64_t frame_offset = 0;
  std::array<float, 9> phase{};
};

struct StyleTts2Tail {
  std::vector<float> text;
  std::vector<float> encoding;
  std::optional<Buffer> device_text;
  std::optional<Buffer> device_encoding;
  std::int32_t channels = 0;
  std::int32_t frames = 0;
  std::array<float, 9> phase{};
  std::uint64_t seed = 0;
  std::uint64_t frame_offset = 0;
};

constexpr std::int32_t kWindowPre = 32;
constexpr std::int32_t kWindowCore = 128;
constexpr std::int32_t kWindowPost = 16;
constexpr std::int32_t kWindowFrames = kWindowPre + kWindowCore + kWindowPost;

}  // namespace tinfer::native
