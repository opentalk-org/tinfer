#include "tinfer_rust/src/models/base/native.rs.h"
#include "tinfer_rust/src/models/base/cpp/tensor.hpp"
#include "tinfer_rust/src/models/stub/cpp/native.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace tinfer::native {

Output StubModel::run(const Batch& batch) const {
  const auto& values_tensor = find_tensor(batch, "tokens");
  const auto& offsets_tensor = find_tensor(batch, "offsets");
  const auto* values = data<std::int64_t>(values_tensor);
  const auto* offsets = data<std::uint32_t>(offsets_tensor);
  const auto value_count = element_count(values_tensor);
  const auto offset_count = element_count(offsets_tensor);
  if (offset_count < 2 || offsets[0] != 0 || offsets[offset_count - 1] != value_count) {
    throw std::invalid_argument("invalid native batch offsets");
  }
  rust::Vec<float> audio;
  rust::Vec<std::uint32_t> output_offsets;
  output_offsets.push_back(0);
  for (std::size_t item = 0; item + 1 < offset_count; ++item) {
    for (std::size_t index = offsets[item]; index < offsets[item + 1]; ++index) {
      const auto value = static_cast<float>(static_cast<std::uint64_t>(values[index]) % 200) / 200.0F;
      for (std::size_t sample = 0; sample < 40 + (index - offsets[item]) % 3; ++sample) audio.push_back(value);
    }
    if (audio.size() > std::numeric_limits<std::uint32_t>::max()) throw std::length_error("native output exceeds offset range");
    output_offsets.push_back(static_cast<std::uint32_t>(audio.size()));
  }
  Tensor audio_tensor;
  audio_tensor.name = "audio";
  audio_tensor.dtype = DType::F32;
  audio_tensor.shape.push_back(static_cast<std::int64_t>(audio.size()));
  audio_tensor.data.reserve(audio.size() * sizeof(float));
  for (const auto value : audio) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    for (std::size_t index = 0; index < sizeof(float); ++index) audio_tensor.data.push_back(bytes[index]);
  }
  Tensor offset_tensor;
  offset_tensor.name = "offsets";
  offset_tensor.dtype = DType::I32;
  offset_tensor.shape.push_back(static_cast<std::int64_t>(output_offsets.size()));
  offset_tensor.data.reserve(output_offsets.size() * sizeof(std::uint32_t));
  for (const auto value : output_offsets) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    for (std::size_t index = 0; index < sizeof(std::uint32_t); ++index) offset_tensor.data.push_back(bytes[index]);
  }
  Output output;
  output.tensors.push_back(std::move(audio_tensor));
  output.tensors.push_back(std::move(offset_tensor));
  return output;
}

std::unique_ptr<Model> load_stub() {
  return std::make_unique<StubModel>();
}

}  // namespace tinfer::native
