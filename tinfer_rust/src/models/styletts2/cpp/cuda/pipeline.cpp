#include "tinfer_rust/src/models/base/native.rs.h"
#include "tinfer_rust/src/models/base/cpp/tensor.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/model.hpp"

#ifdef TINFER_CUDA
#include "tinfer_rust/src/models/styletts2/cpp/cuda/glue.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace tinfer::native {
#ifdef TINFER_CUDA
namespace {
const Buffer& require(const Tensors& tensors, std::string_view name, DType dtype) {
  const auto found = tensors.find(name);
  if (found == tensors.end() || found->second.dtype != dtype) throw std::runtime_error("missing or invalid CUDA tensor: " + std::string(name));
  return found->second;
}

Tensors merge(const Tensors& first, const Tensors& second) {
  Tensors result = first;
  result.insert(second.begin(), second.end());
  return result;
}

Buffer upload(const Tensor& source, DType dtype, std::int32_t device, cudaStream_t stream) {
  std::vector<std::int64_t> shape(source.shape.begin(), source.shape.end());
  auto output = allocate(dtype, std::move(shape), device);
  if (source.dtype == dtype) {
    if (source.data.size() != output.bytes || cudaMemcpyAsync(output.data(), source.data.data(), output.bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      throw std::runtime_error("StyleTTS2 CUDA input upload failed");
    }
  } else if (source.dtype == DType::F32 && dtype == DType::F16) {
    if (source.data.size() % sizeof(float) != 0) throw std::runtime_error("invalid StyleTTS2 float input");
    const auto* values = reinterpret_cast<const float*>(source.data.data());
    std::vector<__half> halves(source.data.size() / sizeof(float));
    std::transform(values, values + halves.size(), halves.begin(), [] (float value) { return __float2half(value); });
    if (cudaMemcpyAsync(output.data(), halves.data(), output.bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      throw std::runtime_error("StyleTTS2 CUDA fp16 upload failed");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) throw std::runtime_error("StyleTTS2 CUDA input synchronization failed");
  } else {
    throw std::runtime_error("unsupported StyleTTS2 CUDA input conversion");
  }
  return output;
}

Tensor output_tensor(const std::string& name, DType dtype,
                     const std::vector<std::int64_t>& shape, const void* data,
                     std::size_t bytes) {
  Tensor output;
  output.name = name;
  output.dtype = dtype;
  for (const auto dimension : shape) output.shape.push_back(dimension);
  output.data.reserve(bytes);
  const auto* source = static_cast<const std::uint8_t*>(data);
  for (std::size_t index = 0; index < bytes; ++index) output.data.push_back(source[index]);
  return output;
}

std::vector<float> copy_half(const Buffer& buffer, cudaStream_t stream) {
  std::vector<__half> halves(buffer.bytes / sizeof(__half));
  if (cudaMemcpyAsync(halves.data(), buffer.data(), buffer.bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) throw std::runtime_error("StyleTTS2 CUDA output copy failed");
  std::vector<float> values(halves.size());
  std::transform(halves.begin(), halves.end(), values.begin(), [] (__half value) { return __half2float(value); });
  return values;
}
}  // namespace

Output StyleTts2Model::run_cuda(const Batch& batch) const {
  if (cudaSetDevice(device_) != cudaSuccess) throw std::runtime_error("cannot select StyleTTS2 CUDA device");
  const auto stream = static_cast<cudaStream_t>(stream_);
  Tensors inputs;
  for (const auto& tensor : batch.tensors) {
    validate_tensor(tensor);
    const auto target = execution_a_->input_dtype(std::string(tensor.name)).value_or(tensor.dtype);
    inputs.emplace(std::string(tensor.name), upload(tensor, target, device_, stream));
  }
  const auto& tokens = require(inputs, "tokens", DType::I64);
  const auto batch_size = static_cast<std::int32_t>(tokens.shape[0]);
  const auto token_count = static_cast<std::int32_t>(tokens.shape[1]);
  auto a = execution_a_->run(merge(weights_a_, inputs), stream_);
  const auto& duration = require(a, "dur", DType::F16);
  auto predicted = allocate(DType::I32, {batch_size, token_count}, device_);
  auto starts = allocate(DType::I32, {batch_size, token_count}, device_);
  auto totals = allocate(DType::I32, {batch_size}, device_);
  styletts2::cuda::duration_prefix(static_cast<__half*>(duration.data()), static_cast<std::int32_t*>(require(inputs, "lengths", DType::I32).data()),
                                   static_cast<float*>(require(inputs, "speed", DType::F32).data()), static_cast<std::int32_t*>(predicted.data()),
                                   static_cast<std::int32_t*>(starts.data()), static_cast<std::int32_t*>(totals.data()), batch_size, token_count, stream);
  std::vector<std::int32_t> host_totals(batch_size);
  if (cudaMemcpyAsync(host_totals.data(), totals.data(), totals.bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) throw std::runtime_error("StyleTTS2 frame count copy failed");
  const auto frames = *std::max_element(host_totals.begin(), host_totals.end());
  auto frame_tokens = allocate(DType::I32, {batch_size, frames}, device_);
  auto asr = allocate(DType::F16, {batch_size, 512, frames}, device_);
  const auto& encoded = require(a, "d", DType::F16);
  const auto channels = static_cast<std::int32_t>(encoded.shape[2]);
  auto expanded = allocate(DType::F16, {batch_size, channels, frames}, device_);
  styletts2::cuda::build_tokens(static_cast<std::int32_t*>(predicted.data()), static_cast<std::int32_t*>(starts.data()),
                                static_cast<std::int32_t*>(totals.data()), static_cast<std::int32_t*>(frame_tokens.data()),
                                batch_size, token_count, frames, stream);
  styletts2::cuda::align_expand(static_cast<__half*>(require(a, "t_en", DType::F16).data()), static_cast<__half*>(encoded.data()),
                                static_cast<std::int32_t*>(frame_tokens.data()), static_cast<std::int32_t*>(totals.data()),
                                static_cast<__half*>(asr.data()), static_cast<__half*>(expanded.data()), batch_size, token_count, channels, frames, stream);
  auto b_inputs = weights_b_;
  b_inputs.emplace("en", std::move(expanded));
  b_inputs.emplace("s", require(a, "s", DType::F16));
  auto b = execution_b_->run(b_inputs, stream_);
  auto phases = allocate(DType::F32, {batch_size, 9, frames * 2}, device_);
  auto source = allocate(DType::F32, {batch_size, frames * 600}, device_);
  auto harmonic = allocate(DType::F16, {batch_size, 22, frames * 120 + 1}, device_);
  styletts2::cuda::source_to_har(static_cast<__half*>(require(b, "f0", DType::F16).data()),
                                 static_cast<__half*>(require(glue_weights_, "linW", DType::F16).data()),
                                 static_cast<__half*>(require(glue_weights_, "linB", DType::F16).data()),
                                 static_cast<std::uint64_t*>(require(inputs, "seeds", DType::I64).data()),
                                 static_cast<__half*>(harmonic.data()), static_cast<float*>(phases.data()),
                                 static_cast<float*>(source.data()), batch_size, frames, true, stream);
  auto c_inputs = weights_c_;
  c_inputs.emplace("asr", std::move(asr));
  c_inputs.emplace("f0", require(b, "f0", DType::F16));
  c_inputs.emplace("noise", require(b, "noise", DType::F16));
  c_inputs.emplace("style", require(a, "ref", DType::F16));
  c_inputs.emplace("har", std::move(harmonic));
  auto c = execution_c_->run(c_inputs, stream_);
  const auto& audio_device = require(c, "audio", DType::F16);
  const auto audio = copy_half(audio_device, stream);
  std::vector<std::int32_t> host_durations(static_cast<std::size_t>(batch_size) * token_count);
  if (cudaMemcpyAsync(host_durations.data(), predicted.data(), predicted.bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error("StyleTTS2 duration copy failed");
  }
  const auto references = copy_half(require(a, "ref", DType::F16), stream);
  const auto styles = copy_half(require(a, "s", DType::F16), stream);
  std::vector<float> state(static_cast<std::size_t>(batch_size) * 256);
  for (std::int32_t item = 0; item < batch_size; ++item) {
    std::copy_n(references.begin() + static_cast<std::size_t>(item) * 128, 128, state.begin() + static_cast<std::size_t>(item) * 256);
    std::copy_n(styles.begin() + static_cast<std::size_t>(item) * 128, 128, state.begin() + static_cast<std::size_t>(item) * 256 + 128);
  }
  Output output;
  output.tensors.push_back(output_tensor("audio", DType::F32, audio_device.shape, audio.data(), audio.size() * sizeof(float)));
  output.tensors.push_back(output_tensor("durations", DType::I32, {batch_size, token_count}, host_durations.data(), host_durations.size() * sizeof(std::int32_t)));
  output.tensors.push_back(output_tensor("frames", DType::I32, {batch_size}, host_totals.data(), host_totals.size() * sizeof(std::int32_t)));
  output.tensors.push_back(output_tensor("style", DType::F32, {batch_size, 256}, state.data(), state.size() * sizeof(float)));
  return output;
}
#else
Output StyleTts2Model::run_cuda(const Batch&) const {
  throw std::runtime_error("CUDA support is not enabled in this build");
}
#endif

}  // namespace tinfer::native
