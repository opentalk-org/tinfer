#include "tinfer_rust/src/models/base/native.rs.h"
#include "tinfer_rust/src/models/base/cpp/tensor.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/cpu/glue.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/model.hpp"

#ifdef TINFER_CUDA
#include "tinfer_rust/src/models/styletts2/cpp/cuda/glue.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace tinfer::native {
namespace {
#ifdef TINFER_CUDA
template <typename T>
Buffer upload_values(const std::vector<T>& values, DType dtype,
                     std::vector<std::int64_t> shape, std::int32_t device,
                     cudaStream_t stream) {
  auto output = allocate(dtype, std::move(shape), device);
  if (output.bytes != values.size() * sizeof(T) ||
      cudaMemcpyAsync(output.data(), values.data(), output.bytes,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    throw std::runtime_error("StyleTTS2 CUDA state upload failed");
  }
  return output;
}
#endif
}  // namespace

Tensors StyleTts2Model::upload(const Batch& batch,
                               const Execution& execution) const {
  Tensors output;
  for (const auto& source : batch.tensors) {
    validate_tensor(source);
    const auto name = std::string(source.name);
    const auto dtype = execution.input_dtype(name).value_or(source.dtype);
    auto value = allocate(dtype, {source.shape.begin(), source.shape.end()},
                          device_);
    if (source.dtype == dtype) {
      if (device_ < 0) {
        std::memcpy(value.data(), source.data.data(), value.bytes);
      } else {
#ifdef TINFER_CUDA
        if (cudaMemcpyAsync(value.data(), source.data.data(), value.bytes,
                            cudaMemcpyHostToDevice,
                            static_cast<cudaStream_t>(stream_)) != cudaSuccess) {
          throw std::runtime_error("StyleTTS2 CUDA upload failed");
        }
#endif
      }
    } else if (device_ >= 0 && source.dtype == DType::F32 &&
               dtype == DType::F16) {
#ifdef TINFER_CUDA
      const auto* floats = reinterpret_cast<const float*>(source.data.data());
      std::vector<__half> halves(source.data.size() / sizeof(float));
      std::transform(floats, floats + halves.size(), halves.begin(),
                     [](float item) { return __float2half(item); });
      if (cudaMemcpyAsync(value.data(), halves.data(), value.bytes,
                          cudaMemcpyHostToDevice,
                          static_cast<cudaStream_t>(stream_)) != cudaSuccess) {
        throw std::runtime_error("StyleTTS2 CUDA fp16 upload failed");
      }
#endif
    } else {
      throw std::runtime_error("unsupported StyleTTS2 input conversion");
    }
    output.emplace(name, std::move(value));
  }
  return output;
}

Buffer StyleTts2Model::upload_floats(const std::vector<float>& values,
                                     std::vector<std::int64_t> shape) const {
  if (device_ < 0) {
    auto output = allocate(DType::F32, std::move(shape), device_);
    if (output.bytes != values.size() * sizeof(float)) {
      throw std::runtime_error("StyleTTS2 host tensor shape differs from data");
    }
    std::memcpy(output.data(), values.data(), output.bytes);
    return output;
  }
#ifdef TINFER_CUDA
  std::vector<__half> halves(values.size());
  std::transform(values.begin(), values.end(), halves.begin(),
                 [](float item) { return __float2half(item); });
  return upload_values(halves, DType::F16, std::move(shape), device_,
                       static_cast<cudaStream_t>(stream_));
#else
  throw std::runtime_error("CUDA support is not enabled in this build");
#endif
}

Buffer StyleTts2Model::pad_columns(const Buffer& source, std::int64_t columns,
                                   int fill) const {
  if (source.shape.size() != 2 || source.shape[1] >= columns) return source;
  const auto rows = source.shape[0];
  const auto source_pitch = static_cast<std::size_t>(source.shape[1]) *
                            element_size(source.dtype);
  const auto target_pitch = static_cast<std::size_t>(columns) *
                            element_size(source.dtype);
  auto output = allocate(source.dtype, {rows, columns}, device_);
  if (device_ < 0) {
    std::memset(output.data(), fill, output.bytes);
    for (std::int64_t row = 0; row < rows; ++row) {
      std::memcpy(static_cast<std::uint8_t*>(output.data()) + row * target_pitch,
                  static_cast<const std::uint8_t*>(source.data()) + row * source_pitch,
                  source_pitch);
    }
    return output;
  }
#ifdef TINFER_CUDA
  const auto stream = static_cast<cudaStream_t>(stream_);
  if (cudaMemsetAsync(output.data(), fill, output.bytes, stream) != cudaSuccess ||
      cudaMemcpy2DAsync(output.data(), target_pitch, source.data(), source_pitch,
                        source_pitch, rows, cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess) {
    throw std::runtime_error("StyleTTS2 TensorRT input padding failed");
  }
  return output;
#else
  throw std::runtime_error("CUDA support is not enabled in this build");
#endif
}

std::vector<float> StyleTts2Model::download_floats(
    const Buffer& buffer) const {
  std::vector<float> output(buffer.bytes / element_size(buffer.dtype));
  if (device_ < 0) {
    if (buffer.dtype != DType::F32) {
      throw std::runtime_error("StyleTTS2 CPU output is not float32");
    }
    std::memcpy(output.data(), buffer.data(), buffer.bytes);
    return output;
  }
#ifdef TINFER_CUDA
  const auto stream = static_cast<cudaStream_t>(stream_);
  if (buffer.dtype == DType::F16) {
    std::vector<__half> halves(output.size());
    if (cudaMemcpyAsync(halves.data(), buffer.data(), buffer.bytes,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("StyleTTS2 CUDA download failed");
    }
    std::transform(halves.begin(), halves.end(), output.begin(),
                   [](__half item) { return __half2float(item); });
  } else if (buffer.dtype == DType::F32) {
    if (cudaMemcpyAsync(output.data(), buffer.data(), buffer.bytes,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("StyleTTS2 CUDA download failed");
    }
  } else {
    throw std::runtime_error("StyleTTS2 output is not floating point");
  }
  return output;
#else
  throw std::runtime_error("CUDA support is not enabled in this build");
#endif
}

Buffer StyleTts2Model::harmonic(const std::vector<float>& f0,
                                const std::vector<std::uint64_t>& seeds,
                                const std::vector<float>& phases) const {
  const auto batch = static_cast<std::int32_t>(seeds.size());
  if (f0.size() != static_cast<std::size_t>(batch) * kWindowFrames * 2 ||
      phases.size() != static_cast<std::size_t>(batch) * 9) {
    throw std::runtime_error("StyleTTS2 harmonic inputs differ from batch");
  }
  if (device_ >= 0) {
    throw std::runtime_error("host harmonic generation used for CUDA");
  }
  std::vector<std::int32_t> advances(batch, kWindowFrames);
  const auto weight = glue_weights_.at("linW");
  const auto bias = glue_weights_.at("linB");
  auto phase_state = phases;
  auto values = styletts2::cpu::source_to_har(
      f0.data(), static_cast<const float*>(weight.data()),
      *static_cast<const float*>(bias.data()), batch, kWindowFrames,
      seeds.data(), true, advances.data(), phase_state.data());
  return upload_floats(values, {batch, 22, kWindowFrames * 120 + 1});
}

}  // namespace tinfer::native
