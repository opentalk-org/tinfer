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
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tinfer::native {
Buffer StyleTts2Model::workspace(const std::string& name, DType dtype,
                                 std::vector<std::int64_t> shape,
                                 std::vector<std::int64_t> capacity) const {
  std::size_t bytes = element_size(dtype);
  for (const auto dimension : shape) bytes *= static_cast<std::size_t>(dimension);
  auto found = workspace_.find(name);
  if (found == workspace_.end() || found->second.bytes < bytes) {
    found = workspace_.insert_or_assign(name, allocate(dtype, std::move(capacity), device_)).first;
  }
  if (found->second.dtype != dtype || found->second.bytes < bytes) {
    throw std::runtime_error("StyleTTS2 workspace capacity is invalid");
  }
  return Buffer{dtype, std::move(shape), found->second.memory, bytes};
}

void* StyleTts2Model::pinned(const std::string& name, std::size_t bytes,
                             std::size_t capacity) const {
#ifdef TINFER_CUDA
  auto found = upload_staging_.find(name);
  if (found == upload_staging_.end() || found->second.bytes < bytes) {
    void* memory = nullptr;
    if (cudaMallocHost(&memory, capacity) != cudaSuccess) {
      throw std::runtime_error("StyleTTS2 pinned upload allocation failed");
    }
    auto release = [](void* value) { cudaFreeHost(value); };
    found = upload_staging_
                .insert_or_assign(name, PinnedBuffer{
                                            std::shared_ptr<void>(memory, release),
                                            capacity})
                .first;
  }
  return found->second.memory.get();
#else
  throw std::runtime_error("CUDA support is not enabled in this build");
#endif
}

Tensors StyleTts2Model::upload(const Batch& batch,
                               const Execution& execution) const {
  Tensors output;
  for (const auto& source : batch.tensors) {
    validate_tensor(source);
    const auto name = std::string(source.name);
    const auto dtype = execution.input_dtype(name).value_or(source.dtype);
    std::vector<std::int64_t> shape(source.shape.begin(), source.shape.end());
    auto capacity = shape;
    if (!capacity.empty()) capacity[0] = max_batch_;
    auto value = workspace("input." + name, dtype, std::move(shape),
                           std::move(capacity));
    if (source.dtype == dtype) {
      if (device_ < 0) {
        std::memcpy(value.data(), source.data.data(), value.bytes);
      } else {
#ifdef TINFER_CUDA
        const auto capacity_bytes = value.bytes /
                                    static_cast<std::size_t>(source.shape[0]) *
                                    static_cast<std::size_t>(max_batch_);
        auto* staging = pinned("input." + name, value.bytes, capacity_bytes);
        std::memcpy(staging, source.data.data(), value.bytes);
        if (cudaMemcpyAsync(value.data(), staging, value.bytes,
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
      const auto count = source.data.size() / sizeof(float);
      const auto capacity_bytes = count * sizeof(__half) /
                                  static_cast<std::size_t>(source.shape[0]) *
                                  static_cast<std::size_t>(max_batch_);
      auto* halves = static_cast<__half*>(
          pinned("input." + name, count * sizeof(__half), capacity_bytes));
      std::transform(floats, floats + count, halves,
                     [](float item) { return __float2half(item); });
      if (cudaMemcpyAsync(value.data(), halves, value.bytes,
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

Buffer StyleTts2Model::upload_floats(const std::string& name,
                                     const std::vector<float>& values,
                                     std::vector<std::int64_t> shape,
                                     DType device_dtype) const {
  auto capacity = shape;
  if (!capacity.empty()) capacity[0] = max_batch_;
  if (device_ < 0) {
    auto output = workspace(name, DType::F32, std::move(shape),
                            std::move(capacity));
    if (output.bytes != values.size() * sizeof(float)) {
      throw std::runtime_error("StyleTTS2 host tensor shape differs from data");
    }
    std::memcpy(output.data(), values.data(), output.bytes);
    return output;
  }
#ifdef TINFER_CUDA
  if (device_dtype == DType::F32) {
    auto output = workspace(name, DType::F32, std::move(shape),
                            std::move(capacity));
    const auto capacity_bytes = values.size() * sizeof(float) /
                                static_cast<std::size_t>(output.shape[0]) *
                                static_cast<std::size_t>(max_batch_);
    auto* staging = pinned(name, output.bytes, capacity_bytes);
    std::memcpy(staging, values.data(), output.bytes);
    if (cudaMemcpyAsync(output.data(), staging, output.bytes,
                        cudaMemcpyHostToDevice,
                        static_cast<cudaStream_t>(stream_)) != cudaSuccess) {
      throw std::runtime_error("StyleTTS2 CUDA state upload failed");
    }
    return output;
  }
  if (device_dtype != DType::F16) {
    throw std::runtime_error("StyleTTS2 CUDA state must be floating point");
  }
  const auto capacity_bytes = values.size() * sizeof(__half) /
                              static_cast<std::size_t>(shape[0]) *
                              static_cast<std::size_t>(max_batch_);
  auto* halves = static_cast<__half*>(
      pinned(name, values.size() * sizeof(__half), capacity_bytes));
  std::transform(values.begin(), values.end(), halves,
                 [](float item) { return __float2half(item); });
  auto output = workspace(name, DType::F16, std::move(shape),
                          std::move(capacity));
  if (cudaMemcpyAsync(output.data(), halves, output.bytes,
                      cudaMemcpyHostToDevice,
                      static_cast<cudaStream_t>(stream_)) != cudaSuccess) {
    throw std::runtime_error("StyleTTS2 CUDA state upload failed");
  }
  return output;
#else
  throw std::runtime_error("CUDA support is not enabled in this build");
#endif
}

Buffer StyleTts2Model::pad_columns(const std::string& name,
                                   const Buffer& source,
                                   std::int64_t columns, int fill) const {
  if (source.shape.size() != 2 || source.shape[1] >= columns) return source;
  const auto rows = source.shape[0];
  const auto source_pitch = static_cast<std::size_t>(source.shape[1]) *
                            element_size(source.dtype);
  const auto target_pitch = static_cast<std::size_t>(columns) *
                            element_size(source.dtype);
  auto output = workspace("input.padding." + name, source.dtype, {rows, columns},
                          {max_batch_, columns});
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
  auto capacity = buffer.bytes;
  if (!buffer.shape.empty() && buffer.shape[0] > 0) {
    capacity = buffer.bytes / static_cast<std::size_t>(buffer.shape[0]) *
               static_cast<std::size_t>(max_batch_);
  }
  if (download_capacity_ < capacity) {
    if (download_staging_ != nullptr) cudaFreeHost(download_staging_);
    if (cudaMallocHost(&download_staging_, capacity) != cudaSuccess) {
      throw std::runtime_error("StyleTTS2 pinned download allocation failed");
    }
    download_capacity_ = capacity;
  }
  if (buffer.dtype == DType::F16) {
    if (cudaMemcpyAsync(download_staging_, buffer.data(), buffer.bytes,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("StyleTTS2 CUDA download failed");
    }
    const auto* halves = static_cast<const __half*>(download_staging_);
    std::transform(halves, halves + output.size(), output.begin(),
                   [](__half item) { return __half2float(item); });
  } else if (buffer.dtype == DType::F32) {
    if (cudaMemcpyAsync(download_staging_, buffer.data(), buffer.bytes,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("StyleTTS2 CUDA download failed");
    }
    std::memcpy(output.data(), download_staging_, buffer.bytes);
  } else {
    throw std::runtime_error("StyleTTS2 output is not floating point");
  }
  return output;
#else
  throw std::runtime_error("CUDA support is not enabled in this build");
#endif
}

Buffer StyleTts2Model::source_noise(
    const std::vector<std::uint64_t>& seeds) const {
  const auto batch = static_cast<std::int32_t>(seeds.size());
  constexpr std::int32_t samples = kWindowFrames * 600;
  auto output = workspace("bc.source_noise", device_ < 0 ? DType::F32 : DType::F16,
                          {batch, samples, 9}, {max_batch_, samples, 9});
  if (device_ < 0) {
    constexpr float pi = 3.14159265358979323846F;
    const auto values = static_cast<std::size_t>(batch) * samples * 9;
    source_noise_cpu_.resize(values);
    const auto hash = [](std::uint32_t value) {
      value ^= value >> 16;
      value *= 0x7feb352dU;
      value ^= value >> 15;
      value *= 0x846ca68bU;
      return value ^ (value >> 16);
    };
    for (std::int32_t item = 0; item < batch; ++item) {
      for (std::int32_t sample = 0; sample < samples; ++sample) {
        for (std::int32_t harmonic = 0; harmonic < 9; ++harmonic) {
          const auto index = static_cast<std::size_t>(item) * samples * 9 +
                             sample * 9 + harmonic;
          const auto seed = static_cast<std::uint32_t>(
              seeds[item] + static_cast<std::uint64_t>(sample) * 9 + harmonic);
          const auto first = static_cast<float>(hash(seed) & 0xFFFFFF) /
                                 16777216.0F +
                             1e-7F;
          const auto second = static_cast<float>(
                                  hash(seed * 2654435761U + 1) & 0xFFFFFF) /
                              16777216.0F;
          source_noise_cpu_[index] =
              std::sqrt(-2.0F * std::log(first)) *
              std::cos(2.0F * pi * second);
        }
      }
    }
    std::memcpy(output.data(), source_noise_cpu_.data(), output.bytes);
    return output;
  }
#ifdef TINFER_CUDA
  styletts2::cuda::fill_noise(static_cast<__half*>(output.data()), seeds.data(),
                              batch, samples,
                              static_cast<cudaStream_t>(stream_));
  return output;
#else
  throw std::runtime_error("CUDA support is not enabled in this build");
#endif
}

}  // namespace tinfer::native
