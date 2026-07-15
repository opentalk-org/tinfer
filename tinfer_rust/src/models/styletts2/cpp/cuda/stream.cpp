#include "tinfer_rust/src/models/base/native.rs.h"
#include "tinfer_rust/src/models/base/cpp/tensor.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/model.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/window.hpp"

#ifdef TINFER_CUDA
#include "tinfer_rust/src/models/styletts2/cpp/cuda/glue.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace tinfer::native {
#ifdef TINFER_CUDA
namespace {
const Tensor& host(const Batch& batch, std::string_view name, DType dtype) {
  const auto found = std::find_if(batch.tensors.begin(), batch.tensors.end(),
                                  [name](const Tensor& value) {
                                    return std::string_view(value.name.data(), value.name.size()) == name;
                                  });
  if (found == batch.tensors.end() || found->dtype != dtype) throw std::runtime_error("missing StyleTTS2 host tensor: " + std::string(name));
  return *found;
}

const Buffer& need(const Tensors& tensors, std::string_view name, DType dtype) {
  const auto found = tensors.find(name);
  if (found == tensors.end() || found->second.dtype != dtype) throw std::runtime_error("missing StyleTTS2 CUDA tensor: " + std::string(name));
  return found->second;
}

Buffer upload(const Tensor& source, DType dtype, std::int32_t device, cudaStream_t stream) {
  auto output = allocate(dtype, {source.shape.begin(), source.shape.end()}, device);
  if (source.dtype == dtype) {
    if (cudaMemcpyAsync(output.data(), source.data.data(), output.bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) throw std::runtime_error("StyleTTS2 CUDA upload failed");
  } else if (source.dtype == DType::F32 && dtype == DType::F16) {
    const auto* values = reinterpret_cast<const float*>(source.data.data());
    std::vector<__half> halves(source.data.size() / sizeof(float));
    std::transform(values, values + halves.size(), halves.begin(), [](float value) { return __float2half(value); });
    if (cudaMemcpyAsync(output.data(), halves.data(), output.bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) throw std::runtime_error("StyleTTS2 CUDA fp16 upload failed");
  } else {
    throw std::runtime_error("unsupported StyleTTS2 CUDA conversion");
  }
  return output;
}

Buffer upload_floats(const std::vector<float>& values, std::vector<std::int64_t> shape,
                     std::int32_t device, cudaStream_t stream) {
  auto output = allocate(DType::F16, std::move(shape), device);
  std::vector<__half> halves(values.size());
  std::transform(values.begin(), values.end(), halves.begin(), [](float value) { return __float2half(value); });
  if (cudaMemcpyAsync(output.data(), halves.data(), output.bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) throw std::runtime_error("StyleTTS2 window upload failed");
  return output;
}

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

Buffer pad_columns(const Buffer& source, std::int64_t columns, int fill,
                   std::int32_t device, cudaStream_t stream) {
  if (source.shape.size() != 2 || source.shape[1] >= columns) return source;
  const auto rows = source.shape[0];
  const auto source_pitch = static_cast<std::size_t>(source.shape[1]) * element_size(source.dtype);
  const auto target_pitch = static_cast<std::size_t>(columns) * element_size(source.dtype);
  auto output = allocate(source.dtype, {rows, columns}, device);
  if (cudaMemsetAsync(output.data(), fill, output.bytes, stream) != cudaSuccess ||
      cudaMemcpy2DAsync(output.data(), target_pitch, source.data(), source_pitch,
                        source_pitch, rows, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
    throw std::runtime_error("StyleTTS2 TensorRT input padding failed");
  }
  return output;
}

std::vector<float> floats(const Buffer& buffer, cudaStream_t stream) {
  std::vector<float> output(buffer.bytes / element_size(buffer.dtype));
  if (buffer.dtype == DType::F16) {
    std::vector<__half> halves(output.size());
    if (cudaMemcpyAsync(halves.data(), buffer.data(), buffer.bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) throw std::runtime_error("StyleTTS2 CUDA download failed");
    if (cudaStreamSynchronize(stream) != cudaSuccess) throw std::runtime_error("StyleTTS2 CUDA synchronization failed");
    std::transform(halves.begin(), halves.end(), output.begin(), [](__half value) { return __half2float(value); });
  } else {
    if (cudaMemcpyAsync(output.data(), buffer.data(), buffer.bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) throw std::runtime_error("StyleTTS2 CUDA download failed");
  }
  return output;
}

template <typename T>
Tensor tensor(const std::string& name, DType dtype, std::vector<std::int64_t> shape,
              const std::vector<T>& values) {
  Tensor result;
  result.name = name;
  result.dtype = dtype;
  for (const auto dimension : shape) result.shape.push_back(dimension);
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
  result.data.reserve(values.size() * sizeof(T));
  for (std::size_t index = 0; index < values.size() * sizeof(T); ++index) result.data.push_back(bytes[index]);
  return result;
}
}  // namespace

Output StyleTts2Model::start_cuda(const Batch& batch) const {
  if (cudaSetDevice(device_) != cudaSuccess) throw std::runtime_error("cannot select StyleTTS2 CUDA device");
  const auto stream = static_cast<cudaStream_t>(stream_);
  Tensors inputs;
  for (const auto& value : batch.tensors) {
    validate_tensor(value);
    inputs.emplace(std::string(value.name), upload(value, execution_a_->input_dtype(std::string(value.name)).value_or(value.dtype), device_, stream));
  }
  const auto actual_tokens = inputs.at("tokens").shape[1];
  if (backend_ == Backend::TensorRt && actual_tokens < 8) {
    inputs["tokens"] = pad_columns(inputs.at("tokens"), 8, 0, device_, stream);
    inputs["mask"] = pad_columns(inputs.at("mask"), 8, 1, device_, stream);
  }
  auto a_inputs = weights_a_;
  a_inputs.insert(inputs.begin(), inputs.end());
  auto a = execution_a_->run(a_inputs, stream_);
  const auto& token_tensor = host(batch, "tokens", DType::I64);
  const auto batch_size = static_cast<std::int32_t>(token_tensor.shape[0]);
  const auto token_count = static_cast<std::int32_t>(token_tensor.shape[1]);
  const auto* lengths = reinterpret_cast<const std::int32_t*>(host(batch, "lengths", DType::I32).data.data());
  const auto* speeds = reinterpret_cast<const float*>(host(batch, "speed", DType::F32).data.data());
  const auto* ids = reinterpret_cast<const std::int64_t*>(host(batch, "stream_ids", DType::I64).data.data());
  const auto* seeds = reinterpret_cast<const std::uint64_t*>(host(batch, "seeds", DType::I64).data.data());
  const auto duration_values = floats(need(a, "dur", DType::F16), stream);
  const auto text = floats(need(a, "t_en", DType::F16), stream);
  const auto encoded = floats(need(a, "d", DType::F16), stream);
  const auto styles = floats(need(a, "s", DType::F16), stream);
  const auto references = floats(need(a, "ref", DType::F16), stream);
  const auto channels = static_cast<std::int32_t>(need(a, "d", DType::F16).shape[2]);
  const auto execution_tokens = static_cast<std::int32_t>(need(a, "dur", DType::F16).shape[1]);
  std::vector<std::int32_t> durations(static_cast<std::size_t>(batch_size) * token_count);
  std::vector<float> state(static_cast<std::size_t>(batch_size) * 256);
  std::lock_guard lock(sessions_mutex_);
  for (std::int32_t item = 0; item < batch_size; ++item) {
    if (sessions_.contains(ids[item])) {
      throw std::runtime_error("StyleTTS2 stream is already active");
    }
    StyleTts2Session session;
    session.tokens = lengths[item];
    session.channels = channels;
    session.seed = seeds[item];
    session.text.resize(static_cast<std::size_t>(512) * session.tokens);
    session.encoding.resize(static_cast<std::size_t>(session.tokens) * channels);
    for (std::int32_t channel = 0; channel < 512; ++channel) {
      std::copy_n(text.begin() + (static_cast<std::size_t>(item) * 512 + channel) * execution_tokens,
                  session.tokens, session.text.begin() + static_cast<std::size_t>(channel) * session.tokens);
    }
    std::copy_n(encoded.begin() + static_cast<std::size_t>(item) * execution_tokens * channels,
                static_cast<std::size_t>(session.tokens) * channels, session.encoding.begin());
    std::int32_t total = 0;
    for (std::int32_t token = 0; token < session.tokens; ++token) {
      const auto duration = std::max(1, static_cast<std::int32_t>(std::lround(
          duration_values[static_cast<std::size_t>(item) * execution_tokens + token] /
          speeds[item])));
      durations[static_cast<std::size_t>(item) * token_count + token] = duration;
      session.starts.push_back(total);
      session.durations.push_back(duration);
      total += duration;
    }
    session.frames = total;
    session.style.assign(styles.begin() + static_cast<std::size_t>(item) * 128, styles.begin() + static_cast<std::size_t>(item + 1) * 128);
    session.reference.assign(references.begin() + static_cast<std::size_t>(item) * 128, references.begin() + static_cast<std::size_t>(item + 1) * 128);
    if (const auto tail = tails_.find(ids[item]); tail != tails_.end()) {
      if (tail->second.channels != channels) {
        throw std::runtime_error("StyleTTS2 retained context has incompatible channels");
      }
      session.tail_text = tail->second.text;
      session.tail_encoding = tail->second.encoding;
      session.tail_frames = tail->second.frames;
      session.phase = tail->second.phase;
      session.seed = tail->second.seed;
      session.frame_offset = tail->second.frame_offset;
    }
    std::copy(session.reference.begin(), session.reference.end(), state.begin() + static_cast<std::size_t>(item) * 256);
    std::copy(session.style.begin(), session.style.end(), state.begin() + static_cast<std::size_t>(item) * 256 + 128);
    sessions_.emplace(ids[item], std::move(session));
    tails_.erase(ids[item]);
  }
  Output output;
  output.tensors.push_back(tensor("durations", DType::I32, {batch_size, token_count}, durations));
  output.tensors.push_back(tensor("style", DType::F32, {batch_size, 256}, state));
  return output;
}

Output StyleTts2Model::continue_cuda(const Batch& batch) const {
  if (cudaSetDevice(device_) != cudaSuccess) throw std::runtime_error("cannot select StyleTTS2 CUDA device");
  const auto stream = static_cast<cudaStream_t>(stream_);
  const auto& ids_tensor = host(batch, "stream_ids", DType::I64);
  const auto* ids = reinterpret_cast<const std::int64_t*>(ids_tensor.data.data());
  const auto batch_size = static_cast<std::int32_t>(ids_tensor.shape[0]);
  std::lock_guard lock(sessions_mutex_);
  const auto channels = sessions_.at(ids[0]).channels;
  std::vector<float> asr(static_cast<std::size_t>(batch_size) * 512 * kWindowFrames);
  std::vector<float> encoding(static_cast<std::size_t>(batch_size) * channels * kWindowFrames);
  std::vector<float> styles(static_cast<std::size_t>(batch_size) * 128), references(styles.size());
  std::vector<std::int32_t> starts(batch_size), counts(batch_size), complete(batch_size);
  std::vector<std::uint64_t> seeds(batch_size);
  std::vector<float> phase_values(static_cast<std::size_t>(batch_size) * 9);
  for (std::int32_t item = 0; item < batch_size; ++item) {
    auto& session = sessions_.at(ids[item]);
    starts[item] = session.cursor;
    counts[item] = std::min(kWindowCore, session.frames - session.cursor);
    complete[item] = session.cursor + counts[item] == session.frames;
    seeds[item] = session.seed + (session.frame_offset + session.cursor) * 600;
    styletts2::expand_window(session, asr.data() + static_cast<std::size_t>(item) * 512 * kWindowFrames,
                            encoding.data() + static_cast<std::size_t>(item) * channels * kWindowFrames);
    std::copy(session.style.begin(), session.style.end(), styles.begin() + static_cast<std::size_t>(item) * 128);
    std::copy(session.reference.begin(), session.reference.end(), references.begin() + static_cast<std::size_t>(item) * 128);
    std::copy(session.phase.begin(), session.phase.end(), phase_values.begin() + static_cast<std::size_t>(item) * 9);
  }
  auto b_inputs = weights_b_;
  b_inputs.emplace("en", upload_floats(encoding, {batch_size, channels, kWindowFrames}, device_, stream));
  b_inputs.emplace("s", upload_floats(styles, {batch_size, 128}, device_, stream));
  auto b = execution_b_->run(b_inputs, stream_);
  auto phase_frames = allocate(DType::F32, {batch_size, 9, kWindowFrames * 2}, device_);
  auto source = allocate(DType::F32, {batch_size, kWindowFrames * 600}, device_);
  auto harmonic = allocate(DType::F16, {batch_size, 22, kWindowFrames * 120 + 1}, device_);
  auto seed_buffer = allocate(DType::I64, {batch_size}, device_);
  auto phase_state = upload_values(phase_values, DType::F32, {batch_size, 9}, device_, stream);
  auto advances = upload_values(counts, DType::I32, {batch_size}, device_, stream);
  if (cudaMemcpyAsync(seed_buffer.data(), seeds.data(), seed_buffer.bytes,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    throw std::runtime_error("StyleTTS2 seed upload failed");
  }
  styletts2::cuda::source_to_har(static_cast<__half*>(need(b, "f0", DType::F16).data()),
      static_cast<__half*>(need(glue_weights_, "linW", DType::F16).data()), static_cast<__half*>(need(glue_weights_, "linB", DType::F16).data()),
      static_cast<std::uint64_t*>(seed_buffer.data()), static_cast<__half*>(harmonic.data()), static_cast<float*>(phase_frames.data()),
      static_cast<float*>(phase_state.data()), static_cast<std::int32_t*>(advances.data()),
      static_cast<float*>(source.data()), batch_size, kWindowFrames, true, stream);
  auto c_inputs = weights_c_;
  c_inputs.emplace("asr", upload_floats(asr, {batch_size, 512, kWindowFrames}, device_, stream));
  c_inputs.emplace("f0", need(b, "f0", DType::F16));
  c_inputs.emplace("noise", need(b, "noise", DType::F16));
  c_inputs.emplace("style", upload_floats(references, {batch_size, 128}, device_, stream));
  c_inputs.emplace("har", std::move(harmonic));
  auto c = execution_c_->run(c_inputs, stream_);
  if (cudaMemcpyAsync(phase_values.data(), phase_state.data(), phase_state.bytes,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error("StyleTTS2 phase download failed");
  }
  const auto all_audio = floats(need(c, "audio", DType::F16), stream);
  std::vector<float> audio;
  std::vector<std::int32_t> offsets{0};
  for (std::int32_t item = 0; item < batch_size; ++item) {
    const auto base = static_cast<std::size_t>(item) * kWindowFrames * 600 + kWindowPre * 600;
    audio.insert(audio.end(), all_audio.begin() + base, all_audio.begin() + base + counts[item] * 600);
    offsets.push_back(static_cast<std::int32_t>(audio.size()));
    auto& session = sessions_.at(ids[item]);
    session.cursor += counts[item];
    std::copy_n(phase_values.begin() + static_cast<std::size_t>(item) * 9, 9,
                session.phase.begin());
    if (complete[item]) {
      tails_.insert_or_assign(ids[item], styletts2::retain_tail(session));
      sessions_.erase(ids[item]);
    }
  }
  Output output;
  output.tensors.push_back(tensor("audio", DType::F32, {static_cast<std::int64_t>(audio.size())}, audio));
  output.tensors.push_back(tensor("audio_offsets", DType::I32, {batch_size + 1}, offsets));
  output.tensors.push_back(tensor("frame_starts", DType::I32, {batch_size}, starts));
  output.tensors.push_back(tensor("frame_counts", DType::I32, {batch_size}, counts));
  output.tensors.push_back(tensor("complete", DType::I32, {batch_size}, complete));
  return output;
}
#else
Output StyleTts2Model::start_cuda(const Batch&) const { throw std::runtime_error("CUDA support is not enabled in this build"); }
Output StyleTts2Model::continue_cuda(const Batch&) const { throw std::runtime_error("CUDA support is not enabled in this build"); }
#endif
}  // namespace tinfer::native
