#include "tinfer_rust/src/models/base/native.rs.h"
#include "tinfer_rust/src/models/styletts2/cpp/cpu/glue.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/cuda/glue.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/model.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace tinfer::native {
namespace {
const Tensor& host(const Batch& batch, std::string_view name, DType dtype) {
  const auto found = std::find_if(batch.tensors.begin(), batch.tensors.end(),
      [name](const Tensor& value) { return std::string_view(value.name.data(), value.name.size()) == name; });
  if (found == batch.tensors.end() || found->dtype != dtype) {
    throw std::runtime_error("missing StyleTTS2 request tensor: " + std::string(name));
  }
  return *found;
}

const Buffer& need(const Tensors& tensors, std::string_view name) {
  const auto found = tensors.find(name);
  if (found == tensors.end()) throw std::runtime_error("missing StyleTTS2 inference tensor: " + std::string(name));
  return found->second;
}

Tensors combine(const Tensors& weights, const Tensors& values) {
  auto result = weights;
  result.insert(values.begin(), values.end());
  return result;
}

Buffer device_copy(const Buffer& source, std::int32_t device, cudaStream_t stream) {
  auto output = allocate(source.dtype, source.shape, device);
  if (cudaMemcpyAsync(output.data(), source.data(), source.bytes,
                      cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
    throw std::runtime_error("StyleTTS2 session copy failed");
  }
  return output;
}

Buffer slice(const Buffer& source, std::size_t offset, DType dtype,
             std::vector<std::int64_t> shape, std::size_t bytes) {
  auto* pointer = static_cast<std::uint8_t*>(source.data()) + offset;
  return Buffer{dtype, std::move(shape), std::shared_ptr<void>(source.memory, pointer), bytes};
}

template <typename T>
Tensor tensor(const std::string& name, DType dtype, std::vector<std::int64_t> shape,
              const std::vector<T>& values) {
  Tensor result;
  result.name = name;
  result.dtype = dtype;
  for (const auto dimension : shape) result.shape.push_back(dimension);
  const auto size = values.size() * sizeof(T);
  result.data = copy_native_bytes(rust::Slice<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(values.data()), size));
  return result;
}
}  // namespace

Output StyleTts2Model::start_cuda(const Batch& batch) const {
  const auto& token_tensor = host(batch, "tokens", DType::I64);
  const auto batch_size = static_cast<std::int32_t>(token_tensor.shape[0]);
  auto values = upload(batch, *execution_a_);
  const auto token_count = static_cast<std::int32_t>(token_tensor.shape[1]);
  if (token_count < 8) {
    values["tokens"] = pad_columns("tokens", values.at("tokens"), 8, 0);
    values["mask"] = pad_columns("mask", values.at("mask"), 8, 1);
  }
  const auto a = execution_a_->run(combine(weights_a_, values), stream_);
  const auto& text_output = need(a, "t_en");
  const auto& encoding_output = need(a, "d");
  const auto& style_output = need(a, "s");
  const auto& reference_output = need(a, "ref");
  if (text_output.dtype != DType::F16 || encoding_output.dtype != DType::F16 ||
      style_output.dtype != DType::F16 || reference_output.dtype != DType::F16) {
    throw std::runtime_error("StyleTTS2 TensorRT session outputs must be fp16");
  }
  const auto durations_output = download_floats(need(a, "dur"));
  const auto execution_tokens = static_cast<std::int32_t>(need(a, "dur").shape[1]);
  const auto channels = static_cast<std::int32_t>(encoding_output.shape[2]);
  const auto* lengths = reinterpret_cast<const std::int32_t*>(host(batch, "lengths", DType::I32).data.data());
  const auto* speeds = reinterpret_cast<const float*>(host(batch, "speed", DType::F32).data.data());
  const auto* ids = reinterpret_cast<const std::int64_t*>(host(batch, "stream_ids", DType::I64).data.data());
  const auto* seeds = reinterpret_cast<const std::uint64_t*>(host(batch, "seeds", DType::I64).data.data());
  const auto alignment = styletts2::cpu::duration_prefix(
      durations_output.data(), lengths, speeds, batch_size, execution_tokens);
  const auto stream = static_cast<cudaStream_t>(stream_);
  const auto texts = device_copy(text_output, device_, stream);
  const auto encodings = device_copy(encoding_output, device_, stream);
  const auto device_styles = device_copy(style_output, device_, stream);
  const auto device_references = device_copy(reference_output, device_, stream);
  auto device_starts = allocate(DType::I32, {batch_size, execution_tokens}, device_);
  const auto starts_bytes = alignment.starts.size() * sizeof(std::int32_t);
  auto* starts_staging = pinned("session.starts", starts_bytes,
      static_cast<std::size_t>(max_batch_) * execution_tokens * sizeof(std::int32_t));
  std::memcpy(starts_staging, alignment.starts.data(), starts_bytes);
  cudaMemcpyAsync(device_starts.data(), starts_staging, starts_bytes, cudaMemcpyHostToDevice, stream);
  const auto styles = download_floats(style_output);
  const auto references = download_floats(reference_output);
  const auto text_stride = static_cast<std::size_t>(512) * execution_tokens * sizeof(__half);
  const auto encoding_stride = static_cast<std::size_t>(execution_tokens) * channels * sizeof(__half);
  const auto state_stride = static_cast<std::size_t>(128) * sizeof(__half);
  const auto starts_stride = static_cast<std::size_t>(execution_tokens) * sizeof(std::int32_t);
  std::vector<std::int32_t> durations(static_cast<std::size_t>(batch_size) * token_count);
  std::vector<float> state(static_cast<std::size_t>(batch_size) * 256);
  std::lock_guard lock(sessions_mutex_);
  for (std::int32_t item = 0; item < batch_size; ++item) {
    if (sessions_.contains(ids[item])) throw std::runtime_error("StyleTTS2 stream is already active");
    StyleTts2Session session;
    session.tokens = lengths[item];
    session.channels = channels;
    session.frames = alignment.totals[item];
    session.seed = seeds[item];
    session.durations.assign(alignment.durations.begin() + item * execution_tokens,
                             alignment.durations.begin() + item * execution_tokens + session.tokens);
    session.starts.assign(alignment.starts.begin() + item * execution_tokens,
                          alignment.starts.begin() + item * execution_tokens + session.tokens);
    std::copy_n(session.durations.begin(), session.tokens, durations.begin() + item * token_count);
    session.device_text = slice(texts, item * text_stride, DType::F16, {512, execution_tokens}, text_stride);
    session.device_encoding = slice(encodings, item * encoding_stride, DType::F16, {execution_tokens, channels}, encoding_stride);
    session.device_starts = slice(device_starts, item * starts_stride, DType::I32, {execution_tokens}, starts_stride);
    session.device_style = slice(device_styles, item * state_stride, DType::F16, {128}, state_stride);
    session.device_reference = slice(device_references, item * state_stride, DType::F16, {128}, state_stride);
    if (const auto tail = tails_.find(ids[item]); tail != tails_.end()) {
      session.device_tail_text = tail->second.device_text;
      session.device_tail_encoding = tail->second.device_encoding;
      session.tail_frames = tail->second.frames;
      session.phase = tail->second.phase;
      session.seed = tail->second.seed;
      session.frame_offset = tail->second.frame_offset;
    }
    std::copy_n(references.begin() + item * 128, 128, state.begin() + item * 256);
    std::copy_n(styles.begin() + item * 128, 128, state.begin() + item * 256 + 128);
    sessions_.emplace(ids[item], std::move(session));
    tails_.erase(ids[item]);
  }
  Output output;
  output.tensors.push_back(tensor("durations", DType::I32, {batch_size, token_count}, durations));
  output.tensors.push_back(tensor("style", DType::F32, {batch_size, 256}, state));
  return output;
}

Output StyleTts2Model::continue_cuda(const Batch& batch) const {
  const auto& id_tensor = host(batch, "stream_ids", DType::I64);
  const auto* ids = reinterpret_cast<const std::int64_t*>(id_tensor.data.data());
  const auto batch_size = static_cast<std::int32_t>(id_tensor.shape[0]);
  std::lock_guard lock(sessions_mutex_);
  std::vector<StyleTts2Session*> sessions(batch_size);
  std::vector<styletts2::cuda::WindowSource> sources(batch_size);
  std::vector<float> phases(static_cast<std::size_t>(batch_size) * 9);
  std::vector<std::uint64_t> seeds(batch_size);
  std::vector<std::int32_t> starts(batch_size), counts(batch_size), complete(batch_size);
  const auto channels = sessions_.at(ids[0]).channels;
  for (std::int32_t item = 0; item < batch_size; ++item) {
    auto& session = sessions_.at(ids[item]);
    sessions[item] = &session;
    starts[item] = session.cursor;
    counts[item] = std::min(kWindowCore, session.frames - session.cursor);
    complete[item] = session.cursor + counts[item] == session.frames;
    std::copy(session.phase.begin(), session.phase.end(), phases.begin() + item * 9);
    const auto sample = static_cast<std::int64_t>(session.frame_offset + session.cursor - kWindowPre) * 600;
    seeds[item] = session.seed + static_cast<std::uint64_t>(sample * 9);
    sources[item] = {
        static_cast<const __half*>(session.device_text->data()), static_cast<const __half*>(session.device_encoding->data()),
        static_cast<const std::int32_t*>(session.device_starts->data()), static_cast<const __half*>(session.device_style->data()),
        static_cast<const __half*>(session.device_reference->data()),
        session.device_tail_text ? static_cast<const __half*>(session.device_tail_text->data()) : nullptr,
        session.device_tail_encoding ? static_cast<const __half*>(session.device_tail_encoding->data()) : nullptr,
        session.tokens, static_cast<std::int32_t>(session.device_text->shape[1]),
        channels, session.frames, session.cursor, session.tail_frames};
  }
  static_assert(sizeof(styletts2::cuda::WindowSource) % sizeof(std::int64_t) == 0);
  const auto descriptor_bytes = sources.size() * sizeof(sources.front());
  auto descriptors = workspace("bc.sources", DType::I64,
      {batch_size, static_cast<std::int64_t>(sizeof(sources.front()) / sizeof(std::int64_t))},
      {max_batch_, static_cast<std::int64_t>(sizeof(sources.front()) / sizeof(std::int64_t))});
  auto* descriptor_staging = pinned("bc.sources", descriptor_bytes,
      static_cast<std::size_t>(max_batch_) * sizeof(sources.front()));
  std::memcpy(descriptor_staging, sources.data(), descriptor_bytes);
  const auto stream = static_cast<cudaStream_t>(stream_);
  cudaMemcpyAsync(descriptors.data(), descriptor_staging, descriptor_bytes, cudaMemcpyHostToDevice, stream);
  auto asr = workspace("bc.asr", DType::F16, {batch_size, 512, kWindowFrames}, {max_batch_, 512, kWindowFrames});
  auto encoding = workspace("bc.en", DType::F16, {batch_size, channels, kWindowFrames}, {max_batch_, channels, kWindowFrames});
  auto styles = workspace("bc.s", DType::F16, {batch_size, 128}, {max_batch_, 128});
  auto references = workspace("bc.ref", DType::F16, {batch_size, 128}, {max_batch_, 128});
  styletts2::cuda::expand_windows(static_cast<const styletts2::cuda::WindowSource*>(descriptors.data()),
      static_cast<__half*>(asr.data()), static_cast<__half*>(encoding.data()),
      static_cast<__half*>(styles.data()), static_cast<__half*>(references.data()),
      batch_size, channels, kWindowFrames, stream);
  auto inputs = weights_bc_;
  inputs.emplace("en", encoding);
  inputs.emplace("asr", asr);
  inputs.emplace("s", styles);
  inputs.emplace("ref", references);
  inputs.emplace("phase", upload_floats("bc.phase", phases, {batch_size, 9}, DType::F32));
  inputs.emplace("source_noise", source_noise(seeds));
  const auto output = execution_bc_->run(inputs, stream_);
  const auto all_audio = download_floats(need(output, "audio"));
  const auto next_phases = download_floats(need(output, "next_phase"));
  const auto completed = std::count(complete.begin(), complete.end(), 1);
  Buffer tail_texts, tail_encodings;
  if (completed > 0) {
    tail_texts = allocate(DType::F16, {completed, 512, kWindowPre}, device_);
    tail_encodings = allocate(DType::F16, {completed, channels, kWindowPre}, device_);
    cudaMemsetAsync(tail_texts.data(), 0, tail_texts.bytes, stream);
    cudaMemsetAsync(tail_encodings.data(), 0, tail_encodings.bytes, stream);
  }
  std::vector<float> audio;
  audio.reserve(static_cast<std::size_t>(batch_size) * kWindowCore * 600);
  std::vector<std::int32_t> offsets{0};
  std::int32_t tail_item = 0;
  for (std::int32_t item = 0; item < batch_size; ++item) {
    const auto base = static_cast<std::size_t>(item) * kWindowCore * 600;
    audio.insert(audio.end(), all_audio.begin() + base, all_audio.begin() + base + counts[item] * 600);
    offsets.push_back(static_cast<std::int32_t>(audio.size()));
    auto& session = *sessions[item];
    std::copy_n(next_phases.begin() + item * 9, 9, session.phase.begin());
    session.cursor += counts[item];
    if (complete[item] != 0) {
      const auto tail_frames = std::min(kWindowPre, session.tail_frames + session.frames);
      const auto source_frame = kWindowPre + counts[item] - tail_frames;
      auto* text_destination = static_cast<__half*>(tail_texts.data()) + static_cast<std::size_t>(tail_item) * 512 * kWindowPre + kWindowPre - tail_frames;
      auto* encoding_destination = static_cast<__half*>(tail_encodings.data()) + static_cast<std::size_t>(tail_item) * channels * kWindowPre + kWindowPre - tail_frames;
      cudaMemcpy2DAsync(text_destination, kWindowPre * sizeof(__half),
          static_cast<const __half*>(asr.data()) + static_cast<std::size_t>(item) * 512 * kWindowFrames + source_frame,
          kWindowFrames * sizeof(__half), tail_frames * sizeof(__half), 512, cudaMemcpyDeviceToDevice, stream);
      cudaMemcpy2DAsync(encoding_destination, kWindowPre * sizeof(__half),
          static_cast<const __half*>(encoding.data()) + static_cast<std::size_t>(item) * channels * kWindowFrames + source_frame,
          kWindowFrames * sizeof(__half), tail_frames * sizeof(__half), channels, cudaMemcpyDeviceToDevice, stream);
      StyleTts2Tail tail;
      tail.device_text = slice(tail_texts, static_cast<std::size_t>(tail_item) * 512 * kWindowPre * sizeof(__half), DType::F16, {512, kWindowPre}, 512 * kWindowPre * sizeof(__half));
      tail.device_encoding = slice(tail_encodings, static_cast<std::size_t>(tail_item) * channels * kWindowPre * sizeof(__half), DType::F16, {channels, kWindowPre}, channels * kWindowPre * sizeof(__half));
      tail.channels = channels;
      tail.frames = tail_frames;
      tail.phase = session.phase;
      tail.seed = session.seed;
      tail.frame_offset = session.frame_offset + session.frames;
      tails_.insert_or_assign(ids[item], std::move(tail));
      sessions_.erase(ids[item]);
      ++tail_item;
    }
  }
  Output result;
  result.tensors.push_back(tensor("audio", DType::F32, {static_cast<std::int64_t>(audio.size())}, audio));
  result.tensors.push_back(tensor("audio_offsets", DType::I32, {batch_size + 1}, offsets));
  result.tensors.push_back(tensor("frame_starts", DType::I32, {batch_size}, starts));
  result.tensors.push_back(tensor("frame_counts", DType::I32, {batch_size}, counts));
  result.tensors.push_back(tensor("complete", DType::I32, {batch_size}, complete));
  return result;
}
}  // namespace tinfer::native
