#include "tinfer_rust/src/models/base/native.rs.h"
#include "tinfer_rust/src/models/base/cpp/tensor.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/cpu/glue.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/model.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/window.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace tinfer::native {
namespace {
const Buffer& need(const Tensors& tensors, std::string_view name, DType dtype) {
  const auto found = tensors.find(name);
  if (found == tensors.end() || found->second.dtype != dtype) {
    throw std::runtime_error("missing StyleTTS2 CPU tensor: " + std::string(name));
  }
  return found->second;
}

Tensors inputs(const Batch& batch) {
  Tensors result;
  for (const auto& input : batch.tensors) {
    validate_tensor(input);
    auto value = allocate(input.dtype, {input.shape.begin(), input.shape.end()}, -1);
    std::memcpy(value.data(), input.data.data(), input.data.size());
    result.emplace(std::string(input.name), std::move(value));
  }
  return result;
}

Tensors combine(const Tensors& weights, const Tensors& values) {
  auto result = weights;
  result.insert(values.begin(), values.end());
  return result;
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

Output StyleTts2Model::start_cpu(const Batch& batch) const {
  const auto values = inputs(batch);
  const auto& tokens = need(values, "tokens", DType::I64);
  const auto batch_size = static_cast<std::int32_t>(tokens.shape[0]);
  const auto token_count = static_cast<std::int32_t>(tokens.shape[1]);
  const auto* stream_ids = static_cast<const std::int64_t*>(need(values, "stream_ids", DType::I64).data());
  auto a = execution_a_->run(combine(weights_a_, values), nullptr);
  const auto alignment = styletts2::cpu::duration_prefix(
      static_cast<float*>(need(a, "dur", DType::F32).data()),
      static_cast<std::int32_t*>(need(values, "lengths", DType::I32).data()),
      static_cast<float*>(need(values, "speed", DType::F32).data()), batch_size,
      token_count);
  const auto& text = need(a, "t_en", DType::F32);
  const auto& encoded = need(a, "d", DType::F32);
  const auto channels = static_cast<std::int32_t>(encoded.shape[2]);
  const auto& styles = need(a, "s", DType::F32);
  const auto& references = need(a, "ref", DType::F32);
  const auto* seeds = static_cast<const std::uint64_t*>(need(values, "seeds", DType::I64).data());
  std::vector<float> state(static_cast<std::size_t>(batch_size) * 256);
  std::lock_guard lock(sessions_mutex_);
  for (std::int32_t item = 0; item < batch_size; ++item) {
    if (sessions_.contains(stream_ids[item])) {
      throw std::runtime_error("StyleTTS2 stream is already active");
    }
    StyleTts2Session session;
    const auto length = static_cast<std::int32_t*>(need(values, "lengths", DType::I32).data())[item];
    session.tokens = length;
    session.channels = channels;
    session.frames = alignment.totals[item];
    session.seed = seeds[item];
    session.text.resize(static_cast<std::size_t>(512) * length);
    session.encoding.resize(static_cast<std::size_t>(length) * channels);
    for (std::int32_t channel = 0; channel < 512; ++channel) {
      const auto* source = static_cast<float*>(text.data()) + (static_cast<std::size_t>(item) * 512 + channel) * token_count;
      std::copy_n(source, length, session.text.begin() + static_cast<std::size_t>(channel) * length);
    }
    const auto* encoding = static_cast<float*>(encoded.data()) + static_cast<std::size_t>(item) * token_count * channels;
    std::copy_n(encoding, static_cast<std::size_t>(length) * channels, session.encoding.begin());
    session.durations.assign(alignment.durations.begin() + static_cast<std::size_t>(item) * token_count,
                             alignment.durations.begin() + static_cast<std::size_t>(item) * token_count + length);
    session.starts.assign(alignment.starts.begin() + static_cast<std::size_t>(item) * token_count,
                          alignment.starts.begin() + static_cast<std::size_t>(item) * token_count + length);
    session.style.assign(static_cast<float*>(styles.data()) + static_cast<std::size_t>(item) * 128,
                         static_cast<float*>(styles.data()) + static_cast<std::size_t>(item + 1) * 128);
    session.reference.assign(static_cast<float*>(references.data()) + static_cast<std::size_t>(item) * 128,
                             static_cast<float*>(references.data()) + static_cast<std::size_t>(item + 1) * 128);
    if (const auto tail = tails_.find(stream_ids[item]); tail != tails_.end()) {
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
    sessions_.emplace(stream_ids[item], std::move(session));
    tails_.erase(stream_ids[item]);
  }
  Output output;
  output.tensors.push_back(tensor("durations", DType::I32, {batch_size, token_count}, alignment.durations));
  output.tensors.push_back(tensor("style", DType::F32, {batch_size, 256}, state));
  return output;
}

Output StyleTts2Model::continue_cpu(const Batch& batch) const {
  const auto values = inputs(batch);
  const auto& id_buffer = need(values, "stream_ids", DType::I64);
  const auto batch_size = static_cast<std::int32_t>(id_buffer.shape[0]);
  const auto* ids = static_cast<const std::int64_t*>(id_buffer.data());
  std::lock_guard lock(sessions_mutex_);
  const auto channels = sessions_.at(ids[0]).channels;
  std::vector<float> asr(static_cast<std::size_t>(batch_size) * 512 * kWindowFrames);
  std::vector<float> encoding(static_cast<std::size_t>(batch_size) * channels * kWindowFrames);
  std::vector<float> styles(static_cast<std::size_t>(batch_size) * 128);
  std::vector<float> references(static_cast<std::size_t>(batch_size) * 128);
  std::vector<std::int32_t> starts(batch_size), counts(batch_size), complete(batch_size);
  std::vector<std::uint64_t> seeds(batch_size);
  std::vector<float> phases(static_cast<std::size_t>(batch_size) * 9);
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
    std::copy(session.phase.begin(), session.phase.end(), phases.begin() + static_cast<std::size_t>(item) * 9);
  }
  auto en = allocate(DType::F32, {batch_size, channels, kWindowFrames}, -1);
  std::memcpy(en.data(), encoding.data(), en.bytes);
  auto style = allocate(DType::F32, {batch_size, 128}, -1);
  std::memcpy(style.data(), styles.data(), style.bytes);
  auto b_inputs = weights_b_;
  b_inputs.emplace("en", std::move(en));
  b_inputs.emplace("s", std::move(style));
  auto b = execution_b_->run(b_inputs, nullptr);
  auto harmonic = styletts2::cpu::source_to_har(
      static_cast<float*>(need(b, "f0", DType::F32).data()),
      static_cast<float*>(need(glue_weights_, "linW", DType::F32).data()),
      *static_cast<float*>(need(glue_weights_, "linB", DType::F32).data()), batch_size,
      kWindowFrames, seeds.data(), true, counts.data(), phases.data());
  auto asr_buffer = allocate(DType::F32, {batch_size, 512, kWindowFrames}, -1);
  std::memcpy(asr_buffer.data(), asr.data(), asr_buffer.bytes);
  auto har = allocate(DType::F32, {batch_size, 22, kWindowFrames * 120 + 1}, -1);
  std::memcpy(har.data(), harmonic.data(), har.bytes);
  auto c_inputs = weights_c_;
  c_inputs.emplace("asr", std::move(asr_buffer));
  c_inputs.emplace("f0", need(b, "f0", DType::F32));
  c_inputs.emplace("noise", need(b, "noise", DType::F32));
  auto reference = allocate(DType::F32, {batch_size, 128}, -1);
  std::memcpy(reference.data(), references.data(), reference.bytes);
  c_inputs.emplace("style", std::move(reference));
  c_inputs.emplace("har", std::move(har));
  auto c = execution_c_->run(c_inputs, nullptr);
  const auto* all_audio = static_cast<float*>(need(c, "audio", DType::F32).data());
  std::vector<float> audio;
  std::vector<std::int32_t> offsets{0};
  for (std::int32_t item = 0; item < batch_size; ++item) {
    const auto base = static_cast<std::size_t>(item) * kWindowFrames * 600 + kWindowPre * 600;
    audio.insert(audio.end(), all_audio + base, all_audio + base + counts[item] * 600);
    offsets.push_back(static_cast<std::int32_t>(audio.size()));
    auto& session = sessions_.at(ids[item]);
    session.cursor += counts[item];
    std::copy_n(phases.begin() + static_cast<std::size_t>(item) * 9, 9,
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

}  // namespace tinfer::native
