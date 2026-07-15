#include "tinfer_rust/src/models/base/native.rs.h"
#include "tinfer_rust/src/models/styletts2/cpp/cpu/glue.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/model.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/window.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace tinfer::native {
  namespace {
    const Tensor &host(const Batch &batch, std::string_view name, DType dtype)
    {
      const auto found = std::find_if(batch.tensors.begin(), batch.tensors.end(),
          [name](const Tensor &value) {
            return std::string_view(value.name.data(), value.name.size()) == name;
          });
      if (found == batch.tensors.end() || found->dtype != dtype)
      {
        throw std::runtime_error("missing StyleTTS2 request tensor: " +
                                 std::string(name));
      }
      return *found;
    }

    const Buffer &need(const Tensors &tensors, std::string_view name)
    {
      const auto found = tensors.find(name);
      if (found == tensors.end()) {
        throw std::runtime_error("missing StyleTTS2 inference tensor: " +
                                 std::string(name));
      }
      return found->second;
    }

    Tensors combine(const Tensors &weights, const Tensors &values)
    {
      auto result = weights;
      result.insert(values.begin(), values.end());
      return result;
    }

    template <typename T>
    Tensor tensor(const std::string &name, DType dtype,
                  std::vector<std::int64_t> shape,
                  const std::vector<T> &values)
    {
      Tensor result;
      result.name = name;
      result.dtype = dtype;
      for (const auto dimension : shape) result.shape.push_back(dimension);
      const auto *bytes = reinterpret_cast<const std::uint8_t *>(values.data());
      const auto size = values.size() * sizeof(T);
      result.data = copy_native_bytes(rust::Slice<const std::uint8_t>(bytes, size));
      return result;
    }

  } // namespace

  Output StyleTts2Model::start(const Batch &batch) const
  {
#ifdef TINFER_CUDA
    if (device_ >= 0 && backend_ == Backend::TensorRt) return start_cuda(batch);
#endif
    const auto &token_tensor = host(batch, "tokens", DType::I64);
    const auto batch_size = static_cast<std::int32_t>(token_tensor.shape[0]);
    auto values = upload(batch, *execution_a_);
    const auto token_count = static_cast<std::int32_t>(token_tensor.shape[1]);
    if (backend_ == Backend::TensorRt && token_count < 8)
    {
      values["tokens"] = pad_columns("tokens", values.at("tokens"), 8, 0);
      values["mask"] = pad_columns("mask", values.at("mask"), 8, 1);
    }
    const auto a = execution_a_->run(combine(weights_a_, values), stream_);
    const auto durations_output = download_floats(need(a, "dur"));
    const auto text = download_floats(need(a, "t_en"));
    const auto encoded = download_floats(need(a, "d"));
    const auto styles = download_floats(need(a, "s"));
    const auto references = download_floats(need(a, "ref"));
    const auto channels = static_cast<std::int32_t>(need(a, "d").shape[2]);
    const auto execution_tokens =
        static_cast<std::int32_t>(need(a, "dur").shape[1]);
    const auto *lengths = reinterpret_cast<const std::int32_t *>(
        host(batch, "lengths", DType::I32).data.data());
    const auto *speeds = reinterpret_cast<const float *>(
        host(batch, "speed", DType::F32).data.data());
    const auto *ids = reinterpret_cast<const std::int64_t *>(
        host(batch, "stream_ids", DType::I64).data.data());
    const auto *seeds = reinterpret_cast<const std::uint64_t *>(
        host(batch, "seeds", DType::I64).data.data());
    const auto alignment = styletts2::cpu::duration_prefix(
        durations_output.data(), lengths, speeds, batch_size, execution_tokens);
    std::vector<std::int32_t> durations(
        static_cast<std::size_t>(batch_size) * token_count);
    std::vector<float> state(static_cast<std::size_t>(batch_size) * 256);
    std::lock_guard lock(sessions_mutex_);
    for (std::int32_t item = 0; item < batch_size; ++item)
    {
      if (sessions_.contains(ids[item]))
      {
        throw std::runtime_error("StyleTTS2 stream is already active");
      }
      StyleTts2Session session;
      session.tokens = lengths[item];
      session.channels = channels;
      session.frames = alignment.totals[item];
      session.seed = seeds[item];
      session.text.resize(static_cast<std::size_t>(512) * session.tokens);
      session.encoding.resize(static_cast<std::size_t>(session.tokens) * channels);
      for (std::int32_t channel = 0; channel < 512; ++channel)
      {
        const auto source = (static_cast<std::size_t>(item) * 512 + channel) *
                            execution_tokens;
        std::copy_n(text.begin() + source, session.tokens,
                    session.text.begin() +
                        static_cast<std::size_t>(channel) * session.tokens);
      }
      const auto encoding_source =
          static_cast<std::size_t>(item) * execution_tokens * channels;
      std::copy_n(encoded.begin() + encoding_source,
                  static_cast<std::size_t>(session.tokens) * channels,
                  session.encoding.begin());
      const auto duration_source = static_cast<std::size_t>(item) *
                                   execution_tokens;
      session.durations.assign(alignment.durations.begin() + duration_source,
                               alignment.durations.begin() + duration_source +
                                   session.tokens);
      session.starts.assign(alignment.starts.begin() + duration_source,
                            alignment.starts.begin() + duration_source +
                                session.tokens);
      std::copy_n(session.durations.begin(), session.tokens,
                  durations.begin() + static_cast<std::size_t>(item) *
                                          token_count);
      session.style.assign(styles.begin() + static_cast<std::size_t>(item) * 128,
                           styles.begin() + static_cast<std::size_t>(item + 1) * 128);
      session.reference.assign(
          references.begin() + static_cast<std::size_t>(item) * 128,
          references.begin() + static_cast<std::size_t>(item + 1) * 128);
      if (const auto tail = tails_.find(ids[item]); tail != tails_.end())
      {
        if (tail->second.channels != channels)
        {
          throw std::runtime_error(
              "StyleTTS2 retained context has incompatible channels");
        }
        session.tail_text = tail->second.text;
        session.tail_encoding = tail->second.encoding;
        session.tail_frames = tail->second.frames;
        session.phase = tail->second.phase;
        session.seed = tail->second.seed;
        session.frame_offset = tail->second.frame_offset;
      }
      std::copy(session.reference.begin(), session.reference.end(),
                state.begin() + static_cast<std::size_t>(item) * 256);
      std::copy(session.style.begin(), session.style.end(),
                state.begin() + static_cast<std::size_t>(item) * 256 + 128);
      sessions_.emplace(ids[item], std::move(session));
      tails_.erase(ids[item]);
    }
    Output output;
    output.tensors.push_back(tensor("durations", DType::I32,
                                    {batch_size, token_count}, durations));
    output.tensors.push_back(
        tensor("style", DType::F32, {batch_size, 256}, state));
    return output;
  }

  Output StyleTts2Model::continue_generation(const Batch &batch) const
  {
#ifdef TINFER_CUDA
    if (device_ >= 0 && backend_ == Backend::TensorRt) return continue_cuda(batch);
#endif
    const auto &id_tensor = host(batch, "stream_ids", DType::I64);
    const auto *ids = reinterpret_cast<const std::int64_t *>(id_tensor.data.data());
    const auto batch_size = static_cast<std::int32_t>(id_tensor.shape[0]);
    std::lock_guard lock(sessions_mutex_);
    std::vector<StyleTts2Session *> sessions(batch_size);
    for (std::int32_t item = 0; item < batch_size; ++item)
    {
      auto &session = sessions_.at(ids[item]);
      sessions[item] = &session;
    }

    std::vector<float> asr(static_cast<std::size_t>(batch_size) * 512 *
                           kWindowFrames);
    const auto channels = sessions.front()->channels;
    std::vector<float> encoding(static_cast<std::size_t>(batch_size) *
                                channels * kWindowFrames);
    std::vector<float> styles(static_cast<std::size_t>(batch_size) * 128);
    std::vector<float> references(static_cast<std::size_t>(batch_size) * 128);
    std::vector<float> phases(static_cast<std::size_t>(batch_size) * 9);
    std::vector<std::uint64_t> seeds(batch_size);
    std::vector<std::int32_t> starts(batch_size), counts(batch_size),
        complete(batch_size);
    for (std::int32_t item = 0; item < batch_size; ++item)
    {
      auto &session = *sessions[item];
      starts[item] = session.cursor;
      counts[item] = std::min(kWindowCore, session.frames - session.cursor);
      complete[item] = session.cursor + counts[item] == session.frames;
      const auto window_start = session.cursor - kWindowPre;
      styletts2::expand_window_at(
          session, window_start,
          asr.data() + static_cast<std::size_t>(item) * 512 * kWindowFrames,
          encoding.data() + static_cast<std::size_t>(item) * channels *
                                kWindowFrames);
      std::copy(session.style.begin(), session.style.end(),
                styles.begin() + static_cast<std::size_t>(item) * 128);
      std::copy(session.reference.begin(), session.reference.end(),
                references.begin() + static_cast<std::size_t>(item) * 128);
      std::copy(session.phase.begin(), session.phase.end(),
                phases.begin() + static_cast<std::size_t>(item) * 9);
      const auto sample = static_cast<std::int64_t>(session.frame_offset) * 600 +
                          static_cast<std::int64_t>(window_start) * 600;
      seeds[item] = session.seed + static_cast<std::uint64_t>(sample * 9);
    }
    const auto model_dtype = device_ < 0 ? DType::F32 : DType::F16;
    auto inputs = weights_bc_;
    inputs.emplace("en", upload_floats("bc.en", encoding,
                                        {batch_size, channels, kWindowFrames},
                                        model_dtype));
    inputs.emplace("asr", upload_floats("bc.asr", asr,
                                         {batch_size, 512, kWindowFrames},
                                         model_dtype));
    inputs.emplace("s", upload_floats("bc.s", styles, {batch_size, 128},
                                       model_dtype));
    inputs.emplace("ref", upload_floats("bc.ref", references,
                                          {batch_size, 128}, model_dtype));
    inputs.emplace("phase", upload_floats("bc.phase", phases,
                                            {batch_size, 9}, DType::F32));
    inputs.emplace("source_noise", source_noise(seeds));
    const auto output = execution_bc_->run(inputs, stream_);
    const auto all_audio = download_floats(need(output, "audio"));
    const auto next_phases = download_floats(need(output, "next_phase"));
    std::vector<float> audio;
    std::vector<std::int32_t> offsets{0};
    for (std::int32_t item = 0; item < batch_size; ++item)
    {
      const auto base = static_cast<std::size_t>(item) * kWindowCore * 600;
      audio.insert(audio.end(), all_audio.begin() + base,
                   all_audio.begin() + base + counts[item] * 600);
      offsets.push_back(static_cast<std::int32_t>(audio.size()));
      auto &session = *sessions[item];
      std::copy_n(next_phases.begin() + static_cast<std::size_t>(item) * 9, 9,
                  session.phase.begin());
      session.cursor += counts[item];
      if (complete[item])
      {
        tails_.insert_or_assign(ids[item], styletts2::retain_tail(session));
        sessions_.erase(ids[item]);
      }
    }
    Output result;
    result.tensors.push_back(
        tensor("audio", DType::F32,
               {static_cast<std::int64_t>(audio.size())}, audio));
    result.tensors.push_back(tensor("audio_offsets", DType::I32,
                                    {batch_size + 1}, offsets));
    result.tensors.push_back(
        tensor("frame_starts", DType::I32, {batch_size}, starts));
    result.tensors.push_back(
        tensor("frame_counts", DType::I32, {batch_size}, counts));
    result.tensors.push_back(
        tensor("complete", DType::I32, {batch_size}, complete));
    return result;
  }

} // namespace tinfer::native
