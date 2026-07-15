#include "tinfer_rust/src/models/styletts2/cpp/model.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/window.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace tinfer::native {
namespace {
constexpr float kPi = 3.14159265358979323846F;
constexpr float kSampleRate = 24000.0F;

const Buffer& need(const Tensors& tensors, const std::string& name) {
  const auto found = tensors.find(name);
  if (found == tensors.end()) {
    throw std::runtime_error("missing StyleTTS2 prosody tensor: " + name);
  }
  return found->second;
}

void append_phases(StyleTts2Session& session, std::int32_t count) {
  std::array<float, 9> phase{};
  std::copy_n(session.phase_frames.end() - 9, 9, phase.begin());
  const auto first = session.prosody_cursor * 2;
  for (std::int32_t frame = 0; frame < count; ++frame) {
    for (std::int32_t half = 0; half < 2; ++half) {
      const auto f0 = session.f0[first + frame * 2 + half];
      for (std::int32_t harmonic = 0; harmonic < 9; ++harmonic) {
        phase[harmonic] += 2.0F * kPi *
            std::fmod(f0 * (harmonic + 1) / kSampleRate, 1.0F);
      }
    }
    for (const auto value : phase) {
      session.phase_frames.push_back(std::fmod(value, 2.0F * kPi));
    }
  }
}
}  // namespace

void StyleTts2Model::ensure_prosody(
    const std::vector<StyleTts2Session*>& sessions,
    const std::vector<std::int32_t>& targets) const {
  while (true) {
    std::vector<StyleTts2Session*> pending;
    for (std::size_t index = 0; index < sessions.size(); ++index) {
      if (sessions[index]->prosody_cursor < targets[index]) {
        pending.push_back(sessions[index]);
      }
    }
    if (pending.empty()) return;

    const auto batch = static_cast<std::int32_t>(pending.size());
    const auto channels = pending.front()->channels;
    std::vector<float> encoding(static_cast<std::size_t>(batch) * channels *
                                kWindowFrames);
    std::vector<float> styles(static_cast<std::size_t>(batch) * 128);
    std::vector<std::int32_t> offsets(batch), counts(batch);
    for (std::int32_t item = 0; item < batch; ++item) {
      auto& session = *pending[item];
      const auto window_start = std::max(0, session.prosody_cursor - kWindowPre);
      offsets[item] = session.prosody_cursor - window_start;
      counts[item] = std::min(kWindowCore,
                              session.frames - session.prosody_cursor);
      styletts2::expand_window_at(
          session, window_start, nullptr,
          encoding.data() + static_cast<std::size_t>(item) * channels *
                                kWindowFrames);
      std::copy(session.style.begin(), session.style.end(),
                styles.begin() + static_cast<std::size_t>(item) * 128);
    }
    auto inputs = weights_b_;
    inputs.emplace("en", upload_floats(
                             encoding, {batch, channels, kWindowFrames}));
    inputs.emplace("s", upload_floats(styles, {batch, 128}));
    const auto output = execution_b_->run(inputs, stream_);
    if (device_ >= 0) {
      for (std::int32_t item = 0; item < batch; ++item) {
        auto& session = *pending[item];
        append_device_prosody(session, need(output, "f0"),
                              need(output, "noise"), item, offsets[item],
                              counts[item]);
        session.prosody_cursor += counts[item];
      }
    } else {
      const auto f0 = download_floats(need(output, "f0"));
      const auto noise = download_floats(need(output, "noise"));
      for (std::int32_t item = 0; item < batch; ++item) {
        auto& session = *pending[item];
        const auto source = static_cast<std::size_t>(item) * kWindowFrames * 2 +
                            offsets[item] * 2;
        session.f0.insert(session.f0.end(), f0.begin() + source,
                          f0.begin() + source + counts[item] * 2);
        session.noise.insert(session.noise.end(), noise.begin() + source,
                             noise.begin() + source + counts[item] * 2);
        append_phases(session, counts[item]);
        session.prosody_cursor += counts[item];
      }
    }
  }
}

}  // namespace tinfer::native
