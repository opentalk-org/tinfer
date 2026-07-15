#include "tinfer_rust/src/models/styletts2/cpp/window.hpp"

#include <algorithm>

namespace tinfer::styletts2 {
namespace {
std::int32_t token_at(const native::StyleTts2Session& session,
                      std::int32_t frame) {
  const auto found = std::upper_bound(session.starts.begin(), session.starts.end(), frame);
  return static_cast<std::int32_t>(std::max<std::ptrdiff_t>(0, found - session.starts.begin() - 1));
}

void copy_frame(const native::StyleTts2Session& session, std::int32_t frame,
                float* asr, float* encoding, std::int32_t output_frame,
                std::int32_t output_frames) {
  if (frame < 0) {
    const auto tail_frame = session.tail_frames + frame;
    if (tail_frame < 0) return;
    for (std::int32_t channel = 0; channel < 512; ++channel) {
      asr[channel * output_frames + output_frame] =
          session.tail_text[channel * session.tail_frames + tail_frame];
    }
    for (std::int32_t channel = 0; channel < session.channels; ++channel) {
      encoding[channel * output_frames + output_frame] =
          session.tail_encoding[channel * session.tail_frames + tail_frame];
    }
    return;
  }
  if (frame >= session.frames) return;
  const auto token = token_at(session, frame);
  for (std::int32_t channel = 0; channel < 512; ++channel) {
    asr[channel * output_frames + output_frame] =
        session.text[channel * session.tokens + token];
  }
  for (std::int32_t channel = 0; channel < session.channels; ++channel) {
    encoding[channel * output_frames + output_frame] =
        session.encoding[token * session.channels + channel];
  }
}
}  // namespace

void expand_window(const native::StyleTts2Session& session, float* asr,
                   float* encoding) {
  const auto window_start = session.cursor - native::kWindowPre;
  for (std::int32_t window = 0; window < native::kWindowFrames; ++window) {
    copy_frame(session, window_start + window, asr, encoding, window,
               native::kWindowFrames);
  }
}

native::StyleTts2Tail retain_tail(const native::StyleTts2Session& session) {
  native::StyleTts2Tail tail;
  tail.channels = session.channels;
  tail.frames = std::min(native::kWindowPre,
                         session.tail_frames + session.frames);
  tail.text.resize(static_cast<std::size_t>(512) * tail.frames);
  tail.encoding.resize(static_cast<std::size_t>(session.channels) * tail.frames);
  tail.phase = session.phase;
  tail.seed = session.seed;
  tail.frame_offset = session.frame_offset + session.frames;
  const auto first = session.frames - tail.frames;
  for (std::int32_t frame = 0; frame < tail.frames; ++frame) {
    copy_frame(session, first + frame, tail.text.data(), tail.encoding.data(),
               frame, tail.frames);
  }
  return tail;
}

}  // namespace tinfer::styletts2
