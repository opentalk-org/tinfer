#pragma once

#include "tinfer_rust/src/models/styletts2/cpp/session.hpp"

namespace tinfer::styletts2 {

void expand_window(const native::StyleTts2Session& session, float* asr,
                   float* encoding);
void expand_window_at(const native::StyleTts2Session& session,
                      std::int32_t window_start, float* asr, float* encoding);
void copy_prosody(const native::StyleTts2Session& session,
                  std::int32_t frame_start, float* f0, float* noise);
native::StyleTts2Tail retain_tail(const native::StyleTts2Session& session);

}  // namespace tinfer::styletts2
