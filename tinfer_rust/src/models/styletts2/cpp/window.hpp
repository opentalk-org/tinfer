#pragma once

#include "tinfer_rust/src/models/styletts2/cpp/session.hpp"

namespace tinfer::styletts2 {

void expand_window(const native::StyleTts2Session& session, float* asr,
                   float* encoding);
native::StyleTts2Tail retain_tail(const native::StyleTts2Session& session);

}  // namespace tinfer::styletts2
