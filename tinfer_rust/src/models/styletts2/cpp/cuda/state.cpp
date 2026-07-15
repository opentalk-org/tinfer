#include "tinfer_rust/src/models/styletts2/cpp/model.hpp"

#ifdef TINFER_CUDA
#include "tinfer_rust/src/models/styletts2/cpp/cuda/glue.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <cstring>
#include <stdexcept>

namespace tinfer::native {
#ifdef TINFER_CUDA
namespace {
constexpr std::size_t kHalf = sizeof(__half);

std::size_t timeline_floats(std::int32_t frames) {
  const auto bytes = static_cast<std::size_t>(frames) * 4 * kHalf +
                     static_cast<std::size_t>(frames + 1) * 9 * sizeof(float);
  return (bytes + sizeof(float) - 1) / sizeof(float);
}

__half* timeline_f0(const StyleTts2Session& session) {
  return static_cast<__half*>(session.device_prosody.data());
}

__half* timeline_noise(const StyleTts2Session& session) {
  return timeline_f0(session) + static_cast<std::size_t>(session.frames) * 2;
}

float* timeline_phases(const StyleTts2Session& session) {
  return reinterpret_cast<float*>(timeline_noise(session) +
                                  static_cast<std::size_t>(session.frames) * 2);
}

Buffer view(const Buffer& storage, DType dtype, void* data,
            std::vector<std::int64_t> shape, std::size_t bytes) {
  return Buffer{dtype, std::move(shape),
                std::shared_ptr<void>(storage.memory, data), bytes};
}

void copy(cudaStream_t stream, void* target, const void* source,
          std::size_t bytes, cudaMemcpyKind kind) {
  if (cudaMemcpyAsync(target, source, bytes, kind, stream) != cudaSuccess) {
    throw std::runtime_error("StyleTTS2 CUDA prosody copy failed");
  }
}
}  // namespace

void StyleTts2Model::initialize_device_prosody(
    const std::vector<StyleTts2Session*>& sessions) const {
  if (device_ < 0) return;
  std::size_t total = 0;
  for (const auto* session : sessions) total += timeline_floats(session->frames);
  auto storage = allocate(DType::F32, {static_cast<std::int64_t>(total)}, device_);
  const auto stream = static_cast<cudaStream_t>(stream_);
  auto* position = static_cast<float*>(storage.data());
  for (auto* session : sessions) {
    const auto values = timeline_floats(session->frames);
    session->device_prosody = view(
        storage, DType::F32, position,
        {static_cast<std::int64_t>(values)}, values * sizeof(float));
    if (cudaMemsetAsync(session->device_prosody.data(), 0,
                        session->device_prosody.bytes, stream) != cudaSuccess) {
      throw std::runtime_error("StyleTTS2 CUDA prosody initialization failed");
    }
    copy(stream, timeline_phases(*session), session->phase.data(),
         session->phase.size() * sizeof(float), cudaMemcpyHostToDevice);
    position += values;
  }
}

void StyleTts2Model::append_device_prosody(
    StyleTts2Session& session, const Buffer& f0, const Buffer& noise,
    std::int32_t item, std::int32_t offset, std::int32_t count) const {
  if (f0.dtype != DType::F16 || noise.dtype != DType::F16) {
    throw std::runtime_error("StyleTTS2 CUDA prosody is not float16");
  }
  const auto stream = static_cast<cudaStream_t>(stream_);
  const auto source = static_cast<std::size_t>(item) * kWindowFrames * 2 +
                      offset * 2;
  const auto target = static_cast<std::size_t>(session.prosody_cursor) * 2;
  const auto bytes = static_cast<std::size_t>(count) * 2 * kHalf;
  copy(stream, timeline_f0(session) + target,
       static_cast<const __half*>(f0.data()) + source, bytes,
       cudaMemcpyDeviceToDevice);
  copy(stream, timeline_noise(session) + target,
       static_cast<const __half*>(noise.data()) + source, bytes,
       cudaMemcpyDeviceToDevice);
  styletts2::cuda::append_phases(timeline_f0(session),
                                timeline_phases(session),
                                session.prosody_cursor, count, stream);
}

DeviceProsodyWindow StyleTts2Model::device_prosody_window(
    const std::vector<StyleTts2Session*>& sessions,
    const std::vector<std::int32_t>& starts) const {
  const auto batch = static_cast<std::int32_t>(sessions.size());
  const auto values = static_cast<std::size_t>(batch) * kWindowFrames * 2;
  const auto f0_bytes = values * kHalf;
  const auto phase_bytes = static_cast<std::size_t>(batch) * 9 * sizeof(float);
  auto storage = allocate(
      DType::F32,
      {static_cast<std::int64_t>((f0_bytes * 2 + phase_bytes) / sizeof(float))},
      device_);
  auto* base = static_cast<std::uint8_t*>(storage.data());
  DeviceProsodyWindow window{
      view(storage, DType::F16, base, {batch, kWindowFrames * 2}, f0_bytes),
      view(storage, DType::F16, base + f0_bytes,
           {batch, kWindowFrames * 2}, f0_bytes),
      view(storage, DType::F32, base + f0_bytes * 2, {batch, 9}, phase_bytes)};
  const auto stream = static_cast<cudaStream_t>(stream_);
  if (cudaMemsetAsync(window.f0.data(), 0, f0_bytes * 2, stream) != cudaSuccess) {
    throw std::runtime_error("StyleTTS2 CUDA prosody window initialization failed");
  }
  for (std::int32_t item = 0; item < batch; ++item) {
    const auto& session = *sessions[item];
    const auto count = std::min(kWindowFrames,
                                session.prosody_cursor - starts[item]);
    const auto bytes = static_cast<std::size_t>(count) * 2 * kHalf;
    copy(stream, static_cast<__half*>(window.f0.data()) +
                     static_cast<std::size_t>(item) * kWindowFrames * 2,
         timeline_f0(session) + static_cast<std::size_t>(starts[item]) * 2,
         bytes, cudaMemcpyDeviceToDevice);
    copy(stream, static_cast<__half*>(window.noise.data()) +
                     static_cast<std::size_t>(item) * kWindowFrames * 2,
         timeline_noise(session) + static_cast<std::size_t>(starts[item]) * 2,
         bytes, cudaMemcpyDeviceToDevice);
    copy(stream, static_cast<float*>(window.phases.data()) +
                     static_cast<std::size_t>(item) * 9,
         timeline_phases(session) + static_cast<std::size_t>(starts[item]) * 9,
         9 * sizeof(float), cudaMemcpyDeviceToDevice);
  }
  return window;
}

Buffer StyleTts2Model::harmonic(
    const DeviceProsodyWindow& window,
    const std::vector<std::uint64_t>& seeds) const {
  const auto batch = static_cast<std::int32_t>(seeds.size());
  const auto phase_values = static_cast<std::size_t>(batch) * 9;
  const auto source_values =
      static_cast<std::size_t>(batch) * kWindowFrames * 600;
  const auto output_values = static_cast<std::size_t>(batch) * 22 *
                             (kWindowFrames * 120 + 1);
  const auto bytes = seeds.size() * sizeof(std::uint64_t) +
                     seeds.size() * sizeof(std::int32_t) +
                     phase_values * sizeof(float) * (kWindowFrames * 2 + 1) +
                     source_values * sizeof(float) + output_values * kHalf;
  auto storage = allocate(
      DType::F32,
      {static_cast<std::int64_t>((bytes + sizeof(float) - 1) / sizeof(float))},
      device_);
  auto* position = static_cast<std::uint8_t*>(storage.data());
  auto* seed_data = reinterpret_cast<std::uint64_t*>(position);
  position += seeds.size() * sizeof(std::uint64_t);
  auto* advances = reinterpret_cast<std::int32_t*>(position);
  position += seeds.size() * sizeof(std::int32_t);
  auto* phase_state = reinterpret_cast<float*>(position);
  position += phase_values * sizeof(float);
  auto* phase_frames = reinterpret_cast<float*>(position);
  position += phase_values * kWindowFrames * 2 * sizeof(float);
  auto* source = reinterpret_cast<float*>(position);
  position += source_values * sizeof(float);
  auto output = view(storage, DType::F16, position,
                     {batch, 22, kWindowFrames * 120 + 1},
                     output_values * kHalf);
  const auto stream = static_cast<cudaStream_t>(stream_);
  std::vector<std::int32_t> advance_values(batch, kWindowFrames);
  copy(stream, seed_data, seeds.data(), seeds.size() * sizeof(std::uint64_t),
       cudaMemcpyHostToDevice);
  copy(stream, advances, advance_values.data(),
       advance_values.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice);
  copy(stream, phase_state, window.phases.data(), phase_values * sizeof(float),
       cudaMemcpyDeviceToDevice);
  styletts2::cuda::source_to_har(
      static_cast<const __half*>(window.f0.data()),
      static_cast<const __half*>(glue_weights_.at("linW").data()),
      static_cast<const __half*>(glue_weights_.at("linB").data()), seed_data,
      static_cast<__half*>(output.data()), phase_frames, phase_state, advances,
      source, batch, kWindowFrames, true, stream);
  return output;
}

void StyleTts2Model::finish_device_prosody(
    StyleTts2Session& session) const {
  if (cudaMemcpy(session.phase.data(),
                 timeline_phases(session) +
                     static_cast<std::size_t>(session.frames) * 9,
                 session.phase.size() * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    throw std::runtime_error("StyleTTS2 CUDA final phase download failed");
  }
}
#else
void StyleTts2Model::initialize_device_prosody(
    const std::vector<StyleTts2Session*>&) const {}
void StyleTts2Model::append_device_prosody(StyleTts2Session&, const Buffer&,
                                           const Buffer&, std::int32_t,
                                           std::int32_t, std::int32_t) const {}
DeviceProsodyWindow StyleTts2Model::device_prosody_window(
    const std::vector<StyleTts2Session*>&,
    const std::vector<std::int32_t>&) const {
  throw std::runtime_error("CUDA support is not enabled in this build");
}
Buffer StyleTts2Model::harmonic(const DeviceProsodyWindow&,
                                const std::vector<std::uint64_t>&) const {
  throw std::runtime_error("CUDA support is not enabled in this build");
}
void StyleTts2Model::finish_device_prosody(StyleTts2Session&) const {}
#endif
}  // namespace tinfer::native
