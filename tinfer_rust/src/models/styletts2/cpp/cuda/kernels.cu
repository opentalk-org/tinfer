#include "tinfer_rust/src/models/styletts2/cpp/cuda/glue.hpp"

#include <cmath>

namespace tinfer::styletts2::cuda {
namespace {
constexpr float kPi = 3.14159265358979323846F;
constexpr int kSampleRate = 24000;
constexpr int kUpsample = 300;
constexpr int kHarmonics = 9;
constexpr int kFft = 20;
constexpr int kHop = 5;
constexpr int kBins = 11;

__global__ void duration_kernel(const __half* durations, const std::int32_t* lengths,
                                const float* speeds, std::int32_t* predicted,
                                std::int32_t* starts, std::int32_t* totals,
                                std::int32_t batch, std::int32_t tokens) {
  const auto item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= batch) return;
  std::int32_t total = 0;
  for (std::int32_t token = 0; token < tokens; ++token) {
    const auto index = item * tokens + token;
    const auto duration = token < lengths[item]
                              ? max(1, static_cast<int>(lroundf(__half2float(durations[index]) / speeds[item])))
                              : 0;
    predicted[index] = duration;
    starts[index] = total;
    total += duration;
  }
  totals[item] = total;
}

__global__ void token_kernel(const std::int32_t* predicted,
                             const std::int32_t* starts,
                             std::int32_t* token_of_frame,
                             std::int32_t batch, std::int32_t tokens,
                             std::int32_t frames) {
  const auto token = blockIdx.x * blockDim.x + threadIdx.x;
  const auto item = blockIdx.y;
  if (item >= batch || token >= tokens) return;
  const auto index = item * tokens + token;
  for (auto frame = starts[index]; frame < starts[index] + predicted[index] && frame < frames; ++frame) {
    token_of_frame[item * frames + frame] = token;
  }
}

__global__ void expand_kernel(const __half* text, const __half* encoding,
                              const std::int32_t* token_of_frame,
                              const std::int32_t* totals, __half* asr,
                              __half* expanded, std::int32_t tokens,
                              std::int32_t channels, std::int32_t frames) {
  const auto frame = blockIdx.x * blockDim.x + threadIdx.x;
  const auto channel = blockIdx.y;
  const auto item = blockIdx.z;
  if (frame >= frames) return;
  const auto valid = frame < totals[item];
  const auto token = valid ? token_of_frame[item * frames + frame] : 0;
  if (channel < 512) {
    asr[(item * 512 + channel) * frames + frame] = valid ? text[(item * 512 + channel) * tokens + token] : __float2half(0.0F);
  }
  if (channel < channels) {
    expanded[(item * channels + channel) * frames + frame] = valid ? encoding[(item * tokens + token) * channels + channel] : __float2half(0.0F);
  }
}

__global__ void phase_kernel(const __half* f0, float* phases,
                             float* phase_state,
                             const std::int32_t* advances,
                             std::int32_t batch, std::int32_t f0_frames) {
  const auto harmonic = blockIdx.x * blockDim.x + threadIdx.x;
  const auto item = blockIdx.y;
  if (harmonic >= kHarmonics || item >= batch) return;
  float phase = phase_state[item * kHarmonics + harmonic];
  for (std::int32_t frame = 0; frame < f0_frames; ++frame) {
    phase += 2.0F * kPi * fmodf(__half2float(f0[item * f0_frames + frame]) * (harmonic + 1) / kSampleRate, 1.0F);
    phases[(item * kHarmonics + harmonic) * f0_frames + frame] = phase;
    if (frame + 1 == advances[item] * 2) {
      phase_state[item * kHarmonics + harmonic] = fmodf(phase, 2.0F * kPi);
    }
  }
}

__device__ std::uint32_t hash(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

__device__ float normal(std::uint32_t seed) {
  const auto first = static_cast<float>(hash(seed) & 0xFFFFFF) / 16777216.0F + 1e-7F;
  const auto second = static_cast<float>(hash(seed * 2654435761U + 1) & 0xFFFFFF) / 16777216.0F;
  return sqrtf(-2.0F * logf(first)) * cosf(2.0F * kPi * second);
}

__global__ void source_kernel(const __half* f0, const float* phases,
                              const __half* weights, const __half* bias,
                              const std::uint64_t* seeds, float* source,
                              std::int32_t f0_frames, bool randomize) {
  const auto sample = blockIdx.x * blockDim.x + threadIdx.x;
  const auto item = blockIdx.y;
  const auto sample_count = f0_frames * kUpsample;
  if (sample >= sample_count) return;
  const auto position = (sample + 0.5F) / kUpsample - 0.5F;
  const auto left = max(0, min(f0_frames - 1, static_cast<int>(floorf(position))));
  const auto right = max(0, min(f0_frames - 1, left + 1));
  const auto fraction = position - floorf(position);
  const auto voiced = __half2float(f0[item * f0_frames + sample / kUpsample]) > 10.0F;
  float value = __half2float(*bias);
  for (int harmonic = 0; harmonic < kHarmonics; ++harmonic) {
    const auto base = (item * kHarmonics + harmonic) * f0_frames;
    const auto phase = ((1.0F - fraction) * phases[base + left] + fraction * phases[base + right]) * kUpsample;
    const auto noise = randomize ? normal(static_cast<std::uint32_t>(seeds[item] + static_cast<std::uint64_t>(sample) * kHarmonics + harmonic)) : 0.0F;
    const auto wave = voiced ? sinf(phase) * 0.1F + noise * 0.003F : noise * (0.1F / 3.0F);
    value += wave * __half2float(weights[harmonic]);
  }
  source[item * sample_count + sample] = tanhf(value);
}

__global__ void stft_kernel(const float* source, __half* harmonic,
                            std::int32_t batch, std::int32_t sample_count) {
  const auto frame = blockIdx.x * blockDim.x + threadIdx.x;
  const auto item = blockIdx.y;
  const auto frame_count = sample_count / kHop + 1;
  if (item >= batch || frame >= frame_count) return;
  for (int bin = 0; bin < kBins; ++bin) {
    float real = 0.0F;
    float imaginary = 0.0F;
    for (int index = 0; index < kFft; ++index) {
      std::int32_t sample = static_cast<std::int32_t>(frame) * kHop + index - kFft / 2;
      if (sample < 0) sample = -sample;
      if (sample >= sample_count) sample = 2 * (sample_count - 1) - sample;
      const auto window = 0.5F - 0.5F * cosf(2.0F * kPi * index / kFft);
      const auto angle = 2.0F * kPi * bin * index / kFft;
      const auto value = source[item * sample_count + sample] * window;
      real += value * cosf(angle);
      imaginary -= value * sinf(angle);
    }
    const auto base = (item * 2 * kBins) * frame_count;
    harmonic[base + bin * frame_count + frame] = __float2half(hypotf(real, imaginary));
    harmonic[base + (kBins + bin) * frame_count + frame] = __float2half(atan2f(imaginary, real));
  }
}
}  // namespace

void duration_prefix(const __half* durations, const std::int32_t* lengths,
                     const float* speeds, std::int32_t* predicted,
                     std::int32_t* starts, std::int32_t* totals,
                     std::int32_t batch, std::int32_t tokens,
                     cudaStream_t stream) {
  duration_kernel<<<(batch + 31) / 32, 32, 0, stream>>>(durations, lengths, speeds, predicted, starts, totals, batch, tokens);
}

void build_tokens(const std::int32_t* predicted, const std::int32_t* starts,
                  const std::int32_t*, std::int32_t* token_of_frame,
                  std::int32_t batch, std::int32_t tokens,
                  std::int32_t frames, cudaStream_t stream) {
  cudaMemsetAsync(token_of_frame, 0, static_cast<std::size_t>(batch) * frames * sizeof(std::int32_t), stream);
  token_kernel<<<dim3((tokens + 127) / 128, batch), 128, 0, stream>>>(predicted, starts, token_of_frame, batch, tokens, frames);
}

void align_expand(const __half* text, const __half* encoding,
                  const std::int32_t* token_of_frame,
                  const std::int32_t* totals, __half* asr, __half* expanded,
                  std::int32_t batch, std::int32_t tokens,
                  std::int32_t channels, std::int32_t frames,
                  cudaStream_t stream) {
  expand_kernel<<<dim3((frames + 127) / 128, max(512, channels), batch), 128, 0, stream>>>(text, encoding, token_of_frame, totals, asr, expanded, tokens, channels, frames);
}

void source_to_har(const __half* f0, const __half* weights,
                   const __half* bias, const std::uint64_t* seeds, __half* har,
                   float* phases, float* phase_state,
                   const std::int32_t* advances, float* source,
                   std::int32_t batch, std::int32_t frames, bool randomize,
                   cudaStream_t stream) {
  const auto f0_frames = frames * 2;
  const auto samples = f0_frames * kUpsample;
  phase_kernel<<<dim3(1, batch), 32, 0, stream>>>(
      f0, phases, phase_state, advances, batch, f0_frames);
  source_kernel<<<dim3((samples + 255) / 256, batch), 256, 0, stream>>>(f0, phases, weights, bias, seeds, source, f0_frames, randomize);
  stft_kernel<<<dim3((samples / kHop + 128) / 128, batch), 128, 0, stream>>>(source, har, batch, samples);
}

}  // namespace tinfer::styletts2::cuda
