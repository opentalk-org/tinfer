#include "tinfer_rust/src/models/styletts2/cpp/cpu/glue.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tinfer::styletts2::cpu {
namespace {
constexpr float kPi = 3.14159265358979323846F;
constexpr int kSampleRate = 24000;
constexpr int kUpsample = 300;
constexpr int kHarmonics = 9;
constexpr int kFft = 20;
constexpr int kHop = 5;
constexpr int kBins = 11;

std::uint32_t hash(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

float normal(std::uint32_t seed) {
  const float first = static_cast<float>(hash(seed) & 0xFFFFFF) / 16777216.0F + 1e-7F;
  const float second = static_cast<float>(hash(seed * 2654435761U + 1) & 0xFFFFFF) / 16777216.0F;
  return std::sqrt(-2.0F * std::log(first)) * std::cos(2.0F * kPi * second);
}
}  // namespace

Alignment duration_prefix(const float* durations, const std::int32_t* lengths,
                          const float* speeds, std::int32_t batch,
                          std::int32_t token_count) {
  if (batch < 1 || token_count < 1) throw std::invalid_argument("invalid alignment dimensions");
  Alignment result;
  result.durations.resize(static_cast<std::size_t>(batch) * token_count);
  result.starts.resize(result.durations.size());
  result.totals.resize(batch);
  for (std::int32_t item = 0; item < batch; ++item) {
    if (lengths[item] < 0 || lengths[item] > token_count || speeds[item] <= 0.0F) {
      throw std::invalid_argument("invalid alignment length or speed");
    }
    std::int32_t total = 0;
    for (std::int32_t token = 0; token < token_count; ++token) {
      const auto index = static_cast<std::size_t>(item) * token_count + token;
      const auto duration = token < lengths[item]
                                ? std::max<std::int32_t>(1, std::lround(durations[index] / speeds[item]))
                                : 0;
      result.durations[index] = duration;
      result.starts[index] = total;
      total += duration;
    }
    result.totals[item] = total;
    result.frames = std::max(result.frames, total);
  }
  result.tokens.assign(static_cast<std::size_t>(batch) * result.frames, 0);
  for (std::int32_t item = 0; item < batch; ++item) {
    for (std::int32_t token = 0; token < token_count; ++token) {
      const auto index = static_cast<std::size_t>(item) * token_count + token;
      for (std::int32_t frame = result.starts[index]; frame < result.starts[index] + result.durations[index]; ++frame) {
        result.tokens[static_cast<std::size_t>(item) * result.frames + frame] = token;
      }
    }
  }
  return result;
}

void align_expand(const float* text, const float* encoding,
                  const Alignment& alignment, std::int32_t batch,
                  std::int32_t token_count, std::int32_t encoding_channels,
                  std::vector<float>& asr, std::vector<float>& expanded) {
  asr.assign(static_cast<std::size_t>(batch) * 512 * alignment.frames, 0.0F);
  expanded.assign(static_cast<std::size_t>(batch) * encoding_channels * alignment.frames, 0.0F);
  for (std::int32_t item = 0; item < batch; ++item) {
    for (std::int32_t frame = 0; frame < alignment.totals[item]; ++frame) {
      const auto token = alignment.tokens[static_cast<std::size_t>(item) * alignment.frames + frame];
      for (std::int32_t channel = 0; channel < 512; ++channel) {
        asr[(static_cast<std::size_t>(item) * 512 + channel) * alignment.frames + frame] =
            text[(static_cast<std::size_t>(item) * 512 + channel) * token_count + token];
      }
      for (std::int32_t channel = 0; channel < encoding_channels; ++channel) {
        expanded[(static_cast<std::size_t>(item) * encoding_channels + channel) * alignment.frames + frame] =
            encoding[(static_cast<std::size_t>(item) * token_count + token) * encoding_channels + channel];
      }
    }
  }
}

std::vector<float> source_to_har(const float* f0, const float* weights,
                                 float bias, std::int32_t batch,
                                 std::int32_t frames, const std::uint64_t* seeds,
                                 bool randomize, const std::int32_t* advances,
                                 float* phase_state) {
  const auto f0_frames = frames * 2;
  const auto samples = f0_frames * kUpsample;
  const auto stft_frames = samples / kHop + 1;
  std::vector<float> output(static_cast<std::size_t>(batch) * 2 * kBins * stft_frames);
  std::vector<float> phases(static_cast<std::size_t>(kHarmonics) * f0_frames);
  std::vector<float> source(samples);
  for (std::int32_t item = 0; item < batch; ++item) {
    for (int harmonic = 0; harmonic < kHarmonics; ++harmonic) {
      float phase = phase_state[static_cast<std::size_t>(item) * kHarmonics + harmonic];
      for (std::int32_t frame = 0; frame < f0_frames; ++frame) {
        phase += 2.0F * kPi * std::fmod(f0[static_cast<std::size_t>(item) * f0_frames + frame] * (harmonic + 1) / kSampleRate, 1.0F);
        phases[static_cast<std::size_t>(harmonic) * f0_frames + frame] = phase;
        if (frame + 1 == advances[item] * 2) {
          phase_state[static_cast<std::size_t>(item) * kHarmonics + harmonic] =
              std::fmod(phase, 2.0F * kPi);
        }
      }
    }
    for (std::int32_t sample = 0; sample < samples; ++sample) {
      const auto frame = std::min(sample / kUpsample, f0_frames - 1);
      const auto position = (sample + 0.5F) / kUpsample - 0.5F;
      const auto left = std::max(0, std::min(f0_frames - 1, static_cast<int>(std::floor(position))));
      const auto right = std::max(0, std::min(f0_frames - 1, left + 1));
      const auto fraction = position - std::floor(position);
      const auto voiced = f0[static_cast<std::size_t>(item) * f0_frames + frame] > 10.0F;
      float value = bias;
      for (int harmonic = 0; harmonic < kHarmonics; ++harmonic) {
        const auto base = static_cast<std::size_t>(harmonic) * f0_frames;
        const auto phase = ((1.0F - fraction) * phases[base + left] + fraction * phases[base + right]) * kUpsample;
        const auto sine = std::sin(phase) * 0.1F;
        const auto noise = randomize ? normal(static_cast<std::uint32_t>(seeds[item] + static_cast<std::uint64_t>(sample) * kHarmonics + harmonic)) : 0.0F;
        value += (voiced ? sine + noise * 0.003F : noise * (0.1F / 3.0F)) * weights[harmonic];
      }
      source[sample] = std::tanh(value);
    }
    for (std::int32_t frame = 0; frame < stft_frames; ++frame) {
      for (int bin = 0; bin < kBins; ++bin) {
        float real = 0.0F;
        float imaginary = 0.0F;
        for (int index = 0; index < kFft; ++index) {
          auto sample = frame * kHop + index - kFft / 2;
          if (sample < 0) sample = -sample;
          if (sample >= samples) sample = 2 * (samples - 1) - sample;
          const auto window = 0.5F - 0.5F * std::cos(2.0F * kPi * index / kFft);
          const auto angle = 2.0F * kPi * bin * index / kFft;
          real += source[sample] * window * std::cos(angle);
          imaginary -= source[sample] * window * std::sin(angle);
        }
        const auto base = (static_cast<std::size_t>(item) * 2 * kBins) * stft_frames;
        output[base + static_cast<std::size_t>(bin) * stft_frames + frame] = std::hypot(real, imaginary);
        output[base + static_cast<std::size_t>(kBins + bin) * stft_frames + frame] = std::atan2(imaginary, real);
      }
    }
  }
  return output;
}

}  // namespace tinfer::styletts2::cpu

namespace tinfer::native {

rust::Vec<std::int32_t> cpu_duration_prefix(rust::Slice<const float> durations,
                                            rust::Slice<const std::int32_t> lengths,
                                            rust::Slice<const float> speeds,
                                            std::int32_t batch, std::int32_t tokens) {
  if (durations.size() != static_cast<std::size_t>(batch) * tokens ||
      lengths.size() != static_cast<std::size_t>(batch) || speeds.size() != lengths.size()) {
    throw std::invalid_argument("CPU duration inputs differ from dimensions");
  }
  const auto result = styletts2::cpu::duration_prefix(durations.data(), lengths.data(), speeds.data(), batch, tokens);
  rust::Vec<std::int32_t> values;
  values.reserve(result.durations.size());
  for (const auto value : result.durations) values.push_back(value);
  return values;
}

}  // namespace tinfer::native
