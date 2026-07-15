#include "tinfer_rust/src/models/styletts2/cpp/cuda/glue.hpp"

#include <cmath>

namespace tinfer::styletts2::cuda
{
  namespace
  {
    constexpr float kPi = 3.14159265358979323846F;
    constexpr int kHarmonics = 9;

    __global__ void duration_kernel(const __half *durations, const std::int32_t *lengths,
                                    const float *speeds, std::int32_t *predicted,
                                    std::int32_t *starts, std::int32_t *totals,
                                    std::int32_t batch, std::int32_t tokens)
    {
      const auto item = blockIdx.x * blockDim.x + threadIdx.x;
      if (item >= batch)
        return;
      std::int32_t total = 0;
      for (std::int32_t token = 0; token < tokens; ++token)
      {
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

    __global__ void token_kernel(const std::int32_t *predicted,
                                 const std::int32_t *starts,
                                 std::int32_t *token_of_frame,
                                 std::int32_t batch, std::int32_t tokens,
                                 std::int32_t frames)
    {
      const auto token = blockIdx.x * blockDim.x + threadIdx.x;
      const auto item = blockIdx.y;
      if (item >= batch || token >= tokens)
        return;
      const auto index = item * tokens + token;
      for (auto frame = starts[index]; frame < starts[index] + predicted[index] && frame < frames; ++frame)
      {
        token_of_frame[item * frames + frame] = token;
      }
    }

    __global__ void expand_kernel(const __half *text, const __half *encoding,
                                  const std::int32_t *token_of_frame,
                                  const std::int32_t *totals, __half *asr,
                                  __half *expanded, std::int32_t tokens,
                                  std::int32_t channels, std::int32_t frames)
    {
      const auto frame = blockIdx.x * blockDim.x + threadIdx.x;
      const auto channel = blockIdx.y;
      const auto item = blockIdx.z;
      if (frame >= frames)
        return;
      const auto valid = frame < totals[item];
      const auto token = valid ? token_of_frame[item * frames + frame] : 0;
      if (channel < 512)
      {
        asr[(item * 512 + channel) * frames + frame] = valid ? text[(item * 512 + channel) * tokens + token] : __float2half(0.0F);
      }
      if (channel < channels)
      {
        expanded[(item * channels + channel) * frames + frame] = valid ? encoding[(item * tokens + token) * channels + channel] : __float2half(0.0F);
      }
    }

    __device__ std::uint32_t hash(std::uint32_t value)
    {
      value ^= value >> 16;
      value *= 0x7feb352dU;
      value ^= value >> 15;
      value *= 0x846ca68bU;
      return value ^ (value >> 16);
    }

    __device__ float normal(std::uint32_t seed)
    {
      const auto first = static_cast<float>(hash(seed) & 0xFFFFFF) / 16777216.0F + 1e-7F;
      const auto second = static_cast<float>(hash(seed * 2654435761U + 1) & 0xFFFFFF) / 16777216.0F;
      return sqrtf(-2.0F * logf(first)) * cosf(2.0F * kPi * second);
    }

    __global__ void noise_kernel(__half *output, std::uint64_t seed,
                                 std::int32_t values)
    {
      const auto index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= values)
        return;
      output[index] = __float2half(normal(static_cast<std::uint32_t>(seed + index)));
    }

    __device__ std::int32_t window_token(const WindowSource &source,
                                         std::int32_t frame)
    {
      std::int32_t first = 0;
      std::int32_t last = source.tokens;
      while (first + 1 < last)
      {
        const auto middle = (first + last) / 2;
        if (source.starts[middle] <= frame)
          first = middle;
        else
          last = middle;
      }
      return first;
    }

    __global__ void window_kernel(const WindowSource *sources, __half *asr,
                                  __half *encoding, std::int32_t frames)
    {
      const auto frame = blockIdx.x * blockDim.x + threadIdx.x;
      const auto channel = blockIdx.y;
      const auto item = blockIdx.z;
      if (frame >= frames)
        return;
      const auto &source = sources[item];
  const auto requested = source.cursor - 32 + static_cast<std::int32_t>(frame);
      __half value = __float2half(0.0F);
      if (requested < 0 && requested >= -source.tail_frames)
      {
        const auto tail = 32 + requested;
        if (channel < 512)
          value = source.tail_text[channel * 32 + tail];
        else if (channel - 512 < source.channels)
        {
          value = source.tail_encoding[(channel - 512) * 32 + tail];
        }
      }
      else if (requested >= 0 && requested < source.frames)
      {
        const auto token = window_token(source, requested);
        if (channel < 512)
          value = source.text[channel * source.text_stride + token];
        else if (channel - 512 < source.channels)
        {
          value = source.encoding[token * source.channels + channel - 512];
        }
      }
      if (channel < 512)
        asr[(item * 512 + channel) * frames + frame] = value;
      else if (channel - 512 < source.channels)
      {
        encoding[(item * source.channels + channel - 512) * frames + frame] = value;
      }
    }

    __global__ void style_kernel(const WindowSource *sources, __half *styles,
                                 __half *references)
    {
      const auto channel = blockIdx.x * blockDim.x + threadIdx.x;
      const auto item = blockIdx.y;
      if (channel >= 128)
        return;
      styles[item * 128 + channel] = sources[item].style[channel];
      references[item * 128 + channel] = sources[item].reference[channel];
    }

  } // namespace

  void duration_prefix(const __half *durations, const std::int32_t *lengths,
                       const float *speeds, std::int32_t *predicted,
                       std::int32_t *starts, std::int32_t *totals,
                       std::int32_t batch, std::int32_t tokens,
                       cudaStream_t stream)
  {
    duration_kernel<<<(batch + 31) / 32, 32, 0, stream>>>(durations, lengths, speeds, predicted, starts, totals, batch, tokens);
  }

  void build_tokens(const std::int32_t *predicted, const std::int32_t *starts,
                    const std::int32_t *, std::int32_t *token_of_frame,
                    std::int32_t batch, std::int32_t tokens,
                    std::int32_t frames, cudaStream_t stream)
  {
    cudaMemsetAsync(token_of_frame, 0, static_cast<std::size_t>(batch) * frames * sizeof(std::int32_t), stream);
    token_kernel<<<dim3((tokens + 127) / 128, batch), 128, 0, stream>>>(predicted, starts, token_of_frame, batch, tokens, frames);
  }

  void align_expand(const __half *text, const __half *encoding,
                    const std::int32_t *token_of_frame,
                    const std::int32_t *totals, __half *asr, __half *expanded,
                    std::int32_t batch, std::int32_t tokens,
                    std::int32_t channels, std::int32_t frames,
                    cudaStream_t stream)
  {
    expand_kernel<<<dim3((frames + 127) / 128, max(512, channels), batch), 128, 0, stream>>>(text, encoding, token_of_frame, totals, asr, expanded, tokens, channels, frames);
  }

  void fill_noise(__half *output, const std::uint64_t *seeds,
                  std::int32_t batch, std::int32_t samples,
                  cudaStream_t stream)
  {
    const auto values = samples * kHarmonics;
    for (std::int32_t item = 0; item < batch; ++item)
    {
      noise_kernel<<<(values + 255) / 256, 256, 0, stream>>>(
          output + static_cast<std::size_t>(item) * values, seeds[item], values);
    }
  }

  void expand_windows(const WindowSource *sources, __half *asr,
                      __half *encoding, __half *styles, __half *references,
                      std::int32_t batch, std::int32_t channels,
                      std::int32_t frames,
                      cudaStream_t stream)
  {
    window_kernel<<<dim3((frames + 127) / 128, 512 + channels, batch), 128, 0, stream>>>(
        sources, asr, encoding, frames);
    style_kernel<<<dim3(1, batch), 128, 0, stream>>>(sources, styles, references);
  }

} // namespace tinfer::styletts2::cuda
