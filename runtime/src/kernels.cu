#include "kernels.cuh"
#include <cuda_runtime.h>
#include <math.h>

namespace glue {

constexpr float PI = 3.14159265358979323846f;
constexpr int SR = 24000;
constexpr int UPS = 300;         // f0 upsample scale
constexpr int NH = 9;            // harmonics (harmonic_num+1)
constexpr float SINE_AMP = 0.1f;
constexpr float NOISE_STD = 0.003f;
constexpr float VOICED_TH = 10.0f;
constexpr int NFFT = 20, HOP = 5, NBIN = 11, PAD = 10;

__device__ __forceinline__ float h2f(__half h) { return __half2float(h); }

// ---------------- glue1: alignment ----------------
__global__ void k_duration_prefix(const __half* dur, const int32_t* lengths, const float* speed,
                                  int32_t* predDur, int32_t* startFrame, int32_t* totalFrames,
                                  int B, int T) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  int len = lengths[b];
  float sp = speed[b];
  int acc = 0;
  for (int t = 0; t < T; ++t) {
    int pd = 0;
    if (t < len) {
      float d = h2f(dur[b * T + t]) / sp;
      int r = (int)lroundf(d);
      pd = r < 1 ? 1 : r;
    }
    predDur[b * T + t] = pd;
    startFrame[b * T + t] = acc;
    acc += pd;
  }
  totalFrames[b] = acc;
}

__global__ void k_token_of_frame(const int32_t* predDur, const int32_t* startFrame,
                                 const int32_t* totalFrames, int32_t* tokenOfFrame,
                                 int B, int T, int Fmax) {
  int b = blockIdx.y;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B || t >= T) return;
  int s = startFrame[b * T + t];
  int pd = predDur[b * T + t];
  for (int f = s; f < s + pd && f < Fmax; ++f) tokenOfFrame[b * Fmax + f] = t;
}

__global__ void k_align_expand(const __half* t_en, const __half* d, const int32_t* tokenOfFrame,
                               const int32_t* totalFrames, __half* asr, __half* en,
                               int B, int T, int Cd, int Fmax) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;  // frame
  int c = blockIdx.y;                             // channel (max of 512,Cd)
  int b = blockIdx.z;
  if (f >= Fmax) return;
  int tot = totalFrames[b];
  int tok = (f < tot) ? tokenOfFrame[b * Fmax + f] : 0;
  bool valid = f < tot;
  if (c < 512)
    asr[(b * 512 + c) * Fmax + f] = valid ? t_en[(b * 512 + c) * T + tok] : __float2half(0.f);
  if (c < Cd)
    en[(b * Cd + c) * Fmax + f] = valid ? d[(b * T + tok) * Cd + c] : __float2half(0.f);
}

// ---------------- glue2: source module + forward STFT ----------------
// phase_d[b,9,2F] = 2pi * cumsum_i (F0[b,i]*(h+1)/SR mod 1)
__global__ void k_phase_cumsum(const __half* F0, float* phase_d, int B, int F2) {
  int h = blockIdx.x * blockDim.x + threadIdx.x;  // 0..8
  int b = blockIdx.y;
  if (h >= NH || b >= B) return;
  float acc = 0.f;
  for (int i = 0; i < F2; ++i) {
    float f0 = h2f(F0[b * F2 + i]);
    float rad = fmodf(f0 * (h + 1) / SR, 1.0f);
    acc += rad;
    phase_d[(b * NH + h) * F2 + i] = 2.0f * PI * acc;
  }
}

__device__ __forceinline__ unsigned hashrng(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}
__device__ __forceinline__ float randn(unsigned seed) {
  float u1 = (hashrng(seed) & 0xFFFFFF) / 16777216.0f + 1e-7f;
  float u2 = (hashrng(seed * 2654435761u + 1) & 0xFFFFFF) / 16777216.0f;
  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
}

// sine_merge[b,600F] over j: interp phase_d up by UPS, sin, harmonic linear+tanh, +noise
__global__ void k_sine_merge(const __half* F0, const float* phase_d, const __half* linW,
                             const __half* linB, float* sine_merge, int B, int F2,
                             uint64_t seed, int randScale) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // 0..600F-1
  int b = blockIdx.y;
  int Fup = F2 * UPS;
  if (j >= Fup || b >= B) return;
  float f0 = h2f(F0[b * F2 + (j / UPS)]);
  float uv = f0 > VOICED_TH ? 1.0f : 0.0f;
  float namp = uv * NOISE_STD + (1.0f - uv) * SINE_AMP / 3.0f;
  float coord = (j + 0.5f) / UPS - 0.5f;
  int c0 = (int)floorf(coord);
  float frac = coord - c0;
  int c1 = c0 + 1;
  if (c0 < 0) c0 = 0; if (c0 > F2 - 1) c0 = F2 - 1;
  if (c1 < 0) c1 = 0; if (c1 > F2 - 1) c1 = F2 - 1;
  float lb = h2f(linB[0]);
  float acc = lb;
  for (int h = 0; h < NH; ++h) {
    float p0 = phase_d[(b * NH + h) * F2 + c0] * UPS;
    float p1 = phase_d[(b * NH + h) * F2 + c1] * UPS;
    float phase = (1.0f - frac) * p0 + frac * p1;
    float sine = sinf(phase) * SINE_AMP;
    float noise = randScale ? namp * randn((unsigned)(seed + ((uint64_t)(b * Fup + j) * NH + h))) : 0.0f;
    float sw = sine * uv + noise;
    acc += sw * h2f(linW[h]);
  }
  sine_merge[b * Fup + j] = tanhf(acc);
}

// har[b,22,120F+1] forward STFT-20 (hann periodic, hop 5, center reflect pad 10)
__global__ void k_stft(const float* sine_merge, __half* har, int B, int F2) {
  int Fup = F2 * UPS;
  int nframe = Fup / HOP + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x;  // frame
  int b = blockIdx.y;
  if (k >= nframe || b >= B) return;
  const float* x = sine_merge + b * Fup;
  float re[NBIN], im[NBIN];
  for (int bin = 0; bin < NBIN; ++bin) { re[bin] = 0.f; im[bin] = 0.f; }
  for (int n = 0; n < NFFT; ++n) {
    int s = k * HOP + n - PAD;                 // center reflect
    if (s < 0) s = -s;
    if (s >= Fup) s = 2 * (Fup - 1) - s;
    float w = 0.5f - 0.5f * cosf(2.0f * PI * n / NFFT);  // periodic hann
    float xv = x[s] * w;
    for (int bin = 0; bin < NBIN; ++bin) {
      float ang = 2.0f * PI * bin * n / NFFT;
      re[bin] += xv * cosf(ang);
      im[bin] += -xv * sinf(ang);
    }
  }
  for (int bin = 0; bin < NBIN; ++bin) {
    float mag = sqrtf(re[bin] * re[bin] + im[bin] * im[bin]);
    float ph = atan2f(im[bin], re[bin]);
    har[(b * 22 + bin) * nframe + k] = __float2half(mag);
    har[(b * 22 + NBIN + bin) * nframe + k] = __float2half(ph);
  }
}

// ---------------- launchers ----------------
void duration_prefix(const __half* dur, const int32_t* lengths, const float* speed,
                     int32_t* predDur, int32_t* startFrame, int32_t* totalFrames,
                     int B, int T, cudaStream_t stream) {
  k_duration_prefix<<<(B + 31) / 32, 32, 0, stream>>>(dur, lengths, speed, predDur, startFrame,
                                                      totalFrames, B, T);
}
void build_token_of_frame(const int32_t* predDur, const int32_t* startFrame,
                          const int32_t* totalFrames, int32_t* tokenOfFrame,
                          int B, int T, int Fmax, cudaStream_t stream) {
  cudaMemsetAsync(tokenOfFrame, 0, (size_t)B * Fmax * sizeof(int32_t), stream);
  dim3 g((T + 127) / 128, B);
  k_token_of_frame<<<g, 128, 0, stream>>>(predDur, startFrame, totalFrames, tokenOfFrame, B, T, Fmax);
}
void align_expand(const __half* t_en, const __half* d, const int32_t* tokenOfFrame,
                  const int32_t* totalFrames, __half* asr, __half* en,
                  int B, int T, int Cd, int Fmax, cudaStream_t stream) {
  int Cmax = Cd > 512 ? Cd : 512;
  dim3 g((Fmax + 127) / 128, Cmax, B);
  k_align_expand<<<g, 128, 0, stream>>>(t_en, d, tokenOfFrame, totalFrames, asr, en, B, T, Cd, Fmax);
}
void source_to_har(const __half* F0, const __half* linW, const __half* linB, __half* har,
                   int B, int F, uint64_t seed, int randScale, cudaStream_t stream) {
  int F2 = 2 * F;
  int Fup = F2 * UPS;
  static float* phase_d = nullptr;
  static float* sine_merge = nullptr;
  static size_t cap_pd = 0, cap_sm = 0;
  size_t need_pd = (size_t)B * NH * F2 * sizeof(float);
  size_t need_sm = (size_t)B * Fup * sizeof(float);
  if (need_pd > cap_pd) { if (phase_d) cudaFree(phase_d); cudaMalloc(&phase_d, need_pd); cap_pd = need_pd; }
  if (need_sm > cap_sm) { if (sine_merge) cudaFree(sine_merge); cudaMalloc(&sine_merge, need_sm); cap_sm = need_sm; }
  dim3 g1((NH + 31) / 32, B);
  k_phase_cumsum<<<g1, 32, 0, stream>>>(F0, phase_d, B, F2);
  dim3 g2((Fup + 255) / 256, B);
  k_sine_merge<<<g2, 256, 0, stream>>>(F0, phase_d, linW, linB, sine_merge, B, F2, seed, randScale);
  int nframe = Fup / HOP + 1;
  dim3 g3((nframe + 127) / 128, B);
  k_stft<<<g3, 128, 0, stream>>>(sine_merge, har, B, F2);
}

}  // namespace glue
