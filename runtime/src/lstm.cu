// Fused bidirectional LSTM CUDA kernels — replacement for the TRT ILoop LSTM that
// (a) emits thousands of tiny per-timestep kernels at batch1 and (b) makes the engine
// non-CUDA-graph-capturable. Input projection is one batched GEMM; the recurrence is a
// single sequential kernel per direction. Matches torch.nn.LSTM (gate order i,f,g,o,
// gates = x@Wih^T + h@Whh^T + bih + bhh). All tensors fp16; math in fp32.
#include "lstm.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>
namespace cg = cooperative_groups;

namespace glue {

__device__ __forceinline__ float hf(__half h) { return __half2float(h); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }

// xp[b,t,4H] = bih + x[b,t,:] @ Wih^T   (per direction). grid over (b*t, 4H tiles).
__global__ void k_input_proj(const __half* x, const __half* Wih, const __half* bih,
                             float* xp, int B, int T, int I, int H4) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // 0..4H-1
  int bt = blockIdx.y;                            // 0..B*T-1
  if (j >= H4) return;
  const __half* xr = x + (size_t)bt * I;
  const __half* wr = Wih + (size_t)j * I;
  float acc = hf(bih[j]);
  for (int k = 0; k < I; ++k) acc += hf(xr[k]) * hf(wr[k]);
  xp[(size_t)bt * H4 + j] = acc;
}

// Sequential recurrence, one block of 4H threads. WhhT is TRANSPOSED [H, 4H] so the
// matvec reads are coalesced across threads. Gates staged in shared memory.
// y layout: [B, T, 2H]; fwd writes [:, :, 0:H], bwd writes [:, :, H:2H].
__global__ void k_recur(const float* xp, const __half* WhhT, const __half* bhh,
                        __half* y, int B, int T, int H, int reverse, int yoff, int y2H) {
  int b = blockIdx.x;
  int j = threadIdx.x;       // gate output index 0..4H-1  (j = gate*H + m)
  int H4 = 4 * H;
  extern __shared__ float sh[];  // h[H], c[H], gates[4H]
  float* h = sh; float* c = sh + H; float* gates = sh + 2 * H;
  if (j < H) { h[j] = 0.f; c[j] = 0.f; }
  __syncthreads();
  for (int s = 0; s < T; ++s) {
    int t = reverse ? (T - 1 - s) : s;
    float g = xp[((size_t)b * T + t) * H4 + j] + hf(bhh[j]);
    for (int k = 0; k < H; ++k) g += h[k] * hf(WhhT[(size_t)k * H4 + j]);  // coalesced over j
    gates[j] = g;
    __syncthreads();
    if (j < H) {
      int m = j;
      float ig = sigm(gates[m]), fg = sigm(gates[H + m]), cg = tanhf(gates[2 * H + m]), og = sigm(gates[3 * H + m]);
      float cm = fg * c[m] + ig * cg;
      float hm = og * tanhf(cm);
      c[m] = cm; h[m] = hm;
      y[((size_t)b * T + t) * y2H + yoff + m] = __float2half(hm);
    }
    __syncthreads();
  }
}

// Cooperative recurrence: the whole grid advances one timestep together, so all SMs
// read Whh in parallel (bandwidth-bound, not one-SM-bound). h/c/gates live in global.
// Runs both directions in one launch (blockIdx.z selects direction).
__global__ void k_recur_coop(const float* xpF, const float* xpR, const __half* WhhF,
                             const __half* WhhR, const __half* bhhF, const __half* bhhR,
                             __half* y, float* hc, float* gbuf, int B, int T, int H, int y2H) {
  cg::grid_group grid = cg::this_grid();
  int H4 = 4 * H;
  int dir = blockIdx.z;                  // 0=fwd, 1=bwd
  int b = blockIdx.y;                    // batch
  const float* xp = dir ? xpR : xpF;
  const __half* WhhT = dir ? WhhR : WhhF;
  const __half* bhh = dir ? bhhR : bhhF;
  int yoff = dir ? H : 0;
  // per (dir,b) state slices
  float* h = hc + ((size_t)(dir * B + b)) * 2 * H;
  float* c = h + H;
  float* gates = gbuf + ((size_t)(dir * B + b)) * H4;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < H; j += gridDim.x * blockDim.x) { h[j] = 0.f; c[j] = 0.f; }
  grid.sync();
  int stride = gridDim.x * blockDim.x;
  for (int s = 0; s < T; ++s) {
    int t = dir ? (T - 1 - s) : s;
    const float* xpt = xp + ((size_t)b * T + t) * H4;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < H4; j += stride) {
      float g = xpt[j] + hf(bhh[j]);
      const __half* col = WhhT + j;
      for (int k = 0; k < H; ++k) g += h[k] * hf(col[(size_t)k * H4]);
      gates[j] = g;
    }
    grid.sync();
    for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < H; m += stride) {
      float ig = sigm(gates[m]), fg = sigm(gates[H + m]), cg2 = tanhf(gates[2 * H + m]), og = sigm(gates[3 * H + m]);
      float cm = fg * c[m] + ig * cg2;
      float hm = og * tanhf(cm);
      c[m] = cm; h[m] = hm;
      y[((size_t)b * T + t) * y2H + yoff + m] = __float2half(hm);
    }
    grid.sync();
  }
}

void lstm_bidir(const __half* x, const __half* WihF, const __half* WhhF, const __half* bihF, const __half* bhhF,
                const __half* WihR, const __half* WhhR, const __half* bihR, const __half* bhhR,
                __half* y, float* xpF, float* xpR, int B, int T, int I, int H, cudaStream_t stream) {
  int H4 = 4 * H;
  dim3 g((H4 + 127) / 128, B * T);
  k_input_proj<<<g, 128, 0, stream>>>(x, WihF, bihF, xpF, B, T, I, H4);
  k_input_proj<<<g, 128, 0, stream>>>(x, WihR, bihR, xpR, B, T, I, H4);
  // cooperative recurrence: choose grid so all blocks are resident.
  static float* hc = nullptr; static float* gbuf = nullptr; static size_t chc = 0, cg_ = 0;
  size_t nhc = (size_t)2 * B * 2 * H * sizeof(float), ng = (size_t)2 * B * H4 * sizeof(float);
  if (nhc > chc) { if (hc) cudaFree(hc); cudaMalloc(&hc, nhc); chc = nhc; }
  if (ng > cg_) { if (gbuf) cudaFree(gbuf); cudaMalloc(&gbuf, ng); cg_ = ng; }
  int blocks_x = 32, threads = 128;
  dim3 grid(blocks_x, B, 2);
  int y2H = 2 * H;
  void* args2[] = {(void*)&xpF, (void*)&xpR, (void*)&WhhF, (void*)&WhhR, (void*)&bhhF, (void*)&bhhR,
                   (void*)&y, (void*)&hc, (void*)&gbuf, (void*)&B, (void*)&T, (void*)&H, (void*)&y2H};
  cudaLaunchCooperativeKernel((void*)k_recur_coop, grid, dim3(threads), args2, 0, stream);
}

}  // namespace glue
