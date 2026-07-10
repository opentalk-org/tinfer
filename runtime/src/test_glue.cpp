// Validate the glue2 CUDA kernel (source module + forward STFT) against the
// PyTorch _preprocess_f0 reference (randomness disabled).
#include "common.h"
#include "kernels.cuh"
#include <algorithm>

static float h2f(uint16_t h) { __half x; std::memcpy(&x, &h, 2); return __half2float(x); }

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "/workspace/tinfer/runtime/engines";
  auto ref = tinfer::loadBundle(dir + "/glue.ref");
  int B = ref["f0"].dims[0];
  int F = ref["f0"].dims[1] / 2;
  int nframe = F * 120 + 1;
  printf("B=%d F=%d har frames=%d\n", B, F, nframe);
  __half* har;
  CUDA_CHECK(cudaMalloc(&har, (size_t)B * 22 * nframe * 2));
  cudaStream_t s; cudaStreamCreate(&s);
  glue::source_to_har((const __half*)ref["f0"].dptr, (const __half*)ref["linW"].dptr,
                      (const __half*)ref["linB"].dptr, har, B, F, 0, 0, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { printf("kernel err %s\n", cudaGetErrorString(e)); return 1; }

  int64_t n = (int64_t)B * 22 * nframe;
  std::vector<uint16_t> o(n), r(ref["har"].numel);
  CUDA_CHECK(cudaMemcpy(o.data(), har, n * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(r.data(), ref["har"].dptr, ref["har"].nbytes, cudaMemcpyDeviceToHost));
  // channel layout: har[b, ch, frame], ch 0..10 magnitude, 11..21 phase
  double magmx = 0, magsm = 0, magrmx = 0, phmx = 0, phsm = 0;
  int64_t magn = 0, phn = 0;
  for (int b = 0; b < B; ++b)
    for (int ch = 0; ch < 22; ++ch)
      for (int f = 0; f < nframe; ++f) {
        int64_t i = ((int64_t)b * 22 + ch) * nframe + f;
        float a = h2f(o[i]), bb = h2f(r[i]);
        double e = std::abs(a - bb);
        if (ch < 11) { magmx = std::max(magmx, e); magsm += e; magrmx = std::max(magrmx, (double)std::abs(bb)); magn++; }
        else {
          double wrap = std::abs(std::abs(e) - 2 * 3.14159265358979);  // phase wrap distance
          double pe = std::min(e, wrap);
          phmx = std::max(phmx, pe); phsm += pe; phn++;
        }
      }
  printf("glue2 MAG: max %.4f mean %.5f ref_absmax %.3f\n", magmx, magsm / magn, magrmx);
  printf("glue2 PHASE (wrap-aware): max %.4f mean %.5f\n", phmx, phsm / phn);
  return 0;
}
