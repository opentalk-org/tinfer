// Shared C++/CUDA runtime helpers: TensorRT logger, engine loading, and the flat
// TINF weight/tensor bundle reader (produced by scripts/build_common.py).
#pragma once
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                                    \
  do {                                                                                   \
    cudaError_t e = (x);                                                                 \
    if (e != cudaSuccess) {                                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,       \
              __LINE__);                                                                 \
      std::abort();                                                                      \
    }                                                                                    \
  } while (0)

namespace tinfer {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
  }
};

// dtype codes match scripts build_common: 0=f16, 1=f32, 2=i32, 3=i64, 4=bool
inline size_t dtypeSize(int code) {
  switch (code) {
    case 0: return 2;
    case 1: return 4;
    case 2: return 4;
    case 3: return 8;
    case 4: return 1;
  }
  throw std::runtime_error("bad dtype code");
}

struct Tensor {
  int dtype = 0;
  std::vector<int64_t> dims;
  void* dptr = nullptr;  // device pointer
  int64_t numel = 1;
  size_t nbytes = 0;
};

// Loads a TINF bundle, uploading every tensor to the GPU. Returns name->Tensor.
inline std::map<std::string, Tensor> loadBundle(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[4];
  f.read(magic, 4);
  if (std::memcmp(magic, "TINF", 4) != 0) throw std::runtime_error("bad magic " + path);
  int32_t count;
  f.read(reinterpret_cast<char*>(&count), 4);
  std::map<std::string, Tensor> out;
  for (int i = 0; i < count; ++i) {
    int32_t nlen;
    f.read(reinterpret_cast<char*>(&nlen), 4);
    std::string name(nlen, '\0');
    f.read(name.data(), nlen);
    Tensor t;
    int32_t code, ndim;
    f.read(reinterpret_cast<char*>(&code), 4);
    f.read(reinterpret_cast<char*>(&ndim), 4);
    t.dtype = code;
    t.dims.resize(ndim);
    t.numel = 1;
    for (int d = 0; d < ndim; ++d) {
      int64_t dim;
      f.read(reinterpret_cast<char*>(&dim), 8);
      t.dims[d] = dim;
      t.numel *= dim;
    }
    t.nbytes = t.numel * dtypeSize(code);
    std::vector<char> host(t.nbytes);
    f.read(host.data(), t.nbytes);
    CUDA_CHECK(cudaMalloc(&t.dptr, t.nbytes));
    CUDA_CHECK(cudaMemcpy(t.dptr, host.data(), t.nbytes, cudaMemcpyHostToDevice));
    out[name] = t;
  }
  return out;
}

inline std::vector<char> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  size_t n = f.tellg();
  f.seekg(0);
  std::vector<char> buf(n);
  f.read(buf.data(), n);
  return buf;
}

inline nvinfer1::Dims toDims(const std::vector<int64_t>& v) {
  nvinfer1::Dims d;
  d.nbDims = static_cast<int>(v.size());
  for (int i = 0; i < d.nbDims; ++i) d.d[i] = v[i];
  return d;
}

}  // namespace tinfer
