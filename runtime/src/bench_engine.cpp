// Generic C++/CUDA validate+benchmark for a single weight-input engine.
// Usage: bench_engine <dir> <name> [out1 out2 ...]
//   loads <name>.engine/.weights/.ref; any engine input present in .ref is an
//   activation (bound from ref), the rest are weights (bound once). Validates the
//   named outputs against ref, then benchmarks across batch sizes (scaling dim0).
#include "common.h"
#include <algorithm>
#include <memory>

using namespace nvinfer1;
using tinfer::Tensor;
static tinfer::Logger gLogger;

static float h2f(uint16_t h) { __half x; std::memcpy(&x, &h, 2); return __half2float(x); }

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "/workspace/tinfer/runtime/engines";
  std::string name = argc > 2 ? argv[2] : "B";
  std::vector<std::string> checkOuts;
  for (int i = 3; i < argc; ++i) checkOuts.push_back(argv[i]);

  auto bytes = tinfer::readFile(dir + "/" + name + ".engine");
  std::unique_ptr<IRuntime> rt(createInferRuntime(gLogger));
  std::unique_ptr<ICudaEngine> engine(rt->deserializeCudaEngine(bytes.data(), bytes.size()));
  std::unique_ptr<IExecutionContext> ctx(engine->createExecutionContext());
  auto weights = tinfer::loadBundle(dir + "/" + name + ".weights");
  auto ref = tinfer::loadBundle(dir + "/" + name + ".ref");

  std::vector<std::string> ins, outs, acts;
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* n = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(n) == TensorIOMode::kINPUT) {
      ins.push_back(n);
      if (ref.count(n)) acts.push_back(n);
    } else outs.push_back(n);
  }
  printf("engine %s: %zu inputs (%zu weights, %zu acts), %zu outputs\n", name.c_str(),
         ins.size(), weights.size(), acts.size(), outs.size());

  cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
  for (auto& n : ins) {
    auto it = weights.find(n);
    if (it == weights.end()) continue;
    ctx->setInputShape(n.c_str(), tinfer::toDims(it->second.dims));
    ctx->setTensorAddress(n.c_str(), it->second.dptr);
  }

  auto allocOutputs = [&](std::map<std::string, void*>& op) {
    for (auto& n : outs) {
      Dims s = ctx->getTensorShape(n.c_str());
      int64_t ne = 1; for (int i = 0; i < s.nbDims; ++i) ne *= s.d[i];
      if (op.count(n)) CUDA_CHECK(cudaFree(op[n]));
      CUDA_CHECK(cudaMalloc(&op[n], std::max<int64_t>(ne, 1) * 2));
      ctx->setTensorAddress(n.c_str(), op[n]);
    }
  };

  // ---- validation at ref shapes ----
  for (auto& n : acts) { ctx->setInputShape(n.c_str(), tinfer::toDims(ref[n].dims));
                         ctx->setTensorAddress(n.c_str(), ref[n].dptr); }
  std::map<std::string, void*> op; allocOutputs(op);
  if (!ctx->enqueueV3(stream)) { fprintf(stderr, "enqueue failed\n"); return 1; }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (auto& on : checkOuts) {
    if (!ref.count(on)) { printf("  (no ref for %s)\n", on.c_str()); continue; }
    Dims s = ctx->getTensorShape(on.c_str());
    int64_t ne = 1; for (int i = 0; i < s.nbDims; ++i) ne *= s.d[i];
    std::vector<uint16_t> o(ne), r(ref[on].numel);
    CUDA_CHECK(cudaMemcpy(o.data(), op[on], ne * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.data(), ref[on].dptr, ref[on].nbytes, cudaMemcpyDeviceToHost));
    double mx = 0, sm = 0, rmx = 0; int64_t c = std::min<int64_t>(ne, ref[on].numel);
    for (int64_t i = 0; i < c; ++i) { float a = h2f(o[i]), b = h2f(r[i]);
      mx = std::max(mx, (double)std::abs(a - b)); sm += std::abs(a - b); rmx = std::max(rmx, (double)std::abs(b)); }
    printf("  validate %-8s max_err %.4f mean %.5f ref_absmax %.3f\n", on.c_str(), mx, sm / c, rmx);
  }

  // ---- benchmark: big zero buffers, scale batch (dim0) ----
  int refB = (int)ref[acts[0]].dims[0];
  std::map<std::string, void*> big;
  std::map<std::string, int64_t> perItem;
  for (auto& n : acts) {
    int64_t pi = ref[n].numel / refB;
    perItem[n] = pi;
    void* p; CUDA_CHECK(cudaMalloc(&p, pi * 16 * tinfer::dtypeSize(ref[n].dtype)));
    CUDA_CHECK(cudaMemset(p, 0, pi * 16 * tinfer::dtypeSize(ref[n].dtype)));
    big[n] = p;
  }
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  printf(" batch |  %s latency (ms)\n", name.c_str());
  for (int B : {1, 2, 4, 8, 16}) {
    for (auto& n : acts) {
      std::vector<int64_t> d = ref[n].dims; d[0] = B;
      ctx->setInputShape(n.c_str(), tinfer::toDims(d));
      ctx->setTensorAddress(n.c_str(), big[n]);
    }
    allocOutputs(op);
    for (int i = 0; i < 5; ++i) ctx->enqueueV3(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int N = 100;
    CUDA_CHECK(cudaEventRecord(a, stream));
    for (int i = 0; i < N; ++i) ctx->enqueueV3(stream);
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    // CUDA-graph timing (collapses launch overhead)
    cudaGraph_t g; cudaGraphExec_t ge;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    ctx->enqueueV3(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &g));
    CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 5; ++i) cudaGraphLaunch(ge, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(a, stream));
    for (int i = 0; i < N; ++i) cudaGraphLaunch(ge, stream);
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float msg; CUDA_CHECK(cudaEventElapsedTime(&msg, a, b));
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    printf("  %3d  |  enqueue %8.4f   graph %8.4f\n", B, ms / N, msg / N);
  }
  return 0;
}
