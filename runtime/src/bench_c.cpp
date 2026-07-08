// Standalone C++/CUDA benchmark+validation for engine C (decoder+generator).
// Loads the weight-input engine, binds weights once, validates against the dumped
// PyTorch reference, and benchmarks across batch sizes at the ~10s (L=400) point.
#include "common.h"

#include <algorithm>
#include <memory>
#include <numeric>

using namespace nvinfer1;
using tinfer::Tensor;

static tinfer::Logger gLogger;

struct LayerProfiler : public IProfiler {
  std::map<std::string, float> ms;
  std::vector<std::string> order;
  void reportLayerTime(const char* name, float t) noexcept override {
    if (!ms.count(name)) order.push_back(name);
    ms[name] += t;
  }
};

struct EngineC {
  std::unique_ptr<IRuntime> runtime;
  std::unique_ptr<ICudaEngine> engine;
  std::unique_ptr<IExecutionContext> ctx;
  std::vector<std::string> inputs, outputs;

  explicit EngineC(const std::string& plan) {
    auto bytes = tinfer::readFile(plan);
    runtime.reset(createInferRuntime(gLogger));
    engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine) throw std::runtime_error("deserialize failed");
    ctx.reset(engine->createExecutionContext());
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
      const char* n = engine->getIOTensorName(i);
      if (engine->getTensorIOMode(n) == TensorIOMode::kINPUT) inputs.push_back(n);
      else outputs.push_back(n);
    }
  }

  void bindWeights(const std::map<std::string, Tensor>& w) {
    for (auto& n : inputs) {
      auto it = w.find(n);
      if (it == w.end()) continue;
      ctx->setInputShape(n.c_str(), tinfer::toDims(it->second.dims));
      ctx->setTensorAddress(n.c_str(), it->second.dptr);
    }
  }

  void setInput(const std::string& n, const std::vector<int64_t>& dims, void* ptr) {
    ctx->setInputShape(n.c_str(), tinfer::toDims(dims));
    ctx->setTensorAddress(n.c_str(), ptr);
  }
};

static float half2float(uint16_t h) {
  __half x;
  std::memcpy(&x, &h, 2);
  return __half2float(x);
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "/workspace/tinfer/runtime/engines";
  EngineC E(dir + "/C.engine");
  auto weights = tinfer::loadBundle(dir + "/C.weights");
  auto ref = tinfer::loadBundle(dir + "/C.ref");
  printf("engine C: %zu inputs (%zu weights), %zu outputs\n", E.inputs.size(), weights.size(),
         E.outputs.size());

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  E.bindWeights(weights);

  // ---- validation against dumped reference ----
  for (const char* n : {"asr", "f0", "noise", "style", "har"})
    E.setInput(n, ref[n].dims, ref[n].dptr);
  // every output (including the virtual junction split) needs an address.
  std::map<std::string, void*> outPtr;
  auto allocOutputs = [&]() {
    for (auto& n : E.outputs) {
      Dims s = E.ctx->getTensorShape(n.c_str());
      int64_t ne = 1;
      for (int i = 0; i < s.nbDims; ++i) ne *= s.d[i];
      if (outPtr.count(n)) CUDA_CHECK(cudaFree(outPtr[n]));
      CUDA_CHECK(cudaMalloc(&outPtr[n], ne * 2));
      E.ctx->setTensorAddress(n.c_str(), outPtr[n]);
    }
  };
  allocOutputs();
  Dims outShape = E.ctx->getTensorShape("audio");
  int64_t outNumel = 1;
  for (int i = 0; i < outShape.nbDims; ++i) outNumel *= outShape.d[i];
  void* audio = outPtr["audio"];
  if (!E.ctx->enqueueV3(stream)) { fprintf(stderr, "enqueue failed\n"); return 1; }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<uint16_t> outHost(outNumel), refHost(ref["audio"].numel);
  CUDA_CHECK(cudaMemcpy(outHost.data(), audio, outNumel * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(refHost.data(), ref["audio"].dptr, ref["audio"].nbytes,
                        cudaMemcpyDeviceToHost));
  double maxErr = 0, sumErr = 0, refMax = 0;
  int64_t cmp = std::min<int64_t>(outNumel, ref["audio"].numel);
  for (int64_t i = 0; i < cmp; ++i) {
    float o = half2float(outHost[i]), r = half2float(refHost[i]);
    maxErr = std::max(maxErr, (double)std::abs(o - r));
    sumErr += std::abs(o - r);
    refMax = std::max(refMax, (double)std::abs(r));
  }
  printf("validation: max_err %.4f mean_err %.5f ref_absmax %.3f  (%lld samples)\n", maxErr,
         sumErr / cmp, refMax, (long long)cmp);

  // ---- benchmark across batch sizes at L=400 (~10s audio) ----
  const int L = 400;
  int64_t maxHar = 16 * 22 * (L * 120 + 1);
  void *dAsr, *dF0, *dNoise, *dStyle, *dHar;
  CUDA_CHECK(cudaMalloc(&dAsr, (int64_t)16 * 512 * L * 2));
  CUDA_CHECK(cudaMalloc(&dF0, (int64_t)16 * 2 * L * 2));
  CUDA_CHECK(cudaMalloc(&dNoise, (int64_t)16 * 2 * L * 2));
  CUDA_CHECK(cudaMalloc(&dStyle, (int64_t)16 * 128 * 2));
  CUDA_CHECK(cudaMalloc(&dHar, maxHar * 2));
  CUDA_CHECK(cudaMemset(dAsr, 0, (int64_t)16 * 512 * L * 2));
  CUDA_CHECK(cudaMemset(dF0, 0, (int64_t)16 * 2 * L * 2));
  CUDA_CHECK(cudaMemset(dNoise, 0, (int64_t)16 * 2 * L * 2));
  CUDA_CHECK(cudaMemset(dStyle, 0, (int64_t)16 * 128 * 2));
  CUDA_CHECK(cudaMemset(dHar, 0, maxHar * 2));

  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  printf("\n batch |  engine C latency (ms)   [L=%d, ~10s audio]\n", L);
  for (int B : {1, 2, 4, 8, 16}) {
    E.setInput("asr", {B, 512, L}, dAsr);
    E.setInput("f0", {B, 2 * L}, dF0);
    E.setInput("noise", {B, 2 * L}, dNoise);
    E.setInput("style", {B, 128}, dStyle);
    E.setInput("har", {B, 22, L * 120 + 1}, dHar);
    allocOutputs();  // audio + any virtual split outputs
    for (int i = 0; i < 5; ++i) E.ctx->enqueueV3(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int N = 50;
    // plain enqueue timing
    CUDA_CHECK(cudaEventRecord(a, stream));
    for (int i = 0; i < N; ++i) E.ctx->enqueueV3(stream);
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    // CUDA graph timing (captures all kernel launches once, replays)
    cudaGraph_t graph;
    cudaGraphExec_t gexec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    E.ctx->enqueueV3(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, 0));
    for (int i = 0; i < 5; ++i) CUDA_CHECK(cudaGraphLaunch(gexec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(a, stream));
    for (int i = 0; i < N; ++i) CUDA_CHECK(cudaGraphLaunch(gexec, stream));
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float msg;
    CUDA_CHECK(cudaEventElapsedTime(&msg, a, b));
    cudaGraphExecDestroy(gexec);
    cudaGraphDestroy(graph);
    printf("  %3d  |  enqueue %8.3f   graph %8.3f   (%.3f ms/item)\n", B, ms / N, msg / N,
           msg / N / B);
  }

  // ---- per-layer profile at batch 1 (separate context so the timings above stay clean) ----
  {
    std::unique_ptr<IExecutionContext> pctx(E.engine->createExecutionContext());
    LayerProfiler prof;
    pctx->setProfiler(&prof);
    for (auto& n : E.inputs) {
      auto it = weights.find(n);
      if (it != weights.end()) {
        pctx->setInputShape(n.c_str(), tinfer::toDims(it->second.dims));
        pctx->setTensorAddress(n.c_str(), it->second.dptr);
      }
    }
    pctx->setInputShape("asr", tinfer::toDims({1, 512, L}));
    pctx->setTensorAddress("asr", dAsr);
    pctx->setInputShape("f0", tinfer::toDims({1, 2 * L}));
    pctx->setTensorAddress("f0", dF0);
    pctx->setInputShape("noise", tinfer::toDims({1, 2 * L}));
    pctx->setTensorAddress("noise", dNoise);
    pctx->setInputShape("style", tinfer::toDims({1, 128}));
    pctx->setTensorAddress("style", dStyle);
    pctx->setInputShape("har", tinfer::toDims({1, 22, L * 120 + 1}));
    pctx->setTensorAddress("har", dHar);
    for (auto& n : E.outputs) {
      Dims s = pctx->getTensorShape(n.c_str());
      int64_t ne = 1;
      for (int i = 0; i < s.nbDims; ++i) ne *= s.d[i];
      void* p;
      CUDA_CHECK(cudaMalloc(&p, ne * 2));
      pctx->setTensorAddress(n.c_str(), p);
    }
    for (int i = 0; i < 20; ++i) pctx->enqueueV3(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::pair<float, std::string>> v;
    for (auto& n : prof.order) v.push_back({prof.ms[n] / 20, n});
    std::sort(v.rbegin(), v.rend());
    float tot = 0;
    for (auto& p : v) tot += p.first;
    printf("\nper-layer profile (batch1, %zu layers, sum %.3f ms), top 18:\n", v.size(), tot);
    for (int i = 0; i < (int)std::min<size_t>(18, v.size()); ++i)
      printf("  %6.3f ms  %s\n", v[i].first, v[i].second.c_str());
  }
  return 0;
}
