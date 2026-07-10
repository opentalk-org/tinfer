// Full-pipeline benchmark across batch sizes. Tiles the A.ref 150-char utterance
// to each batch, runs A -> glue1 -> B -> glue2 -> C, and reports latency + per-item.
#include "common.h"
#include "kernels.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

using namespace nvinfer1;
using tinfer::Tensor;
static tinfer::Logger gLogger;

struct Engine {
  std::unique_ptr<IRuntime> rt; std::unique_ptr<ICudaEngine> eng;
  std::unique_ptr<IExecutionContext> ctx;
  std::vector<std::string> inputs, outputs; std::map<std::string, void*> out;
  Engine(const std::string& p, tinfer::Logger& lg) {
    auto b = tinfer::readFile(p); rt.reset(createInferRuntime(lg));
    eng.reset(rt->deserializeCudaEngine(b.data(), b.size())); ctx.reset(eng->createExecutionContext());
    for (int i = 0; i < eng->getNbIOTensors(); ++i) { const char* n = eng->getIOTensorName(i);
      (eng->getTensorIOMode(n) == TensorIOMode::kINPUT ? inputs : outputs).push_back(n); }
  }
  void bindWeights(std::map<std::string, Tensor>& w) {
    for (auto& n : inputs) { auto it = w.find(n); if (it == w.end()) continue;
      ctx->setInputShape(n.c_str(), tinfer::toDims(it->second.dims)); ctx->setTensorAddress(n.c_str(), it->second.dptr); } }
  void setIn(const std::string& n, std::vector<int64_t> d, void* p) {
    ctx->setInputShape(n.c_str(), tinfer::toDims(d)); ctx->setTensorAddress(n.c_str(), p); }
  void allocOuts(int maxB) {
    for (auto& n : outputs) { Dims s = ctx->getTensorShape(n.c_str()); int64_t ne = 1;
      for (int i = 0; i < s.nbDims; ++i) ne *= (s.d[i] < 0 ? maxB : s.d[i]);
      size_t bytes = std::max<size_t>((size_t)std::max<int64_t>(ne, 1) * 2, 256ull << 20);
      cudaMalloc(&out[n], bytes); ctx->setTensorAddress(n.c_str(), out[n]); } }
};

// tile a ref tensor [1, ...] to [maxB, ...] on device
static void* tile(Tensor& t, int maxB) {
  size_t itemBytes = t.nbytes; void* p; CUDA_CHECK(cudaMalloc(&p, itemBytes * maxB));
  for (int b = 0; b < maxB; ++b)
    CUDA_CHECK(cudaMemcpy((char*)p + b * itemBytes, t.dptr, itemBytes, cudaMemcpyDeviceToDevice));
  return p;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "/workspace/tinfer/runtime/engines";
  const int MAXB = 16;
  cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
  Engine A(dir + "/A.engine", gLogger), B(dir + "/B.engine", gLogger), C(dir + "/C.engine", gLogger);
  auto wA = tinfer::loadBundle(dir + "/A.weights"); auto wB = tinfer::loadBundle(dir + "/B.weights");
  auto wC = tinfer::loadBundle(dir + "/C.weights"); auto glueW = tinfer::loadBundle(dir + "/glue.weights");
  auto ref = tinfer::loadBundle(dir + "/A.ref");
  A.bindWeights(wA); B.bindWeights(wB); C.bindWeights(wC);
  int T = ref["tokens"].dims[1], Cd = 640;

  std::map<std::string, void*> in;
  for (const char* n : {"tokens", "mask", "ref_s", "noise", "step_noise", "alpha", "beta"})
    in[n] = tile(ref[n], MAXB);
  // sigmas is a shared (non-batched) diffusion schedule -> bind directly, not tiled.
  A.allocOuts(MAXB); B.allocOuts(MAXB); C.allocOuts(MAXB);

  // scratch buffers sized for MAXB and a generous Fmax
  const int FCAP = 600;
  int32_t *lengths, *predDur, *startFrame, *totalFrames, *tokenOfFrame; float* speed;
  __half *asr, *en, *har;
  CUDA_CHECK(cudaMalloc(&lengths, MAXB * 4)); CUDA_CHECK(cudaMalloc(&speed, MAXB * 4));
  CUDA_CHECK(cudaMalloc(&predDur, (size_t)MAXB * T * 4)); CUDA_CHECK(cudaMalloc(&startFrame, (size_t)MAXB * T * 4));
  CUDA_CHECK(cudaMalloc(&totalFrames, MAXB * 4)); CUDA_CHECK(cudaMalloc(&tokenOfFrame, (size_t)MAXB * FCAP * 4));
  CUDA_CHECK(cudaMalloc(&asr, (size_t)MAXB * 512 * FCAP * 2)); CUDA_CHECK(cudaMalloc(&en, (size_t)MAXB * Cd * FCAP * 2));
  CUDA_CHECK(cudaMalloc(&har, (size_t)MAXB * 22 * (FCAP * 120 + 1) * 2));
  std::vector<int32_t> hLen(MAXB, T); std::vector<float> hSpeed(MAXB, 1.f);
  CUDA_CHECK(cudaMemcpy(lengths, hLen.data(), MAXB * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(speed, hSpeed.data(), MAXB * 4, cudaMemcpyHostToDevice));

  auto once = [&](int Bn) {
    A.setIn("tokens", {Bn, T}, in["tokens"]); A.setIn("mask", {Bn, T}, in["mask"]);
    A.setIn("ref_s", {Bn, 256}, in["ref_s"]); A.setIn("noise", {Bn, 1, 256}, in["noise"]);
    A.setIn("step_noise", {Bn, (int)ref["step_noise"].dims[1], 1, 256}, in["step_noise"]);
    A.setIn("alpha", {Bn, 1}, in["alpha"]); A.setIn("beta", {Bn, 1}, in["beta"]);
    A.setIn("sigmas", ref["sigmas"].dims, ref["sigmas"].dptr);
    if (!A.ctx->enqueueV3(stream)) std::abort();
    glue::duration_prefix((const __half*)A.out["dur"], lengths, speed, predDur, startFrame, totalFrames, Bn, T, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int32_t> hTot(Bn); CUDA_CHECK(cudaMemcpy(hTot.data(), totalFrames, Bn * 4, cudaMemcpyDeviceToHost));
    int Fmax = *std::max_element(hTot.begin(), hTot.end());
    glue::build_token_of_frame(predDur, startFrame, totalFrames, tokenOfFrame, Bn, T, Fmax, stream);
    glue::align_expand((const __half*)A.out["t_en"], (const __half*)A.out["d"], tokenOfFrame, totalFrames, asr, en, Bn, T, Cd, Fmax, stream);
    B.setIn("en", {Bn, Cd, Fmax}, en); B.setIn("s", {Bn, 128}, A.out["s"]);
    if (!B.ctx->enqueueV3(stream)) std::abort();
    glue::source_to_har((const __half*)B.out["f0"], (const __half*)glueW["linW"].dptr, (const __half*)glueW["linB"].dptr, har, Bn, Fmax, 1234, 1, stream);
    C.setIn("asr", {Bn, 512, Fmax}, asr); C.setIn("f0", {Bn, 2 * Fmax}, B.out["f0"]);
    C.setIn("noise", {Bn, 2 * Fmax}, B.out["noise"]); C.setIn("style", {Bn, 128}, A.out["ref"]);
    C.setIn("har", {Bn, 22, Fmax * 120 + 1}, har);
    if (!C.ctx->enqueueV3(stream)) std::abort();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return Fmax;
  };

  // Phase-1 (A + duration prefix) and phase-2 (align + B + source + C) as replayable
  // work, with the frame-count read between them. For a fixed input Fmax is constant,
  // so the two phases capture into CUDA graphs and the mid-pipeline host sync drops out
  // of the steady-state loop (production would keep one graph per frame-count bucket).
  auto setShapes = [&](int Bn) {
    A.setIn("tokens", {Bn, T}, in["tokens"]); A.setIn("mask", {Bn, T}, in["mask"]);
    A.setIn("ref_s", {Bn, 256}, in["ref_s"]); A.setIn("noise", {Bn, 1, 256}, in["noise"]);
    A.setIn("step_noise", {Bn, (int)ref["step_noise"].dims[1], 1, 256}, in["step_noise"]);
    A.setIn("alpha", {Bn, 1}, in["alpha"]); A.setIn("beta", {Bn, 1}, in["beta"]);
    A.setIn("sigmas", ref["sigmas"].dims, ref["sigmas"].dptr);
  };
  auto phaseA = [&](int Bn) {
    A.ctx->enqueueV3(stream);
    glue::duration_prefix((const __half*)A.out["dur"], lengths, speed, predDur, startFrame, totalFrames, Bn, T, stream);
  };
  auto phaseBC = [&](int Bn, int Fmax) {
    glue::build_token_of_frame(predDur, startFrame, totalFrames, tokenOfFrame, Bn, T, Fmax, stream);
    glue::align_expand((const __half*)A.out["t_en"], (const __half*)A.out["d"], tokenOfFrame, totalFrames, asr, en, Bn, T, Cd, Fmax, stream);
    B.setIn("en", {Bn, Cd, Fmax}, en); B.setIn("s", {Bn, 128}, A.out["s"]);
    B.ctx->enqueueV3(stream);
    glue::source_to_har((const __half*)B.out["f0"], (const __half*)glueW["linW"].dptr, (const __half*)glueW["linB"].dptr, har, Bn, Fmax, 1234, 1, stream);
    C.setIn("asr", {Bn, 512, Fmax}, asr); C.setIn("f0", {Bn, 2 * Fmax}, B.out["f0"]);
    C.setIn("noise", {Bn, 2 * Fmax}, B.out["noise"]); C.setIn("style", {Bn, 128}, A.out["ref"]);
    C.setIn("har", {Bn, 22, Fmax * 120 + 1}, har);
    C.ctx->enqueueV3(stream);
  };

  int Fm = once(1);
  printf("Full pipeline (150 chars, %d frames ~ %.1fs audio)\n", Fm, Fm * 600 / 24000.0);
  printf(" batch |  enqueue ms/item |  cudagraph ms/item\n");
  for (int Bn : {1, 2, 4, 8, 16}) {
    // ---- enqueue (no graph) ----
    for (int i = 0; i < 5; ++i) once(Bn);
    int N = 30;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) once(Bn);
    auto t1 = std::chrono::high_resolution_clock::now();
    double msE = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;

    // ---- cuda graphs ----
    setShapes(Bn);
    phaseA(Bn); CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int32_t> hTot(Bn); cudaMemcpy(hTot.data(), totalFrames, Bn * 4, cudaMemcpyDeviceToHost);
    int Fmax = *std::max_element(hTot.begin(), hTot.end());
    // shapes for BC must be set before capture (setInputShape is illegal during capture)
    B.setIn("en", {Bn, Cd, Fmax}, en); B.setIn("s", {Bn, 128}, A.out["s"]);
    C.setIn("asr", {Bn, 512, Fmax}, asr); C.setIn("f0", {Bn, 2 * Fmax}, B.out["f0"]);
    C.setIn("noise", {Bn, 2 * Fmax}, B.out["noise"]); C.setIn("style", {Bn, 128}, A.out["ref"]);
    C.setIn("har", {Bn, 22, Fmax * 120 + 1}, har);
    cudaGraph_t gA, gBC; cudaGraphExec_t geA, geBC;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    phaseA(Bn); CUDA_CHECK(cudaStreamEndCapture(stream, &gA)); CUDA_CHECK(cudaGraphInstantiate(&geA, gA, 0));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    phaseBC(Bn, Fmax); CUDA_CHECK(cudaStreamEndCapture(stream, &gBC)); CUDA_CHECK(cudaGraphInstantiate(&geBC, gBC, 0));
    for (int i = 0; i < 5; ++i) { cudaGraphLaunch(geA, stream); cudaGraphLaunch(geBC, stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) { cudaGraphLaunch(geA, stream); cudaGraphLaunch(geBC, stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    t1 = std::chrono::high_resolution_clock::now();
    double msG = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
    cudaGraphExecDestroy(geA); cudaGraphExecDestroy(geBC); cudaGraphDestroy(gA); cudaGraphDestroy(gBC);
    printf("  %3d  |  %10.3f (%6.3f/item) |  %10.3f (%6.3f/item)\n", Bn, msE, msE / Bn, msG, msG / Bn);
  }
  return 0;
}
