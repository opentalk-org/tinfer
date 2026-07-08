// tinfer C++/CUDA runtime: full StyleTTS2 pipeline across 3 weight-input TensorRT
// engines glued by CUDA kernels.  A -> glue1(align) -> B -> glue2(source) -> C.
//   Usage: pipeline <engines_dir> [out.wav]
// Inputs come from A.ref (the real 150-char utterance). Weights bound once.
#include "common.h"
#include "kernels.cuh"
#include <cmath>
#include <algorithm>
#include <map>
#include <memory>
#include <vector>

using namespace nvinfer1;
using tinfer::Tensor;
static tinfer::Logger gLogger;

struct Engine {
  std::unique_ptr<IRuntime> rt;
  std::unique_ptr<ICudaEngine> eng;
  std::unique_ptr<IExecutionContext> ctx;
  std::vector<std::string> inputs, outputs;
  std::map<std::string, void*> out;
  Engine(const std::string& path, tinfer::Logger& lg) {
    auto b = tinfer::readFile(path);
    rt.reset(createInferRuntime(lg));
    eng.reset(rt->deserializeCudaEngine(b.data(), b.size()));
    ctx.reset(eng->createExecutionContext());
    for (int i = 0; i < eng->getNbIOTensors(); ++i) {
      const char* n = eng->getIOTensorName(i);
      (eng->getTensorIOMode(n) == TensorIOMode::kINPUT ? inputs : outputs).push_back(n);
    }
  }
  void bindWeights(std::map<std::string, Tensor>& w) {
    for (auto& n : inputs) {
      auto it = w.find(n);
      if (it == w.end()) continue;
      ctx->setInputShape(n.c_str(), tinfer::toDims(it->second.dims));
      ctx->setTensorAddress(n.c_str(), it->second.dptr);
    }
  }
  void setIn(const std::string& n, std::vector<int64_t> d, void* p) {
    ctx->setInputShape(n.c_str(), tinfer::toDims(d));
    ctx->setTensorAddress(n.c_str(), p);
  }
  void allocOuts() {
    for (auto& n : outputs) {
      Dims s = ctx->getTensorShape(n.c_str());
      int64_t ne = 1; for (int i = 0; i < s.nbDims; ++i) ne *= s.d[i];
      if (out.count(n)) cudaFree(out[n]);
      cudaMalloc(&out[n], std::max<int64_t>(ne, 1) * 2);
      ctx->setTensorAddress(n.c_str(), out[n]);
    }
  }
  void run(cudaStream_t s) { if (!ctx->enqueueV3(s)) { fprintf(stderr, "enqueue fail\n"); std::abort(); } }
};

static void writeWav(const std::string& path, const std::vector<float>& audio, int sr) {
  std::vector<int16_t> pcm(audio.size());
  for (size_t i = 0; i < audio.size(); ++i) {
    float v = std::max(-1.f, std::min(1.f, audio[i]));
    pcm[i] = (int16_t)lroundf(v * 32767.f);
  }
  int32_t dataBytes = pcm.size() * 2, fmtChunk = 16, byteRate = sr * 2, sr_ = sr;
  int16_t chans = 1, bits = 16, blockAlign = 2, audioFmt = 1;
  int32_t riff = 36 + dataBytes;
  FILE* f = fopen(path.c_str(), "wb");
  fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVE", 1, 4, f);
  fwrite("fmt ", 1, 4, f); fwrite(&fmtChunk, 4, 1, f); fwrite(&audioFmt, 2, 1, f);
  fwrite(&chans, 2, 1, f); fwrite(&sr_, 4, 1, f); fwrite(&byteRate, 4, 1, f);
  fwrite(&blockAlign, 2, 1, f); fwrite(&bits, 2, 1, f);
  fwrite("data", 1, 4, f); fwrite(&dataBytes, 4, 1, f); fwrite(pcm.data(), 2, pcm.size(), f);
  fclose(f);
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "/workspace/tinfer/runtime/engines";
  std::string wav = argc > 2 ? argv[2] : "/workspace/tinfer/runtime/out.wav";
  cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

  Engine A(dir + "/A.engine", gLogger), B(dir + "/B.engine", gLogger), C(dir + "/C.engine", gLogger);
  auto wA = tinfer::loadBundle(dir + "/A.weights");
  auto wB = tinfer::loadBundle(dir + "/B.weights");
  auto wC = tinfer::loadBundle(dir + "/C.weights");
  auto glueW = tinfer::loadBundle(dir + "/glue.weights");
  auto ref = tinfer::loadBundle(dir + "/A.ref");
  A.bindWeights(wA); B.bindWeights(wB); C.bindWeights(wC);

  int Bn = ref["tokens"].dims[0], T = ref["tokens"].dims[1];
  int Cd = 640;
  printf("batch=%d tokens=%d\n", Bn, T);

  // ---------- A ----------
  for (const char* n : {"tokens", "mask", "ref_s", "noise", "step_noise", "alpha", "beta", "sigmas"})
    A.setIn(n, ref[n].dims, ref[n].dptr);
  A.allocOuts();
  A.run(stream);

  // ---------- glue1: durations -> alignment ----------
  int32_t *lengths, *speedI;
  float* speed;
  int32_t *predDur, *startFrame, *totalFrames, *tokenOfFrame;
  CUDA_CHECK(cudaMalloc(&lengths, Bn * 4));
  CUDA_CHECK(cudaMalloc(&speed, Bn * 4));
  std::vector<int32_t> hLen(Bn, T);
  std::vector<float> hSpeed(Bn, 1.0f);
  CUDA_CHECK(cudaMemcpy(lengths, hLen.data(), Bn * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(speed, hSpeed.data(), Bn * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&predDur, (size_t)Bn * T * 4));
  CUDA_CHECK(cudaMalloc(&startFrame, (size_t)Bn * T * 4));
  CUDA_CHECK(cudaMalloc(&totalFrames, Bn * 4));
  glue::duration_prefix((const __half*)A.out["dur"], lengths, speed, predDur, startFrame,
                        totalFrames, Bn, T, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<int32_t> hTot(Bn);
  CUDA_CHECK(cudaMemcpy(hTot.data(), totalFrames, Bn * 4, cudaMemcpyDeviceToHost));
  int Fmax = *std::max_element(hTot.begin(), hTot.end());
  printf("frames per item:"); for (int v : hTot) printf(" %d", v); printf("  Fmax=%d\n", Fmax);

  CUDA_CHECK(cudaMalloc(&tokenOfFrame, (size_t)Bn * Fmax * 4));
  glue::build_token_of_frame(predDur, startFrame, totalFrames, tokenOfFrame, Bn, T, Fmax, stream);
  __half *asr, *en;
  CUDA_CHECK(cudaMalloc(&asr, (size_t)Bn * 512 * Fmax * 2));
  CUDA_CHECK(cudaMalloc(&en, (size_t)Bn * Cd * Fmax * 2));
  glue::align_expand((const __half*)A.out["t_en"], (const __half*)A.out["d"], tokenOfFrame,
                     totalFrames, asr, en, Bn, T, Cd, Fmax, stream);

  // ---------- B: F0Ntrain ----------
  B.setIn("en", {Bn, Cd, Fmax}, en);
  B.setIn("s", ref.count("s") ? ref["s"].dims : std::vector<int64_t>{Bn, 128}, A.out["s"]);
  B.allocOuts();
  B.run(stream);

  // ---------- glue2: source module -> har ----------
  __half* har;
  int harFrames = Fmax * 120 + 1;
  CUDA_CHECK(cudaMalloc(&har, (size_t)Bn * 22 * harFrames * 2));
  glue::source_to_har((const __half*)B.out["f0"], (const __half*)glueW["linW"].dptr,
                      (const __half*)glueW["linB"].dptr, har, Bn, Fmax, 1234, 1, stream);

  // ---------- C: decoder + generator ----------
  C.setIn("asr", {Bn, 512, Fmax}, asr);
  C.setIn("f0", {Bn, 2 * Fmax}, B.out["f0"]);
  C.setIn("noise", {Bn, 2 * Fmax}, B.out["noise"]);
  C.setIn("style", {Bn, 128}, A.out["ref"]);
  C.setIn("har", {Bn, 22, harFrames}, har);
  C.allocOuts();
  C.run(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // ---------- write item 0 wav (trim to actual frames) ----------
  int samples = hTot[0] * 600;
  std::vector<uint16_t> ah(samples);
  CUDA_CHECK(cudaMemcpy(ah.data(), C.out["audio"], samples * 2, cudaMemcpyDeviceToHost));
  std::vector<float> audio(samples);
  for (int i = 0; i < samples; ++i) { __half h; std::memcpy(&h, &ah[i], 2); audio[i] = __half2float(h); }
  if (samples > 100) audio.resize(samples - 100);  // match pytorch tail trim
  writeWav(wav, audio, 24000);
  printf("wrote %s  (%zu samples, %.2fs)\n", wav.c_str(), audio.size(), audio.size() / 24000.0);
  return 0;
}
