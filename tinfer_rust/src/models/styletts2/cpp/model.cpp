#include "tinfer_rust/src/models/base/native.rs.h"
#include "tinfer_rust/src/models/base/cpp/tensor.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/cpu/glue.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/model.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#ifdef TINFER_CUDA
#include <cuda_runtime.h>
#endif

namespace tinfer::native {
namespace {
Tensors merge(const Tensors& first, const Tensors& second) {
  Tensors result = first;
  result.insert(second.begin(), second.end());
  return result;
}

const Buffer& require(const Tensors& tensors, std::string_view name, DType dtype) {
  const auto found = tensors.find(name);
  if (found == tensors.end() || found->second.dtype != dtype) throw std::runtime_error("missing or invalid native tensor: " + std::string(name));
  return found->second;
}

Tensors host_inputs(const Batch& batch) {
  Tensors tensors;
  for (const auto& input : batch.tensors) {
    validate_tensor(input);
    std::vector<std::int64_t> shape(input.shape.begin(), input.shape.end());
    auto tensor = allocate(input.dtype, std::move(shape), -1);
    std::memcpy(tensor.data(), input.data.data(), input.data.size());
    tensors.emplace(std::string(input.name), std::move(tensor));
  }
  return tensors;
}

Tensor host_output(const std::string& name, DType dtype,
                   std::vector<std::int64_t> shape, const void* data,
                   std::size_t bytes) {
  Tensor tensor;
  tensor.name = name;
  tensor.dtype = dtype;
  for (const auto dimension : shape) tensor.shape.push_back(dimension);
  tensor.data.reserve(bytes);
  const auto* source = static_cast<const std::uint8_t*>(data);
  for (std::size_t index = 0; index < bytes; ++index) tensor.data.push_back(source[index]);
  return tensor;
}
}  // namespace

StyleTts2Model::StyleTts2Model(const std::string& root,
                               const std::string& architecture,
                               Backend backend, std::int32_t device)
    : device_(device) {
  auto backend_dir = std::filesystem::path(root);
  if (backend == Backend::Onnx) backend_dir /= device < 0 ? "onnx/cpu" : "onnx/cuda";
  else backend_dir /= "tensorrt";
  const auto extension = backend == Backend::Onnx ? ".onnx" : ".engine";
  program_a_ = load_program(architecture, "A", (backend_dir / ("A" + std::string(extension))).string(), backend, device);
  program_b_ = load_program(architecture, "B", (backend_dir / ("B" + std::string(extension))).string(), backend, device);
  program_c_ = load_program(architecture, "C", (backend_dir / ("C" + std::string(extension))).string(), backend, device);
  weights_a_ = load_bundle((backend_dir / "A.tinf").string(), device);
  weights_b_ = load_bundle((backend_dir / "B.tinf").string(), device);
  weights_c_ = load_bundle((backend_dir / "C.tinf").string(), device);
  glue_weights_ = load_bundle((backend_dir / "glue.tinf").string(), device);
  if (device >= 0) {
#ifdef TINFER_CUDA
    if (cudaSetDevice(device) != cudaSuccess || cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream_)) != cudaSuccess) {
      throw std::runtime_error("cannot create StyleTTS2 CUDA stream");
    }
#else
    throw std::runtime_error("CUDA support is not enabled in this build");
#endif
  }
  execution_a_ = program_a_->create_execution(stream_);
  execution_b_ = program_b_->create_execution(stream_);
  execution_c_ = program_c_->create_execution(stream_);
}

Output StyleTts2Model::run(const Batch& batch) const {
  return device_ < 0 ? run_cpu(batch) : run_cuda(batch);
}

StyleTts2Model::~StyleTts2Model() {
#ifdef TINFER_CUDA
  if (stream_ != nullptr) {
    cudaSetDevice(device_);
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
    execution_c_.reset();
    execution_b_.reset();
    execution_a_.reset();
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
  }
#endif
}

Output StyleTts2Model::run_cpu(const Batch& batch) const {
  const auto inputs = host_inputs(batch);
  const auto tokens = require(inputs, "tokens", DType::I64);
  if (tokens.shape.size() != 2) throw std::runtime_error("StyleTTS2 tokens must be rank two");
  const auto batch_size = static_cast<std::int32_t>(tokens.shape[0]);
  const auto token_count = static_cast<std::int32_t>(tokens.shape[1]);
  auto a = execution_a_->run(merge(weights_a_, inputs), nullptr);
  const auto& durations = require(a, "dur", DType::F32);
  const auto& lengths = require(inputs, "lengths", DType::I32);
  const auto& speeds = require(inputs, "speed", DType::F32);
  const auto alignment = styletts2::cpu::duration_prefix(static_cast<float*>(durations.data()), static_cast<std::int32_t*>(lengths.data()),
                                                         static_cast<float*>(speeds.data()), batch_size, token_count);
  std::vector<float> asr;
  std::vector<float> encoding;
  const auto& text = require(a, "t_en", DType::F32);
  const auto& encoded = require(a, "d", DType::F32);
  const auto channels = static_cast<std::int32_t>(encoded.shape[2]);
  styletts2::cpu::align_expand(static_cast<float*>(text.data()), static_cast<float*>(encoded.data()), alignment,
                              batch_size, token_count, channels, asr, encoding);
  auto en = allocate(DType::F32, {batch_size, channels, alignment.frames}, -1);
  std::memcpy(en.data(), encoding.data(), en.bytes);
  Tensors b_inputs = weights_b_;
  b_inputs.emplace("en", std::move(en));
  b_inputs.emplace("s", require(a, "s", DType::F32));
  auto b = execution_b_->run(b_inputs, nullptr);
  const auto& f0 = require(b, "f0", DType::F32);
  const auto& linear = require(glue_weights_, "linW", DType::F32);
  const auto& bias = require(glue_weights_, "linB", DType::F32);
  const auto& seeds = require(inputs, "seeds", DType::I64);
  auto harmonic = styletts2::cpu::source_to_har(static_cast<float*>(f0.data()), static_cast<float*>(linear.data()),
                                                *static_cast<float*>(bias.data()), batch_size, alignment.frames,
                                                static_cast<std::uint64_t*>(seeds.data()), true);
  auto c_inputs = weights_c_;
  auto asr_buffer = allocate(DType::F32, {batch_size, 512, alignment.frames}, -1);
  std::memcpy(asr_buffer.data(), asr.data(), asr_buffer.bytes);
  auto har = allocate(DType::F32, {batch_size, 22, alignment.frames * 120 + 1}, -1);
  std::memcpy(har.data(), harmonic.data(), har.bytes);
  c_inputs.emplace("asr", std::move(asr_buffer));
  c_inputs.emplace("f0", f0);
  c_inputs.emplace("noise", require(b, "noise", DType::F32));
  c_inputs.emplace("style", require(a, "ref", DType::F32));
  c_inputs.emplace("har", std::move(har));
  auto c = execution_c_->run(c_inputs, nullptr);
  const auto& audio = require(c, "audio", DType::F32);
  Output output;
  output.tensors.push_back(host_output("audio", audio.dtype, audio.shape, audio.data(), audio.bytes));
  output.tensors.push_back(host_output("durations", DType::I32, {batch_size, token_count}, alignment.durations.data(), alignment.durations.size() * sizeof(std::int32_t)));
  output.tensors.push_back(host_output("frames", DType::I32, {batch_size}, alignment.totals.data(), alignment.totals.size() * sizeof(std::int32_t)));
  const auto& references = require(a, "ref", DType::F32);
  const auto& styles = require(a, "s", DType::F32);
  std::vector<float> state(static_cast<std::size_t>(batch_size) * 256);
  for (std::int32_t item = 0; item < batch_size; ++item) {
    std::copy_n(static_cast<float*>(references.data()) + static_cast<std::size_t>(item) * 128, 128,
                state.begin() + static_cast<std::size_t>(item) * 256);
    std::copy_n(static_cast<float*>(styles.data()) + static_cast<std::size_t>(item) * 128, 128,
                state.begin() + static_cast<std::size_t>(item) * 256 + 128);
  }
  output.tensors.push_back(host_output("style", DType::F32, {batch_size, 256}, state.data(), state.size() * sizeof(float)));
  return output;
}

std::unique_ptr<Model> load_styletts2(rust::Str root, rust::Str architecture,
                                      std::uint8_t backend, std::int32_t device) {
  if (backend > static_cast<std::uint8_t>(Backend::TensorRt)) throw std::invalid_argument("invalid native backend");
  return std::make_unique<StyleTts2Model>(std::string(root), std::string(architecture), static_cast<Backend>(backend), device);
}

}  // namespace tinfer::native
