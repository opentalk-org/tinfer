#include "tinfer_rust/src/models/base/cpp/tensorrt.hpp"
#include "tinfer_rust/src/models/base/cpp/engine.hpp"

#ifdef TINFER_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime.h>
#endif

#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace tinfer::native {
#ifdef TINFER_TENSORRT
namespace {
class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", message);
  }
};

DType dtype(nvinfer1::DataType value) {
  switch (value) {
    case nvinfer1::DataType::kHALF: return DType::F16;
    case nvinfer1::DataType::kFLOAT: return DType::F32;
    case nvinfer1::DataType::kINT32: return DType::I32;
    case nvinfer1::DataType::kINT64: return DType::I64;
    case nvinfer1::DataType::kBOOL: return DType::Bool;
    default: throw std::runtime_error("unsupported TensorRT tensor dtype");
  }
}

nvinfer1::Dims dims(const std::vector<std::int64_t>& shape) {
  nvinfer1::Dims value;
  value.nbDims = static_cast<std::int32_t>(shape.size());
  if (value.nbDims > nvinfer1::Dims::MAX_DIMS) throw std::runtime_error("TensorRT tensor rank exceeds limit");
  for (std::int32_t index = 0; index < value.nbDims; ++index) value.d[index] = shape[index];
  return value;
}

std::vector<char> read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open TensorRT engine: " + path);
  const auto size = input.tellg();
  input.seekg(0);
  std::vector<char> bytes(size);
  input.read(bytes.data(), size);
  if (!input) throw std::runtime_error("cannot read TensorRT engine: " + path);
  return bytes;
}

class TrtProgram;

class TrtExecution final : public Execution {
 public:
  TrtExecution(std::shared_ptr<const TrtProgram> program, void* stream);
  std::optional<DType> input_dtype(const std::string& name) const override;
  Tensors run(const Tensors& inputs, void* stream) override;

 private:
  std::shared_ptr<const TrtProgram> program_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  Tensors outputs_;
  std::int64_t max_batch_ = 1;
};

class TrtProgram final : public Program, public std::enable_shared_from_this<TrtProgram> {
 public:
  TrtProgram(const std::string& path, std::int32_t device) : device_(device) {
    if (cudaSetDevice(device) != cudaSuccess) throw std::runtime_error("cannot select TensorRT CUDA device");
    const auto bytes = read_file(path);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("cannot create TensorRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine_) throw std::runtime_error("cannot deserialize TensorRT engine: " + path);
  }

  std::unique_ptr<Execution> create_execution(void* stream) const override {
    return std::make_unique<TrtExecution>(shared_from_this(), stream);
  }

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::int32_t device_;
};

TrtExecution::TrtExecution(std::shared_ptr<const TrtProgram> program, void*) : program_(std::move(program)) {
  if (cudaSetDevice(program_->device_) != cudaSuccess) throw std::runtime_error("cannot select TensorRT CUDA device");
  context_.reset(program_->engine_->createExecutionContext());
  if (!context_) throw std::runtime_error("cannot create TensorRT execution context");
  for (std::int32_t index = 0; index < program_->engine_->getNbIOTensors(); ++index) {
    const auto* name = program_->engine_->getIOTensorName(index);
    if (program_->engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
    const auto shape = program_->engine_->getTensorShape(name);
    if (shape.nbDims == 0 || shape.d[0] >= 0) continue;
    const auto maximum = program_->engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
    if (maximum.nbDims == shape.nbDims && maximum.d[0] > max_batch_) max_batch_ = maximum.d[0];
  }
}

std::optional<DType> TrtExecution::input_dtype(const std::string& name) const {
  for (std::int32_t index = 0; index < program_->engine_->getNbIOTensors(); ++index) {
    const auto* input = program_->engine_->getIOTensorName(index);
    if (name == input && program_->engine_->getTensorIOMode(input) == nvinfer1::TensorIOMode::kINPUT) {
      return dtype(program_->engine_->getTensorDataType(input));
    }
  }
  return std::nullopt;
}

Tensors TrtExecution::run(const Tensors& inputs, void* stream) {
  if (cudaSetDevice(program_->device_) != cudaSuccess) throw std::runtime_error("cannot select TensorRT CUDA device");
  for (std::int32_t index = 0; index < program_->engine_->getNbIOTensors(); ++index) {
    const auto* name = program_->engine_->getIOTensorName(index);
    if (program_->engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
    const auto found = inputs.find(name);
    if (found == inputs.end()) throw std::runtime_error("missing TensorRT input: " + std::string(name));
    const auto& tensor = found->second;
    if (tensor.dtype != dtype(program_->engine_->getTensorDataType(name)) ||
        !context_->setInputShape(name, dims(tensor.shape)) || !context_->setTensorAddress(name, tensor.data())) {
      throw std::runtime_error("cannot bind TensorRT input: " + std::string(name));
    }
  }
  Tensors outputs;
  for (std::int32_t index = 0; index < program_->engine_->getNbIOTensors(); ++index) {
    const auto* name = program_->engine_->getIOTensorName(index);
    if (program_->engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT) continue;
    const auto output_dims = context_->getTensorShape(name);
    std::vector<std::int64_t> shape(output_dims.d, output_dims.d + output_dims.nbDims);
    auto capacity_shape = shape;
    if (!capacity_shape.empty()) capacity_shape[0] = max_batch_;
    const auto output_dtype = dtype(program_->engine_->getTensorDataType(name));
    auto cached = outputs_.find(name);
    if (cached == outputs_.end()) {
      cached = outputs_.emplace(name, allocate(output_dtype, std::move(capacity_shape), program_->device_)).first;
    }
    std::size_t bytes = element_size(output_dtype);
    for (const auto dimension : shape) bytes *= static_cast<std::size_t>(dimension);
    if (cached->second.bytes < bytes) throw std::runtime_error("TensorRT output exceeds its profile capacity");
    auto output = Buffer{output_dtype, std::move(shape), cached->second.memory, bytes};
    if (!context_->setTensorAddress(name, output.data())) throw std::runtime_error("cannot bind TensorRT output: " + std::string(name));
    outputs.emplace(name, std::move(output));
  }
  if (!context_->enqueueV3(static_cast<cudaStream_t>(stream))) throw std::runtime_error("TensorRT enqueue failed");
  return outputs;
}
}  // namespace

std::shared_ptr<Program> load_tensorrt(const std::string& path, std::int32_t device) {
  if (device < 0) throw std::invalid_argument("TensorRT requires CUDA");
  return std::make_shared<TrtProgram>(path, device);
}
#else
std::shared_ptr<Program> load_tensorrt(const std::string&, std::int32_t) {
  throw std::runtime_error("TensorRT support is not enabled in this build");
}
#endif

}  // namespace tinfer::native
