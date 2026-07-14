#include "tinfer_rust/src/models/base/cpp/onnx.hpp"
#include "tinfer_rust/src/models/base/cpp/engine.hpp"

#ifdef TINFER_ONNX
#include <onnxruntime_cxx_api.h>
#ifdef TINFER_CUDA
#include <cuda_runtime.h>
#endif
#endif

#include <cstring>
#include <stdexcept>

namespace tinfer::native {
#ifdef TINFER_ONNX
namespace {
ONNXTensorElementDataType dtype(DType value) {
  switch (value) {
    case DType::F16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DType::F32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DType::I32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DType::I64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DType::Bool: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default: throw std::runtime_error("unsupported ONNX input dtype");
  }
}

DType dtype(ONNXTensorElementDataType value) {
  switch (value) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return DType::F16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return DType::F32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return DType::I32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return DType::I64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return DType::Bool;
    default: throw std::runtime_error("unsupported ONNX output dtype");
  }
}

Ort::MemoryInfo memory_info(std::int32_t device) {
  if (device < 0) return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  return Ort::MemoryInfo("Cuda", OrtDeviceAllocator, device, OrtMemTypeDefault);
}

class OnnxProgram;

class OnnxExecution final : public Execution {
 public:
  explicit OnnxExecution(std::shared_ptr<const OnnxProgram> program) : program_(std::move(program)) {}
  std::optional<DType> input_dtype(const std::string& name) const override;
  Tensors run(const Tensors& inputs, void* stream) override;

 private:
  std::shared_ptr<const OnnxProgram> program_;
};

class OnnxProgram final : public Program, public std::enable_shared_from_this<OnnxProgram> {
 public:
  OnnxProgram(const std::string& path, std::int32_t device)
      : device_(device), options_(), session_(nullptr) {
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (device >= 0) {
#ifdef TINFER_CUDA
      OrtCUDAProviderOptions cuda{};
      cuda.device_id = device;
      options_.AppendExecutionProvider_CUDA(cuda);
#else
      throw std::runtime_error("ONNX CUDA support is not enabled in this build");
#endif
    }
    session_ = Ort::Session(environment(), path.c_str(), options_);
  }

  std::unique_ptr<Execution> create_execution(void*) const override {
    return std::make_unique<OnnxExecution>(shared_from_this());
  }

  static Ort::Env& environment() {
    static Ort::Env value(ORT_LOGGING_LEVEL_WARNING, "tinfer");
    return value;
  }

  std::int32_t device_;
  Ort::SessionOptions options_;
  mutable Ort::Session session_;
};

Tensors OnnxExecution::run(const Tensors& inputs, void* stream) {
#ifndef TINFER_CUDA
  (void)stream;
#endif
  auto memory = memory_info(program_->device_);
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<Ort::AllocatedStringPtr> input_names;
  std::vector<const char*> input_name_views;
  std::vector<Ort::Value> values;
  const auto input_count = program_->session_.GetInputCount();
  input_names.reserve(input_count);
  input_name_views.reserve(input_count);
  values.reserve(input_count);
  for (std::size_t index = 0; index < input_count; ++index) {
    input_names.push_back(program_->session_.GetInputNameAllocated(index, allocator));
    const auto* name = input_names.back().get();
    const auto found = inputs.find(name);
    if (found == inputs.end()) throw std::runtime_error("missing ONNX input: " + std::string(name));
    input_name_views.push_back(name);
    const auto& tensor = found->second;
    values.push_back(Ort::Value::CreateTensor(memory, tensor.data(), tensor.bytes,
                                               tensor.shape.data(), tensor.shape.size(),
                                               dtype(tensor.dtype)));
  }
  std::vector<Ort::AllocatedStringPtr> output_names;
  std::vector<const char*> output_name_views;
  const auto output_count = program_->session_.GetOutputCount();
  output_names.reserve(output_count);
  output_name_views.reserve(output_count);
  for (std::size_t index = 0; index < output_count; ++index) {
    output_names.push_back(program_->session_.GetOutputNameAllocated(index, allocator));
    output_name_views.push_back(output_names.back().get());
  }
  auto outputs = program_->session_.Run(Ort::RunOptions{nullptr}, input_name_views.data(),
                                        values.data(), values.size(), output_name_views.data(),
                                        output_name_views.size());
  Tensors result;
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    auto info = outputs[index].GetTensorTypeAndShapeInfo();
    auto output = allocate(dtype(info.GetElementType()), info.GetShape(), program_->device_);
    if (program_->device_ < 0) {
      std::memcpy(output.data(), outputs[index].GetTensorRawData(), output.bytes);
    } else {
#ifdef TINFER_CUDA
      if (cudaMemcpyAsync(output.data(), outputs[index].GetTensorRawData(), output.bytes,
                          cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)) != cudaSuccess) {
        throw std::runtime_error("ONNX CUDA output copy failed");
      }
#endif
    }
    result.emplace(output_name_views[index], std::move(output));
  }
  return result;
}

std::optional<DType> OnnxExecution::input_dtype(const std::string& name) const {
  Ort::AllocatorWithDefaultOptions allocator;
  for (std::size_t index = 0; index < program_->session_.GetInputCount(); ++index) {
    const auto input = program_->session_.GetInputNameAllocated(index, allocator);
    if (name == input.get()) return dtype(program_->session_.GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetElementType());
  }
  return std::nullopt;
}
}  // namespace

std::shared_ptr<Program> load_onnx(const std::string& path, std::int32_t device) {
  return std::make_shared<OnnxProgram>(path, device);
}
#else
std::shared_ptr<Program> load_onnx(const std::string&, std::int32_t) {
  throw std::runtime_error("ONNX Runtime support is not enabled in this build");
}
#endif

}  // namespace tinfer::native
