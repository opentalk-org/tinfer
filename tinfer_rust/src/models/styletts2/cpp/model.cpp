#include "tinfer_rust/src/models/base/native.rs.h"
#include "tinfer_rust/src/models/base/cpp/tensor.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/model.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#ifdef TINFER_CUDA
#include <cuda_runtime.h>
#endif

namespace tinfer::native {
StyleTts2Model::StyleTts2Model(const std::string& root,
                               const std::string& architecture,
                               Backend backend, std::int32_t device,
                               std::int32_t max_batch)
    : backend_(backend), device_(device), max_batch_(max_batch) {
  auto backend_dir = std::filesystem::path(root);
  if (backend == Backend::Onnx) backend_dir /= device < 0 ? "onnx/cpu" : "onnx/cuda";
  else backend_dir /= "tensorrt";
  const auto extension = backend == Backend::Onnx ? ".onnx" : ".engine";
  program_a_ = load_program(architecture, "A", (backend_dir / ("A" + std::string(extension))).string(), backend, device);
  program_bc_ = load_program(architecture, "BC", (backend_dir / ("BC" + std::string(extension))).string(), backend, device);
  weights_a_ = load_bundle((backend_dir / "A.tinf").string(), device);
  weights_bc_ = load_bundle((backend_dir / "BC.tinf").string(), device);
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
  execution_bc_ = program_bc_->create_execution(stream_);
}

Output StyleTts2Model::run(const Batch& batch) const {
  const auto operation = std::find_if(batch.tensors.begin(), batch.tensors.end(),
                                      [](const Tensor& tensor) { return tensor.name == "operation"; });
  if (operation == batch.tensors.end() || operation->dtype != DType::I32 || operation->data.size() != sizeof(std::int32_t)) {
    throw std::invalid_argument("StyleTTS2 request is missing its operation");
  }
  const auto value = *reinterpret_cast<const std::int32_t*>(operation->data.data());
  if (value == 0) return start(batch);
  if (value == 1) return continue_generation(batch);
  if (value == 2) return close(batch);
  throw std::invalid_argument("invalid StyleTTS2 operation");
}

Output StyleTts2Model::close(const Batch& batch) const {
  const auto ids = std::find_if(batch.tensors.begin(), batch.tensors.end(),
                                [](const Tensor& tensor) { return tensor.name == "stream_ids"; });
  if (ids == batch.tensors.end() || ids->dtype != DType::I64 ||
      ids->shape.size() != 1 ||
      ids->data.size() != static_cast<std::size_t>(ids->shape[0]) * sizeof(std::int64_t)) {
    throw std::invalid_argument("StyleTTS2 close has invalid stream IDs");
  }
  const auto* values = reinterpret_cast<const std::int64_t*>(ids->data.data());
  std::lock_guard lock(sessions_mutex_);
  for (std::int64_t index = 0; index < ids->shape[0]; ++index) {
    sessions_.erase(values[index]);
    tails_.erase(values[index]);
  }
  return {};
}

StyleTts2Model::~StyleTts2Model() {
#ifdef TINFER_CUDA
  if (stream_ != nullptr) {
    cudaSetDevice(device_);
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
    if (download_staging_ != nullptr) cudaFreeHost(download_staging_);
    execution_bc_.reset();
    execution_a_.reset();
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
  }
#endif
}

std::unique_ptr<Model> load_styletts2(rust::Str root, rust::Str architecture,
                                      std::uint8_t backend, std::int32_t device,
                                      std::int32_t max_batch) {
  if (backend > static_cast<std::uint8_t>(Backend::TensorRt)) throw std::invalid_argument("invalid native backend");
  return std::make_unique<StyleTts2Model>(std::string(root), std::string(architecture), static_cast<Backend>(backend), device, max_batch);
}

}  // namespace tinfer::native
