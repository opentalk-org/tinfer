#include "tinfer_rust/src/models/base/cpp/engine.hpp"
#include "tinfer_rust/src/models/base/cpp/onnx.hpp"
#include "tinfer_rust/src/models/base/cpp/tensorrt.hpp"

#ifdef TINFER_CUDA
#include <cuda_runtime.h>
#endif

#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace tinfer::native {
namespace {
std::mutex cache_mutex;
std::map<std::string, std::weak_ptr<Program>> cache;

std::size_t checked_count(const std::vector<std::int64_t>& shape) {
  std::size_t count = 1;
  for (const auto dimension : shape) {
    if (dimension < 0 || (dimension > 0 && count > std::numeric_limits<std::size_t>::max() / dimension)) {
      throw std::invalid_argument("invalid native tensor shape");
    }
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

template <typename T>
T read(std::ifstream& input) {
  T value;
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) throw std::runtime_error("truncated TINF bundle");
  return value;
}

std::string read_string(std::ifstream& input) {
  const auto length = read<std::int32_t>(input);
  if (length < 1) throw std::runtime_error("invalid TINF tensor name");
  std::string value(length, '\0');
  input.read(value.data(), length);
  if (!input) throw std::runtime_error("truncated TINF tensor name");
  return value;
}
}  // namespace

Buffer allocate(DType dtype, std::vector<std::int64_t> shape, std::int32_t device) {
  const auto bytes = checked_count(shape) * element_size(dtype);
  void* pointer = nullptr;
  if (device >= 0) {
#ifdef TINFER_CUDA
    if (cudaSetDevice(device) != cudaSuccess || cudaMalloc(&pointer, bytes) != cudaSuccess) {
      throw std::runtime_error("CUDA tensor allocation failed");
    }
#else
    throw std::runtime_error("CUDA support is not enabled in this build");
#endif
  } else {
    pointer = ::operator new(bytes);
  }
  auto release = [device](void* value) {
    if (device >= 0) {
#ifdef TINFER_CUDA
      cudaSetDevice(device);
      cudaFree(value);
#endif
    } else {
      ::operator delete(value);
    }
  };
  return Buffer{dtype, std::move(shape), std::shared_ptr<void>(pointer, release), bytes};
}

Tensors load_bundle(const std::string& path, std::int32_t device) {
  std::ifstream input(path, std::ios::binary);
  char magic[4];
  input.read(magic, sizeof(magic));
  if (!input || std::memcmp(magic, "TINF", sizeof(magic)) != 0) throw std::runtime_error("invalid TINF bundle: " + path);
  const auto tensor_count = read<std::int32_t>(input);
  if (tensor_count < 0) throw std::runtime_error("invalid TINF tensor count");
  Tensors tensors;
  for (std::int32_t index = 0; index < tensor_count; ++index) {
    const auto name = read_string(input);
    const auto dtype = static_cast<DType>(read<std::uint8_t>(input));
    input.seekg(3, std::ios::cur);
    const auto rank = read<std::int32_t>(input);
    if (rank < 0 || rank > 16) throw std::runtime_error("invalid TINF tensor rank");
    std::vector<std::int64_t> shape(rank);
    for (auto& dimension : shape) dimension = read<std::int64_t>(input);
    auto tensor = allocate(dtype, std::move(shape), device);
    std::vector<std::uint8_t> host(tensor.bytes);
    input.read(reinterpret_cast<char*>(host.data()), host.size());
    if (!input) throw std::runtime_error("truncated TINF tensor data");
    if (device >= 0) {
#ifdef TINFER_CUDA
      if (cudaMemcpy(tensor.data(), host.data(), host.size(), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("CUDA weight upload failed");
#endif
    } else {
      std::memcpy(tensor.data(), host.data(), host.size());
    }
    if (!tensors.emplace(name, std::move(tensor)).second) throw std::runtime_error("duplicate TINF tensor: " + name);
  }
  return tensors;
}

std::shared_ptr<Program> load_program(const std::string& architecture,
                                      const std::string& stage,
                                      const std::string& path, Backend backend,
                                      std::int32_t device) {
  const auto key = architecture + ':' + std::to_string(static_cast<int>(backend)) + ':' + std::to_string(device) + ':' + stage;
  std::lock_guard lock(cache_mutex);
  if (const auto found = cache[key].lock()) return found;
  auto program = backend == Backend::Onnx ? load_onnx(path, device) : load_tensorrt(path, device);
  cache[key] = program;
  return program;
}

}  // namespace tinfer::native
