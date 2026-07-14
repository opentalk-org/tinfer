#include "tinfer_rust/src/models/base/cpp/tensor.hpp"

#include <limits>

namespace tinfer::native {

const Tensor& find_tensor(const Batch& batch, std::string_view name) {
  for (const auto& tensor : batch.tensors) {
    if (std::string_view(tensor.name.data(), tensor.name.size()) == name) return tensor;
  }
  throw std::invalid_argument("missing tensor: " + std::string(name));
}

std::size_t element_size(DType dtype) {
  switch (dtype) {
    case DType::F16: return 2;
    case DType::F32:
    case DType::I32: return 4;
    case DType::I64: return 8;
    case DType::Bool: return 1;
    default: throw std::invalid_argument("invalid tensor dtype");
  }
}

std::size_t element_count(const Tensor& tensor) {
  std::size_t count = 1;
  for (const auto dimension : tensor.shape) {
    if (dimension < 0 || (dimension > 0 && count > std::numeric_limits<std::size_t>::max() / dimension)) {
      throw std::invalid_argument("invalid tensor shape");
    }
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

void validate_tensor(const Tensor& tensor) {
  if (element_count(tensor) * element_size(tensor.dtype) != tensor.data.size()) {
    throw std::invalid_argument("tensor byte length differs from shape");
  }
}

}  // namespace tinfer::native
