#pragma once

#include "tinfer_rust/src/models/base/native.rs.h"

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace tinfer::native {

const Tensor& find_tensor(const Batch& batch, std::string_view name);
std::size_t element_size(DType dtype);
std::size_t element_count(const Tensor& tensor);
void validate_tensor(const Tensor& tensor);

template <typename T>
const T* data(const Tensor& tensor) {
  validate_tensor(tensor);
  return reinterpret_cast<const T*>(tensor.data.data());
}

}  // namespace tinfer::native
