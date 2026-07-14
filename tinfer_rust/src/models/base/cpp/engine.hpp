#pragma once

#include "tinfer_rust/src/models/base/cpp/tensor.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tinfer::native {

enum class Backend : std::uint8_t { Onnx, TensorRt };

struct Buffer {
  DType dtype;
  std::vector<std::int64_t> shape;
  std::shared_ptr<void> memory;
  std::size_t bytes;

  void* data() const { return memory.get(); }
};

using Tensors = std::map<std::string, Buffer, std::less<>>;

class Execution {
 public:
  virtual ~Execution() = default;
  virtual std::optional<DType> input_dtype(const std::string& name) const = 0;
  virtual Tensors run(const Tensors& inputs, void* stream) = 0;
};

class Program {
 public:
  virtual ~Program() = default;
  virtual std::unique_ptr<Execution> create_execution(void* stream) const = 0;
};

Buffer allocate(DType dtype, std::vector<std::int64_t> shape, std::int32_t device);
Tensors load_bundle(const std::string& path, std::int32_t device);
std::shared_ptr<Program> load_program(const std::string& architecture,
                                      const std::string& stage,
                                      const std::string& path, Backend backend,
                                      std::int32_t device);

}  // namespace tinfer::native
