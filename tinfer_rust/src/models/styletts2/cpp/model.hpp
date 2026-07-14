#pragma once

#include "tinfer_rust/src/models/base/cpp/engine.hpp"
#include "tinfer_rust/src/models/base/cpp/model.hpp"

#include <memory>

namespace tinfer::native {

class StyleTts2Model final : public Model {
 public:
  StyleTts2Model(const std::string& root, const std::string& architecture,
                 Backend backend, std::int32_t device);
  ~StyleTts2Model() override;

 private:
  Output run(const Batch& batch) const override;
  Output run_cpu(const Batch& batch) const;
  Output run_cuda(const Batch& batch) const;

  Backend backend_;
  std::int32_t device_;
  void* stream_ = nullptr;
  Tensors weights_a_;
  Tensors weights_b_;
  Tensors weights_c_;
  Tensors glue_weights_;
  std::shared_ptr<Program> program_a_;
  std::shared_ptr<Program> program_b_;
  std::shared_ptr<Program> program_c_;
  std::unique_ptr<Execution> execution_a_;
  std::unique_ptr<Execution> execution_b_;
  std::unique_ptr<Execution> execution_c_;
};

}  // namespace tinfer::native
