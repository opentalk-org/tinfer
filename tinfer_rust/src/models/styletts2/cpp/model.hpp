#pragma once

#include "tinfer_rust/src/models/base/cpp/engine.hpp"
#include "tinfer_rust/src/models/base/cpp/model.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/session.hpp"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace tinfer::native {

class StyleTts2Model final : public Model {
 public:
  StyleTts2Model(const std::string& root, const std::string& architecture,
                 Backend backend, std::int32_t device);
  ~StyleTts2Model() override;

 private:
  Output run(const Batch& batch) const override;
  Output close(const Batch& batch) const;
  Output start_cpu(const Batch& batch) const;
  Output continue_cpu(const Batch& batch) const;
  Output start_cuda(const Batch& batch) const;
  Output continue_cuda(const Batch& batch) const;

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
  mutable std::mutex sessions_mutex_;
  mutable std::unordered_map<std::int64_t, StyleTts2Session> sessions_;
  mutable std::unordered_map<std::int64_t, StyleTts2Tail> tails_;
};

}  // namespace tinfer::native
