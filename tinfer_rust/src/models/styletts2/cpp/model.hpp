#pragma once

#include "tinfer_rust/src/models/base/cpp/engine.hpp"
#include "tinfer_rust/src/models/base/cpp/model.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/session.hpp"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace tinfer::native {

struct PinnedBuffer {
  std::shared_ptr<void> memory;
  std::size_t bytes;
};

class StyleTts2Model final : public Model {
 public:
  StyleTts2Model(const std::string& root, const std::string& architecture,
                 Backend backend, std::int32_t device,
                 std::int32_t max_batch);
  ~StyleTts2Model() override;

 private:
  Output run(const Batch& batch) const override;
  Output close(const Batch& batch) const;
  Output start(const Batch& batch) const;
  Output continue_generation(const Batch& batch) const;
  Output start_cuda(const Batch& batch) const;
  Output continue_cuda(const Batch& batch) const;
  Tensors upload(const Batch& batch, const Execution& execution) const;
  Buffer upload_floats(const std::string& name,
                       const std::vector<float>& values,
                       std::vector<std::int64_t> shape,
                       DType device_dtype) const;
  Buffer workspace(const std::string& name, DType dtype,
                   std::vector<std::int64_t> shape,
                   std::vector<std::int64_t> capacity) const;
  void* pinned(const std::string& name, std::size_t bytes,
               std::size_t capacity) const;
  Buffer pad_columns(const std::string& name, const Buffer& source,
                     std::int64_t columns, int fill) const;
  std::vector<float> download_floats(const Buffer& buffer) const;
  Buffer source_noise(const std::vector<std::uint64_t>& seeds) const;

  Backend backend_;
  std::int32_t device_;
  std::int32_t max_batch_;
  void* stream_ = nullptr;
  Tensors weights_a_;
  Tensors weights_bc_;
  std::shared_ptr<Program> program_a_;
  std::shared_ptr<Program> program_bc_;
  std::unique_ptr<Execution> execution_a_;
  std::unique_ptr<Execution> execution_bc_;
  mutable Tensors workspace_;
  mutable std::map<std::string, PinnedBuffer, std::less<>> upload_staging_;
  mutable void* download_staging_ = nullptr;
  mutable std::size_t download_capacity_ = 0;
  mutable std::vector<float> source_noise_cpu_;
  mutable std::mutex sessions_mutex_;
  mutable std::unordered_map<std::int64_t, StyleTts2Session> sessions_;
  mutable std::unordered_map<std::int64_t, StyleTts2Tail> tails_;
};

}  // namespace tinfer::native
