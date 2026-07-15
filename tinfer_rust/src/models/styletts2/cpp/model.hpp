#pragma once

#include "tinfer_rust/src/models/base/cpp/engine.hpp"
#include "tinfer_rust/src/models/base/cpp/model.hpp"
#include "tinfer_rust/src/models/styletts2/cpp/session.hpp"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace tinfer::native {

struct DeviceProsodyWindow {
  Buffer f0;
  Buffer noise;
  Buffer phases;
};

class StyleTts2Model final : public Model {
 public:
  StyleTts2Model(const std::string& root, const std::string& architecture,
                 Backend backend, std::int32_t device);
  ~StyleTts2Model() override;

 private:
  Output run(const Batch& batch) const override;
  Output close(const Batch& batch) const;
  Output start(const Batch& batch) const;
  Output continue_generation(const Batch& batch) const;
  Tensors upload(const Batch& batch, const Execution& execution) const;
  Buffer upload_floats(const std::vector<float>& values,
                       std::vector<std::int64_t> shape) const;
  Buffer pad_columns(const Buffer& source, std::int64_t columns, int fill) const;
  std::vector<float> download_floats(const Buffer& buffer) const;
  Buffer harmonic(const std::vector<float>& f0,
                  const std::vector<std::uint64_t>& seeds,
                  const std::vector<float>& phases) const;
  Buffer harmonic(const DeviceProsodyWindow& window,
                  const std::vector<std::uint64_t>& seeds) const;
  void initialize_device_prosody(
      const std::vector<StyleTts2Session*>& sessions) const;
  void append_device_prosody(StyleTts2Session& session,
                             const Buffer& f0, const Buffer& noise,
                             std::int32_t item, std::int32_t offset,
                             std::int32_t count) const;
  DeviceProsodyWindow device_prosody_window(
      const std::vector<StyleTts2Session*>& sessions,
      const std::vector<std::int32_t>& starts) const;
  void finish_device_prosody(StyleTts2Session& session) const;
  void ensure_prosody(const std::vector<StyleTts2Session*>& sessions,
                      const std::vector<std::int32_t>& targets) const;

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
