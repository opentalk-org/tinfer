#pragma once

#include <memory>
#include "rust/cxx.h"

namespace tinfer::native {

struct Batch;
struct Output;

class Model {
 public:
  virtual ~Model() = default;
  Output generate_batch(const Batch& batch) const;

 private:
  virtual Output run(const Batch& batch) const = 0;
};

std::unique_ptr<Model> load_stub();
std::unique_ptr<Model> load_styletts2(rust::Str root, rust::Str architecture,
                                      std::uint8_t backend, std::int32_t device);
rust::Vec<std::int32_t> cpu_duration_prefix(rust::Slice<const float> durations,
                                            rust::Slice<const std::int32_t> lengths,
                                            rust::Slice<const float> speeds,
                                            std::int32_t batch, std::int32_t tokens);

}  // namespace tinfer::native
