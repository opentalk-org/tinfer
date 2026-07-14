#pragma once

#include "tinfer_rust/src/models/base/cpp/model.hpp"

namespace tinfer::native {

struct Batch;
struct Output;

class StubModel final : public Model {
 private:
  Output run(const Batch& batch) const override;
};

}  // namespace tinfer::native
