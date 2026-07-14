#include "tinfer_rust/src/models/base/native.rs.h"
#include "tinfer_rust/src/models/base/cpp/model.hpp"

namespace tinfer::native {

Output Model::generate_batch(const Batch& batch) const {
  return run(batch);
}

}  // namespace tinfer::native
