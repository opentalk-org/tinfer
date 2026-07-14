#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace tinfer::native {

class Program;
std::shared_ptr<Program> load_tensorrt(const std::string& path, std::int32_t device);

}  // namespace tinfer::native
