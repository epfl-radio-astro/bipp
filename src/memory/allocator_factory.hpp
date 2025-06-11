#pragma once

#include <memory>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "memory/allocator.hpp"

namespace bipp {

class AllocatorFactory {
public:
  static auto simple_host() -> std::unique_ptr<Allocator>;
  static auto host() -> std::unique_ptr<Allocator>;
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  static auto pinned() -> std::unique_ptr<Allocator>;
  static auto device() -> std::unique_ptr<Allocator>;
#endif
};

}  // namespace bipp
