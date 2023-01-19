#pragma once

#include <tuple>

#include "bipp/config.h"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto is_device_ptr(const T* ptr) -> bool {
  api::PointerAttributes attr;
  auto status = api::pointer_get_attributes(&attr, static_cast<const void*>(ptr));

  if (status != api::status::Success) {
    // throw error if unexpected error
    if (status != api::status::ErrorInvalidValue) api::check_status(status);
    // clear error from cache and return otherwise
    std::ignore = api::get_last_error();
    return false;
  }

  // get memory type - cuda 10 changed attribute name
#if defined(BIPP_CUDA) && (CUDART_VERSION >= 10000)
  auto memoryType = attr.type;
#else
  auto memoryType = attr.memoryType;
#endif

  if (memoryType == api::flag::MemoryTypeDevice) {
    return true;
  }
  return false;
}
}  // namespace gpu
}  // namespace bipp
