#pragma once

#include <cstddef>

#include "bipp/enums.h"
#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto apply_filter(Queue& q, BippFilter filter, std::size_t n, const T* in, T* out) -> void;
}
}  // namespace bipp
