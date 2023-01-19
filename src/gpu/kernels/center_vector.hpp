#pragma once

#include <cstddef>

#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto center_vector(Queue& q, std::size_t n, T* vec) -> void;
}  // namespace gpu
}  // namespace bipp
