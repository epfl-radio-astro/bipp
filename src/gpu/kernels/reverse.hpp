#pragma once

#include <cstddef>

#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto reverse_1d(Queue& q, std::size_t n, T* x) -> void;

template <typename T>
auto reverse_2d(Queue& q, std::size_t m, std::size_t n, T* x, std::size_t ld) -> void;
}  // namespace gpu
}  // namespace bipp
