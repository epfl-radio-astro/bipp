#pragma once

#include <complex>
#include <cstddef>

#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto gram(Queue& q, std::size_t n, const T* x, const T* y, const T* z, T wl, api::ComplexType<T>* g,
          std::size_t ldg) -> void;
}
}  // namespace bipp
