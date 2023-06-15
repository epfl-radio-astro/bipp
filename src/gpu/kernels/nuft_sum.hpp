#pragma once

#include <cstddef>

#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
// Direct computation of the type 3 NUFT sum, where only the real valued output is considered
template <typename T>
auto nuft_sum(const api::DevicePropType& prop, const api::StreamType& stream, T alpha,
              std::size_t nIn, const api::ComplexType<T>* __restrict__ input, const T* u,
              const T* v, const T* w, std::size_t nOut, const T* x, const T* y, const T* z, T* out)
    -> void;
}  // namespace gpu
}  // namespace bipp
