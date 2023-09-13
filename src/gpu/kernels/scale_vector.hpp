#pragma once

#include <cstddef>

#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

// Compute b = alpha * a;
template <typename T>
auto scale_vector(const api::DevicePropType& prop, const api::StreamType& stream, std::size_t n,
                  const T* a, T alpha, T* b) -> void;

// Compute a = alpha * a;
template <typename T>
auto scale_vector(const api::DevicePropType& prop, const api::StreamType& stream, std::size_t n,
                  T alpha, T* a) -> void;
}
}  // namespace bipp
