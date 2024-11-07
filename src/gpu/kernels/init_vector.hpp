#pragma once

#include <cstddef>

#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto init_vector(const api::DevicePropType& prop, const api::StreamType& stream, std::size_t n,
                 T value, T* a) -> void;
}
}  // namespace bipp
