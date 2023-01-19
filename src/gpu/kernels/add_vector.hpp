#pragma once

#include <cstddef>

#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto add_vector_real_to_complex(Queue& q, std::size_t n, const api::ComplexType<T>* a, T* b)
    -> void;
}
}  // namespace bipp
