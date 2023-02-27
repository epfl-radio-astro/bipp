#pragma once

#include <cstddef>

#include "bipp//config.h"
#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto min_element(Queue& q, std::size_t n, const T* x, T* minElement) -> void;

template <typename T>
auto max_element(Queue& q, std::size_t n, const T* x, T* maxElement) -> void;

}  // namespace gpu
}  // namespace bipp
