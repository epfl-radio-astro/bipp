#pragma once

#include <cstddef>

#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto scale_matrix(Queue& q, std::size_t m, std::size_t n, const api::ComplexType<T>* A,
                  std::size_t lda, const T* x, api::ComplexType<T>* B, std::size_t ldb) -> void;
}
}  // namespace bipp
