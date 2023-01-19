#pragma once

#include <complex>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto gram_matrix(ContextInternal& ctx, std::size_t m, std::size_t n, const api::ComplexType<T>* w,
                 std::size_t ldw, const T* xyz, std::size_t ldxyz, T wl, api::ComplexType<T>* g,
                 std::size_t ldg) -> void;
}
}  // namespace bipp
