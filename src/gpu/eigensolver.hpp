#pragma once

#include <complex>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto eigh(ContextInternal& ctx, std::size_t m, std::size_t nEig, const api::ComplexType<T>* a,
          std::size_t lda, const api::ComplexType<T>* b, std::size_t ldb, T* d,
          api::ComplexType<T>* v, std::size_t ldv) -> void;
}
}  // namespace bipp
