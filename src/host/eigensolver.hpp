#pragma once

#include <complex>
#include <cstddef>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"

namespace bipp {
namespace host {

template <typename T>
auto eigh(ContextInternal& ctx, std::size_t m, std::size_t nEig, const std::complex<T>* a,
          std::size_t lda, const std::complex<T>* b, std::size_t ldb, const char range,
          T* d, std::complex<T>* v, std::size_t ldv) -> void;

}  // namespace host
}  // namespace bipp
