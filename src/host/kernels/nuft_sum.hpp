#pragma once

#include <complex>

#include "bipp/config.h"

namespace bipp {
namespace host {

template <typename T>
auto nuft_sum(T alpha, std::size_t nIn, const std::complex<T>* __restrict__ input, const T* u,
              const T* v, const T* w, std::size_t nOut, const T* x, const T* y, const T* z, T* out)
    -> void;

}  // namespace host
}  // namespace bipp
