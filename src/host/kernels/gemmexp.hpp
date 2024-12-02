#pragma once

#include <complex>
#include <cstddef>

#include "bipp/config.h"
#include "context_internal.hpp"

namespace bipp {
namespace host {

template <typename T>
auto gemmexp(const std::size_t M, const std::size_t N, const std::size_t K, const T alpha,
             const T* __restrict__ A, const std::size_t lda, const T* __restrict__ B,
             const std::size_t ldb, std::complex<T>* __restrict__ C, const std::size_t ldc) -> void;

template <typename T>
auto gemmexp(std::size_t nEig, std::size_t nPixel, std::size_t nAntenna, T alpha,
             const std::complex<T>* __restrict__ vUnbeam, std::size_t ldvw,
             const T* __restrict__ xyz, std::size_t ldxyz, const T* __restrict__ pixelX,
             const T* __restrict__ pixelY, const T* __restrict__ pixelZ, T* __restrict__ out,
             std::size_t ldout) -> void;
}  // namespace host
}  // namespace bipp
