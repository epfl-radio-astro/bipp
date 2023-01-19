#pragma once

#include <cstddef>

#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto gemmexp(Queue& q, std::size_t nEig, std::size_t nPixel, std::size_t nAntenna, T alpha,
             const api::ComplexType<T>* vUnbeam, std::size_t ldv, const T* xyz, std::size_t ldxyz,
             const T* pixelX, const T* pixelY, const T* pixelZ, T* out, std::size_t ldout) -> void;
}  // namespace gpu
}  // namespace bipp
