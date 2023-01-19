#pragma once

#include <complex>
#include <cstddef>

#include "bipp/bipp.hpp"
#include "bipp/config.h"
#include "context_internal.hpp"

namespace bipp {
namespace host {

template <typename T>
auto virtual_vis(ContextInternal& ctx, std::size_t nFilter, const BippFilter* filter,
                 std::size_t nIntervals, const T* intervals, std::size_t ldIntervals,
                 std::size_t nEig, const T* D, std::size_t nAntenna, const std::complex<T>* V,
                 std::size_t ldv, std::size_t nBeam, const std::complex<T>* W, std::size_t ldw,
                 std::complex<T>* virtVis, std::size_t ldVirtVis1, std::size_t ldVirtVis2,
                 std::size_t ldVirtVis3) -> void;

}  // namespace host
}  // namespace bipp
