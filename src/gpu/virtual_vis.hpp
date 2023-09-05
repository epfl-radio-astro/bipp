#pragma once

#include <cstddef>

#include "bipp/bipp.hpp"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto virtual_vis(ContextInternal& ctx, std::size_t nFilter, const BippFilter* filterHost,
                 std::size_t nIntervals, const T* intervalsHost, std::size_t ldIntervals,
                 std::size_t nEig, const T* D, std::size_t nAntenna, const api::ComplexType<T>* V,
                 std::size_t ldv, std::size_t nBeam, const api::ComplexType<T>* W, std::size_t ldw,
                 api::ComplexType<T>* virtVis, std::size_t ldVirtVis1, std::size_t ldVirtVis2,
                 std::size_t ldVirtVis3, const std::size_t nz_vis) -> void;
}  // namespace gpu
}  // namespace bipp
