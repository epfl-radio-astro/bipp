#pragma once

#include <cstddef>

#include "bipp/bipp.hpp"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto virtual_vis(ContextInternal& ctx, const std::size_t nVis, ConstHostView<T, 2> dMasked,
                 ConstDeviceView<api::ComplexType<T>, 2> vAll,
                 DeviceView<api::ComplexType<T>, 2> virtVis) -> void;
}  // namespace gpu
}  // namespace bipp
