#pragma once

#include <cstddef>

#include "bipp/bipp.hpp"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto virtual_vis(ContextInternal& ctx, ConstHostView<BippFilter, 1> filter,
                 ConstHostView<T, 2> intervals, ConstDeviceView<T, 1> d,
                 ConstDeviceView<api::ComplexType<T>, 2> v,
                 DeviceView<api::ComplexType<T>, 3> virtVis) -> void;
}  // namespace gpu
}  // namespace bipp
