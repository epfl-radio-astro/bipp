#pragma once

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/array.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto gram_matrix(ContextInternal& ctx, ConstDeviceView<api::ComplexType<T>, 2> w,
                 ConstDeviceView<T, 2> xyz, T wl, DeviceView<api::ComplexType<T>, 2> g) -> void;
}
}  // namespace bipp
