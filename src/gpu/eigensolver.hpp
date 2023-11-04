#pragma once

#include <complex>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto eigh(ContextInternal& ctx, std::size_t nEig, ConstHostView<api::ComplexType<T>, 2> aHost,
          ConstDeviceView<api::ComplexType<T>, 2> a, ConstDeviceView<api::ComplexType<T>, 2> b,
          DeviceView<T, 1> d, DeviceView<api::ComplexType<T>, 2> v) -> void;
}
}  // namespace bipp
