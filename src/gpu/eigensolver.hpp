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
auto eigh(ContextInternal& ctx, T wl, ConstDeviceView<api::ComplexType<T>, 2> s,
          ConstDeviceView<api::ComplexType<T>, 2> w, ConstDeviceView<T, 2> xyz, DeviceView<T, 1> d,
          DeviceView<api::ComplexType<T>, 2> vUnbeam) -> std::pair<std::size_t, std::size_t>;

template <typename T>
auto eigh(ContextInternal& ctx, T wl, ConstDeviceView<api::ComplexType<T>, 2> s,
          ConstDeviceView<api::ComplexType<T>, 2> w, ConstDeviceView<T, 2> xyz, DeviceView<T, 1> d)
    -> std::pair<std::size_t, std::size_t> {
  return eigh<T>(ctx, wl, s, w, xyz, d, DeviceView<api::ComplexType<T>, 2>());
}

}  // namespace gpu
}  // namespace bipp
