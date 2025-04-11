#pragma once

#include <complex>
#include <cstddef>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace host {

template <typename T>
auto eigh(T wl, ConstHostView<std::complex<T>, 2> s, ConstHostView<std::complex<T>, 2> w,
          ConstHostView<T, 2> xyz, HostView<T, 1> d, HostView<std::complex<T>, 2> vUnbeam)
    -> std::pair<std::size_t, T>;

}  // namespace host
}  // namespace bipp
