#pragma once

#include <complex>
#include <cstddef>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace host {

template <typename T>
auto eigh(ContextInternal& ctx, std::size_t nEig, const ConstHostView<std::complex<T>, 2>& a,
          const ConstHostView<std::complex<T>, 2>& b, HostView<T, 1> d,
          HostView<std::complex<T>, 2> v) -> void;

}  // namespace host
}  // namespace bipp
