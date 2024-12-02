#pragma once

#include <complex>
#include <cstddef>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace host {

template <typename T>
auto gram_matrix(ContextInternal& ctx, ConstHostView<std::complex<T>, 2> w, ConstHostView<T, 2> xyz,
                 T wl, HostView<std::complex<T>, 2> g) -> void;

}  // namespace host
}  // namespace bipp
