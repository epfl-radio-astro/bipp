#pragma once

#include <complex>
#include <cstddef>
#include <memory>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "memory/view.hpp"
#include "memory/allocator.hpp"

namespace bipp {
namespace host {

template <typename T>
auto gram_matrix(const std::shared_ptr<Allocator>& alloc, ConstHostView<std::complex<T>, 2> w, ConstHostView<T, 2> xyz,
                 T wl, HostView<std::complex<T>, 2> g) -> void;

}  // namespace host
}  // namespace bipp
