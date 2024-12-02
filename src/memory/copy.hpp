#pragma once

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "memory/view.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

namespace bipp {

template <typename T, std::size_t DIM, typename F>
inline auto copy(const ConstHostView<T, DIM>& source, HostView<F, DIM> dest) {
  static_assert(std::is_convertible_v<T, F>);

  if (source.shape() != dest.shape()) throw InternalError("Host view copy: shapes do not match.");

  if (source.size() == 0) return;

  if constexpr (DIM == 1) {
    if constexpr (std::is_same_v<T, F> && std::is_trivially_copyable_v<T>) {
      std::memcpy(dest.data(), source.data(), source.shape() * sizeof(T));
    } else {
      for (std::size_t i = 0; i < source.size(); ++i) {
        dest[i] = F(source[i]);
      }
    }
  } else {
    for (std::size_t i = 0; i < source.shape()[DIM - 1]; ++i) {
      copy<T, DIM - 1>(source.slice_view(i), dest.slice_view(i));
    }
  }
}

template <typename T, std::size_t DIM, typename F>
inline auto copy(const HostView<T, DIM>& source, HostView<F, DIM> dest) {
  copy(ConstHostView<T, DIM>(source), std::move(dest));
}

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
template <typename T, std::size_t DIM>
inline auto copy(gpu::Queue& queue, const ConstView<T, DIM>& source, View<T, DIM> dest) {
  if (source.shape() != dest.shape())
    throw InternalError("Device to device view copy: shapes do not match.");

  if (source.size() == 0) return;

  if constexpr (DIM == 1) {
    gpu::api::memcpy_async(dest.data(), source.data(), dest.shape() * sizeof(T),
                           gpu::api::flag::MemcpyDefault, queue.stream());

  } else if constexpr (DIM == 2) {
    gpu::api::memcpy_2d_async(dest.data(), dest.strides(1) * sizeof(T), source.data(),
                              source.strides(1) * sizeof(T), dest.shape(0) * sizeof(T),
                              dest.shape(1), gpu::api::flag::MemcpyDefault, queue.stream());
  } else {
    for (std::size_t i = 0; i < source.shape()[DIM - 1]; ++i) {
      copy<T, DIM - 1>(source.slice_view(i), dest.slice_view(i));
    }
  }
}
#endif

}  // namespace bipp
