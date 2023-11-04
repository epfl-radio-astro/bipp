#pragma once

#include <cstddef>
#include <cstring>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "memory/view.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

namespace bipp {

template <typename T, std::size_t DIM>
inline auto copy(const ConstHostView<T, DIM>& source, HostView<T, DIM> dest) {
  if (source.shape() != dest.shape()) throw InternalError("Host view copy: shapes do not match.");

  if (source.size() == 0) return;

  if constexpr (DIM == 1) {
    std::memcpy(dest.data(), source.data(), source.shape()[0] * sizeof(T));
  } else {
    for (std::size_t i = 0; i < source.shape()[DIM - 1]; ++i) {
      copy<T, DIM - 1>(source.slice_view(i), dest.slice_view(i));
    }
  }
}

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
template <typename T, std::size_t DIM>
inline auto copy(gpu::Queue& queue, const ViewBase<T, DIM>& source, DeviceView<T, DIM> dest) {
  if (source.shape() != dest.shape())
    throw InternalError("Device to device view copy: shapes do not match.");

  if (source.size() == 0) return;

  if constexpr (DIM == 1) {
    gpu::api::memcpy_async(dest.data(), source.data(), dest.shape()[0] * sizeof(T),
                           gpu::api::flag::MemcpyDefault, queue.stream());

  } else if constexpr (DIM == 2) {
    gpu::api::memcpy_2d_async(dest.data(), dest.strides()[1] * sizeof(T), source.data(),
                              source.strides()[1] * sizeof(T), dest.shape()[0] * sizeof(T),
                              dest.shape()[1], gpu::api::flag::MemcpyDefault, queue.stream());
  } else {
    for (std::size_t i = 0; i < source.shape()[DIM - 1]; ++i) {
      copy<T, DIM - 1>(source.slice_view(i), dest.slice_view(i));
    }
  }
}

template <typename T, std::size_t DIM>
inline auto copy(gpu::Queue& queue, const ViewBase<T, DIM>& source, HostView<T, DIM> dest) {
  if (source.shape() != dest.shape())
    throw InternalError("Device to device view copy: shapes do not match.");

  if (source.size() == 0) return;

  if constexpr (DIM == 1) {
    gpu::api::memcpy_async(dest.data(), source.data(), dest.shape()[0] * sizeof(T),
                           gpu::api::flag::MemcpyDefault, queue.stream());

  } else if constexpr (DIM == 2) {
    gpu::api::memcpy_2d_async(dest.data(), dest.strides()[1] * sizeof(T), source.data(),
                              source.strides()[1] * sizeof(T), dest.shape()[0] * sizeof(T),
                              dest.shape()[1], gpu::api::flag::MemcpyDefault, queue.stream());
  } else {
    for (std::size_t i = 0; i < source.shape()[DIM - 1]; ++i) {
      copy<T, DIM - 1>(source.slice_view(i), dest.slice_view(i));
    }
  }
}

#endif

}  // namespace bipp
