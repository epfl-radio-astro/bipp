#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "memory/allocator.hpp"
#include "memory/view.hpp"

namespace bipp {
template <typename T, std::size_t DIM>
class HostArray : public HostView<T, DIM> {
public:
  using value_type = T;
  using base_type = HostView<T, DIM>;
  using index_type = typename base_type::index_type;
  using slice_type = HostArray<T, DIM - 1>;

  HostArray() : base_type(){};

  HostArray(const HostArray&) = delete;

  HostArray(HostArray&&) = default;

  auto operator=(const HostArray&) -> HostArray& = delete;

  auto operator=(HostArray&& b) -> HostArray& = default;

  HostArray(std::shared_ptr<Allocator> alloc, const index_type& shape)
      : base_type(nullptr, shape, shape_to_stride(shape)) {
    if (alloc->type() != MemoryType::Host)
      throw InternalError("View: Memory type and allocator type mismatch.");
    if (this->totalSize_) {
      auto ptr = alloc->allocate(this->totalSize_ * sizeof(T));
      data_ = std::shared_ptr<void>(ptr, [alloc = std::move(alloc)](void* p) {
        if (p) alloc->deallocate(p);
      });
      this->constPtr_ = static_cast<T*>(ptr);
    }
  };

  inline auto data_handler() const noexcept -> const std::shared_ptr<void>& { return data_; }

  inline auto view() -> HostView<T, DIM> { return *this; }

  inline auto view() const -> ConstHostView<T, DIM> { return *this; }

private:
  static inline auto shape_to_stride(const index_type& shape) -> index_type {
    index_type strides;
    strides[0] = 1;
    for (std::size_t i = 1; i < DIM; ++i) {
      strides[i] = shape[i - 1] * strides[i - 1];
    }
    strides[0] = 1;
    return strides;
  }
  std::shared_ptr<void> data_;
};

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
template <typename T, std::size_t DIM>
class DeviceArray : public DeviceView<T, DIM> {
public:
  using value_type = T;
  using base_type = DeviceView<T, DIM>;
  using index_type = typename base_type::index_type;
  using slice_type = DeviceArray<T, DIM - 1>;

  DeviceArray() : base_type(){};

  DeviceArray(std::shared_ptr<Allocator> alloc, const index_type& shape)
      : base_type(shape, shape_to_stride(shape)) {
    if (alloc->type() != MemoryType::Device)
      throw InternalError("View: Memory type and allocator type mismatch.");
    if (this->totalSize_) {
      auto ptr = alloc->allocate(this->totalSize_ * sizeof(T));
      data_ = std::shared_ptr<void>(ptr, [alloc = std::move(alloc)](void* p) {
        if (p) alloc->deallocate(p);
      });
    }
  };

  inline auto data_handler() const noexcept -> const std::shared_ptr<void>& { return data_; }

  inline auto view() -> DeviceView<T, DIM> { return *this; }

  inline auto view() const -> ConstDeviceView<T, DIM> { return *this; }

private:
  static inline auto shape_to_stride(const index_type& shape) -> index_type {
    index_type strides;
    strides[0] = 1;
    for (std::size_t i = 1; i < DIM; ++i) {
      strides[i] = shape[i - 1] * strides[i - 1];
    }
    strides[0] = 1;
    return strides;
  }
  std::shared_ptr<void> data_;
};

#endif

}  // namespace bipp
