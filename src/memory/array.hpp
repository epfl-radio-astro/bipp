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
  using ValueType = T;
  using BaseType = HostView<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = HostArray<T, DIM - 1>;

  HostArray() : BaseType(){};

  HostArray(const HostArray&) = delete;

  HostArray(HostArray&&) = default;

  auto operator=(const HostArray&) -> HostArray& = delete;

  auto operator=(HostArray&& b) -> HostArray& = default;

  HostArray(std::shared_ptr<Allocator> alloc, const IndexType& shape)
      : BaseType(nullptr, shape, shape_to_stride(shape)) {
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
  static inline auto shape_to_stride(const IndexType& shape) -> IndexType {
    IndexType strides;
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
  using ValueType = T;
  using BaseType = DeviceView<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = DeviceArray<T, DIM - 1>;

  DeviceArray() : BaseType(){};

  DeviceArray(const DeviceArray&) = delete;

  DeviceArray(DeviceArray&&) = default;

  auto operator=(const DeviceArray&) -> DeviceArray& = delete;

  auto operator=(DeviceArray&& b) -> DeviceArray& = default;

  DeviceArray(std::shared_ptr<Allocator> alloc, const IndexType& shape)
      : BaseType(nullptr, shape, shape_to_stride(shape)) {
    if (alloc->type() != MemoryType::Device)
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

  inline auto view() -> DeviceView<T, DIM> { return *this; }

  inline auto view() const -> ConstDeviceView<T, DIM> { return *this; }

private:
  static inline auto shape_to_stride(const IndexType& shape) -> IndexType {
    IndexType strides;
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
