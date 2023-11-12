#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"
#include "memory/copy.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/queue.hpp"
#include "gpu/util/device_pointer.hpp"

namespace bipp {
template <typename T, std::size_t DIM>
class DeviceAccessor {
public:
  using IndexType = typename ConstView<T, DIM>::IndexType;

  DeviceAccessor(gpu::Queue& q, HostView<T, DIM> v) : sourceView_(v) {
    if(v.size()){
      deviceArray_.reset(new DeviceArray<T, DIM>(q.create_device_array<T, DIM>(v.shape())));
      deviceView_ = deviceArray_->view();
      copy(q, v, deviceView_);
    }
  }

  DeviceAccessor(gpu::Queue& q, DeviceView<T, DIM> v) : deviceView_(v) {}

  DeviceAccessor(gpu::Queue& q, T* ptr, IndexType shape, IndexType strides) {
    if(gpu::is_device_ptr(ptr)) {
      *this = DeviceAccessor(q, DeviceView<T, DIM>(ptr, shape, strides));
    } else {
      *this = DeviceAccessor(q, HostView<T, DIM>(ptr, shape, strides));
    }
  }

  inline auto view() const -> const DeviceView<T, DIM>& { return deviceView_; }

  inline auto copy_back(gpu::Queue& q) -> void {
    if (sourceView_) copy(q, deviceView_, sourceView_.value());
  }

private:
  std::shared_ptr<DeviceArray<T, DIM>> deviceArray_;
  DeviceView<T, DIM> deviceView_;
  std::optional<HostView<T, DIM>> sourceView_;
};

template <typename T, std::size_t DIM>
class ConstDeviceAccessor {
public:
  using IndexType = typename ConstView<T, DIM>::IndexType;

  ConstDeviceAccessor(gpu::Queue& q, ConstHostView<T, DIM> v) {
    if(v.size()) {
      deviceArray_.reset(new DeviceArray<T, DIM>(q.create_device_array<T, DIM>(v.shape())));
      deviceView_ = deviceArray_->view();
      copy(q, v, *deviceArray_);
    }
  }

  ConstDeviceAccessor(gpu::Queue& q, ConstDeviceView<T, DIM> v) : deviceView_(v) {}

  ConstDeviceAccessor(gpu::Queue& q, const T* ptr, IndexType shape, IndexType strides) {
    if (gpu::is_device_ptr(ptr)) {
      *this = ConstDeviceAccessor(q, ConstDeviceView<T, DIM>(ptr, shape, strides));
    } else {
      *this = ConstDeviceAccessor(q, ConstHostView<T, DIM>(ptr, shape, strides));
    }
  }

  inline auto view() const -> const ConstDeviceView<T, DIM>& { return deviceView_; }

private:
  std::shared_ptr<DeviceArray<T, DIM>> deviceArray_;
  ConstDeviceView<T, DIM> deviceView_;
};

template <typename T, std::size_t DIM>
class HostAccessor {
public:
  using IndexType = typename ConstView<T, DIM>::IndexType;

  HostAccessor(gpu::Queue& q, DeviceView<T, DIM> v) : sourceView_(v) {
    if (v.size()) {
      hostArray_.reset(new HostArray<T, DIM>(q.create_host_array<T, DIM>(v.shape())));
      hostView_ = hostArray_->view();
      copy(q, v, hostView_);
    }
  }

  HostAccessor(gpu::Queue& q, HostView<T, DIM> v) : hostView_(v) {}

  HostAccessor(gpu::Queue& q, T* ptr, IndexType shape, IndexType strides) {
    if (gpu::is_device_ptr(ptr)) {
      *this = HostAccessor(q, ConstDeviceView<T, DIM>(ptr, shape, strides));
    } else {
      *this = HostAccessor(q, HostView<T, DIM>(ptr, shape, strides));
    }
  }

  inline auto view() const -> const HostView<T, DIM>& { return hostView_; }

  inline auto copy_back(gpu::Queue& q) -> void {
    if (sourceView_) copy(q, hostView_, sourceView_.value());
  }

private:
  std::shared_ptr<HostArray<T, DIM>> hostArray_;
  HostView<T, DIM> hostView_;
  std::optional<DeviceView<T, DIM>> sourceView_;
};

template <typename T, std::size_t DIM>
class ConstHostAccessor {
public:
  using IndexType = typename ConstView<T, DIM>::IndexType;

  ConstHostAccessor(gpu::Queue& q, ConstDeviceView<T, DIM> v) {
    if(v.size()) {
      hostArray_.reset(new HostArray<T, DIM>(q.create_host_array<T, DIM>(v.shape())));
      hostView_ = hostArray_->view();
      copy(q, v, *hostArray_);
    }
  }

  ConstHostAccessor(gpu::Queue& q, ConstHostView<T, DIM> v) : hostView_(v) {}

  ConstHostAccessor(gpu::Queue& q, const T* ptr, IndexType shape, IndexType strides) {
    if (gpu::is_device_ptr(ptr)) {
      *this = ConstHostAccessor(q, ConstDeviceView<T, DIM>(ptr, shape, strides));
    } else {
      *this = ConstHostAccessor(q, ConstHostView<T, DIM>(ptr, shape, strides));
    }
  }

  inline auto view() const -> const ConstHostView<T, DIM>& { return hostView_; }

private:
  std::shared_ptr<HostArray<T, DIM>> hostArray_;
  ConstHostView<T, DIM> hostView_;
};

}  // namespace bipp
   //
#endif
