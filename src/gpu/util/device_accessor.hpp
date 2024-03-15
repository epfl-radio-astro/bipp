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

/*
 * Allows access to multi-dimensional memory from GPU. Copies if input is located on host, otherwise
 * only presents to the initial input.
 */
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

  DeviceAccessor(gpu::Queue& q, T* ptr, IndexType shape, IndexType strides)
      : DeviceAccessor(q, View<T, DIM>(ptr, shape, strides)) {}

  DeviceAccessor(gpu::Queue& q, View<T, DIM> v) {
    if (gpu::is_device_ptr(v.data())) {
      *this = DeviceAccessor(q, DeviceView<T, DIM>(v));
    } else {
      *this = DeviceAccessor(q, HostView<T, DIM>(v));
    }
  }

  // Access from device
  inline auto view() const -> const DeviceView<T, DIM>& { return deviceView_; }

  // Copy back to host if copy to device was performed at construction
  inline auto copy_back(gpu::Queue& q) -> void {
    if (sourceView_) copy(q, deviceView_, sourceView_.value());
  }

private:
  std::shared_ptr<DeviceArray<T, DIM>> deviceArray_;
  DeviceView<T, DIM> deviceView_;
  std::optional<HostView<T, DIM>> sourceView_;
};

/*
 * Allows access to multi-dimensional memory from GPU. Copies if input is located on host, otherwise
 * only presents to the initial input.
 */
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

  ConstDeviceAccessor(gpu::Queue& q, const T* ptr, IndexType shape, IndexType strides)
      : ConstDeviceAccessor(q, ConstView<T, DIM>(ptr, shape, strides)) {}

  ConstDeviceAccessor(gpu::Queue& q, ConstView<T, DIM> v) {
    if (gpu::is_device_ptr(v.data())) {
      *this = ConstDeviceAccessor(q, ConstDeviceView<T, DIM>(v));
    } else {
      *this = ConstDeviceAccessor(q, ConstHostView<T, DIM>(v));
    }
  }

  inline auto view() const -> const ConstDeviceView<T, DIM>& { return deviceView_; }

private:
  std::shared_ptr<DeviceArray<T, DIM>> deviceArray_;
  ConstDeviceView<T, DIM> deviceView_;
};

/*
 * Allows access to multi-dimensional memory from Host. Copies if input is located on device,
 * otherwise only presents to the initial input.
 *
 * Note: May require queue synchronization before available.
 */
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

  HostAccessor(gpu::Queue& q, T* ptr, IndexType shape, IndexType strides)
      : HostAccessor(q, View<T, DIM>(ptr, shape, strides)) {}

  HostAccessor(gpu::Queue& q, View<T, DIM> v) {
    if (gpu::is_device_ptr(v.data())) {
      *this = HostAccessor(q, DeviceView<T, DIM>(v));
    } else {
      *this = HostAccessor(q, HostView<T, DIM>(v));
    }
  }

  inline auto view() const -> const HostView<T, DIM>& { return hostView_; }

  // Copy back to device if copy to host was performed at construction
  inline auto copy_back(gpu::Queue& q) -> void {
    if (sourceView_) copy(q, hostView_, sourceView_.value());
  }

private:
  std::shared_ptr<HostArray<T, DIM>> hostArray_;
  HostView<T, DIM> hostView_;
  std::optional<DeviceView<T, DIM>> sourceView_;
};

/*
 * Allows access to multi-dimensional memory from Host. Copies if input is located on device,
 * otherwise only presents to the initial input.
 *
 * Note: May require queue synchronization before available.
 */
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

  ConstHostAccessor(gpu::Queue& q, const T* ptr, IndexType shape, IndexType strides)
      : ConstHostAccessor(q, ConstView<T, DIM>(ptr, shape, strides)) {}

  ConstHostAccessor(gpu::Queue& q, ConstView<T, DIM> v) {
    if (gpu::is_device_ptr(v.data())) {
      *this = ConstHostAccessor(q, ConstDeviceView<T, DIM>(v));
    } else {
      *this = ConstHostAccessor(q, ConstHostView<T, DIM>(v));
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
