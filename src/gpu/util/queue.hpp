#pragma once

#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <tuple>

#include "bipp/config.h"
#include "gpu/util/blas_api.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/allocator.hpp"
#include "memory/allocator_factory.hpp"
#include "memory/array.hpp"

namespace bipp {
namespace gpu {

// A workqueue on GPU for ordered execution and memory allocations on a single GPU stream
class Queue {
public:
  Queue()
      : hostAllocator_(AllocatorFactory::host()),
        pinnedAllocator_(AllocatorFactory::pinned()),
        deviceAllocator_(AllocatorFactory::device()),
        deviceId_(0) {
    api::get_device(&deviceId_);
    api::get_device_properties(&properties_, deviceId_);

    // create stream
    api::StreamType stream;
    api::stream_create_with_flags(&stream, api::flag::StreamNonBlocking);
    stream_ = std::unique_ptr<api::StreamType, std::function<void(api::StreamType*)>>(
        new api::StreamType(stream), [](api::StreamType* ptr) {
          std::ignore = api::stream_destroy(*ptr);
          delete ptr;
        });

    // create event
    api::EventType event;
    api::event_create_with_flags(&event, api::flag::EventDisableTiming);
    event_ = std::unique_ptr<api::EventType, std::function<void(api::EventType*)>>(
        new api::EventType(event), [](api::EventType* ptr) {
          std::ignore = api::event_destroy(*ptr);
          delete ptr;
        });

    // create blas handle
    api::blas::HandleType blasHandle;
    api::blas::create(&blasHandle);
    blasHandle_ =
        std::unique_ptr<api::blas::HandleType, std::function<void(api::blas::HandleType*)>>(
            new api::blas::HandleType(blasHandle), [](api::blas::HandleType* ptr) {
              std::ignore = api::blas::destroy(*ptr);
              delete ptr;
            });
    api::blas::set_stream(*blasHandle_, *stream_);

  }

  Queue(const Queue&) = delete;

  Queue(Queue&& g) = default;

  auto operator=(const Queue&) -> Queue& = delete;

  auto operator=(Queue&&) -> Queue& = default;

  auto stream() const -> const api::StreamType& { return *stream_; }

  auto blas_handle() const -> const api::blas::HandleType& { return *blasHandle_; }

  auto device_id() const -> int { return deviceId_; }

  auto device_prop() const -> const api::DevicePropType& { return properties_; }

  template <typename T, std::size_t N>
  auto create_host_array(typename HostArray<T, N>::IndexType shape) -> HostArray<T, N> {
    HostArray<T, N> a(hostAllocator_, shape);
    allocatedData_.emplace_back(a.data_handler());
    return a;
  }

  template <typename T, std::size_t N>
  auto create_pinned_array(typename HostArray<T, N>::IndexType shape) -> HostArray<T, N> {
    HostArray<T, N> a(pinnedAllocator_, shape);
    allocatedData_.emplace_back(a.data_handler());
    return a;
  }

  template <typename T, std::size_t N>
  auto create_device_array(typename DeviceArray<T, N>::IndexType shape) -> DeviceArray<T, N> {
    DeviceArray<T, N> a(deviceAllocator_, shape);
    allocatedData_.emplace_back(a.data_handler());
    return a;
  }

  // Enter "wait for stream" into queue
  inline auto sync_with_stream(const api::StreamType& s) -> void {
    api::event_record(*event_, s);
    api::stream_wait_event(*stream_, *event_, 0);
  }

  // Input stream waits for queue
  inline auto signal_stream(const api::StreamType& s) -> void {
    api::event_record(*event_, *stream_);
    api::stream_wait_event(s, *event_, 0);
  }

  // Sync queue with host and deallocate unused memory
  inline auto sync() -> void {
    api::stream_synchronize(*stream_);
    allocatedData_.remove_if([](const auto& b) { return b.use_count() <= 1; });
  }

  ~Queue() {
    try {
      sync();
    } catch (...) {
    }
  }

  // A guard, which calls sync on queue on destruction
  struct SyncGuard {
    explicit SyncGuard(Queue& q) : q_(&q) {}
    ~SyncGuard() {
      if (q_) q_->sync();
    }

    SyncGuard(const SyncGuard&) = delete;

    SyncGuard(SyncGuard&& g) { *this = std::move(g); }

    auto operator=(const SyncGuard&) -> SyncGuard& = delete;

    auto operator=(SyncGuard&& g) -> SyncGuard& {
      q_ = g.q_;
      g.q_ = nullptr;
      return *this;
    }
    Queue* q_;
  };

  inline auto sync_guard() -> SyncGuard { return SyncGuard(*this); }

  // return size of allocated (host, pinned, device) memory
  inline auto allocated_memory() {
    return std::make_tuple(hostAllocator_->size(), pinnedAllocator_->size(),
                           deviceAllocator_->size());
  }

  auto host_alloc() -> std::shared_ptr<Allocator>& { return hostAllocator_; }

  auto pinned_alloc() -> std::shared_ptr<Allocator>& { return pinnedAllocator_; }

  auto device_alloc() -> std::shared_ptr<Allocator>& { return deviceAllocator_; }

private:
  std::shared_ptr<Allocator> hostAllocator_;
  std::shared_ptr<Allocator> pinnedAllocator_;
  std::shared_ptr<Allocator> deviceAllocator_;
  std::list<std::shared_ptr<void>> allocatedData_;

  int deviceId_;
  api::DevicePropType properties_;
  std::unique_ptr<api::StreamType, std::function<void(api::StreamType*)>> stream_;
  std::unique_ptr<api::EventType, std::function<void(api::EventType*)>> event_;
  std::unique_ptr<api::blas::HandleType, std::function<void(api::blas::HandleType*)>> blasHandle_;

  auto remove_unused_data() -> void {}
};
}  // namespace gpu

}  // namespace bipp
