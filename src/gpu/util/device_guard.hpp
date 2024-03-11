#pragma once

#include "bipp/config.h"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util//runtime_api.hpp"
namespace bipp {
namespace gpu {
class DeviceGuard {
public:
  explicit DeviceGuard(const int deviceId) : targetDeviceId_(deviceId), originalDeviceId_(0) {
    api::get_device(&originalDeviceId_);
    if (originalDeviceId_ != deviceId) {
      api::set_device(deviceId);
    }
  };

  DeviceGuard() = delete;
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard(DeviceGuard&&) = delete;
  auto operator=(const DeviceGuard&) -> DeviceGuard& = delete;
  auto operator=(DeviceGuard&&) -> DeviceGuard& = delete;

  ~DeviceGuard() {
    if (targetDeviceId_ != originalDeviceId_) {
      try {
        api::set_device(originalDeviceId_);
      } catch (...) {
      }
    }
  }

private:
  int targetDeviceId_ = 0;
  int originalDeviceId_ = 0;
};
}  // namespace gpu
}  // namespace bipp
#endif
