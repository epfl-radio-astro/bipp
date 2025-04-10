#pragma once

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "bipp/config.h"
#include "bipp/context.hpp"
#include "bipp/enums.h"
#include "bipp/exceptions.hpp"
#include "logger.hpp"
#include "memory/allocator.hpp"
#include "memory/allocator_factory.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

namespace bipp {

class ContextInternal {
public:
  explicit ContextInternal(BippProcessingUnit pu);

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  auto gpu_queue() -> gpu::Queue& { return queue_.value(); }

  auto gpu_queue() const -> const gpu::Queue& { return queue_.value(); }
#endif

  auto processing_unit() const -> BippProcessingUnit { return pu_; }

  auto device_id() const -> int { return deviceId_; }

  auto host_alloc() const -> const std::shared_ptr<Allocator>& { return hostAlloc_; }

private:
  BippProcessingUnit pu_;
  std::shared_ptr<Allocator> hostAlloc_;
  int deviceId_ = 0;

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  std::optional<gpu::Queue> queue_;
#endif
};

struct InternalContextAccessor {
  static auto get(const Context& ctx) -> const std::shared_ptr<ContextInternal>& {
    return ctx.ctx_;
  }
};

}  // namespace bipp
