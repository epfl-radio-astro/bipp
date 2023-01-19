#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <tuple>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "bipp/context.hpp"
#include "bipp/exceptions.hpp"
#include "memory/allocator.hpp"
#include "memory/allocator_factory.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/blas_api.hpp"
#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

#if !defined(BIPP_MAGMA) && defined(BIPP_CUDA)
#include <cusolverDn.h>
#endif
namespace bipp {

class ContextInternal {
public:
  explicit ContextInternal(BippProcessingUnit pu) : hostAlloc_(AllocatorFactory::host()) {
    if (pu == BIPP_PU_AUTO) {
      // select GPU if available
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      try {
        int deviceId = 0;
        gpu::api::get_device(&deviceId);
        pu_ = BIPP_PU_GPU;
      } catch (const GPUError& e) {
        pu_ = BIPP_PU_CPU;
      }
#else
      pu_ = BIPP_PU_CPU;
#endif
    } else {
      pu_ = pu;
    }
  }

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  auto gpu_queue() -> gpu::Queue& { return queue_; }

  auto gpu_queue() const -> const gpu::Queue& { return queue_; }
#endif

  auto processing_unit() const -> BippProcessingUnit { return pu_; }

  auto host_alloc() const -> const std::shared_ptr<Allocator>& { return hostAlloc_; }

private:
  BippProcessingUnit pu_;
  std::shared_ptr<Allocator> hostAlloc_;

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  gpu::Queue queue_;
#endif
};

struct InternalContextAccessor {
  static auto get(const Context& ctx) -> const std::shared_ptr<ContextInternal>& {
    return ctx.ctx_;
  }
};

}  // namespace bipp
