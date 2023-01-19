#pragma once

#include "bipp/config.h"
#include "gpu/util/runtime_api.hpp"

#ifdef BIPP_ROCM
#include <hip/hip_runtime.h>
#endif
#ifdef BIPP_CUDA
#include <cuda_runtime.h>
#endif

namespace bipp {
namespace gpu {
namespace api {

template <typename F, typename... ARGS>
inline auto launch_kernel(F func, const dim3 threadGrid, const dim3 threadBlock,
                          const size_t sharedMemoryBytes, const StreamType stream, ARGS&&... args)
    -> void {
#ifdef BIPP_CUDA
  func<<<threadGrid, threadBlock, sharedMemoryBytes, stream>>>(std::forward<ARGS>(args)...);
#elif defined(BIPP_ROCM)
  hipLaunchKernelGGL(func, threadGrid, threadBlock, sharedMemoryBytes, stream,
                     std::forward<ARGS>(args)...);
#endif
}

}  // namespace api
}  // namespace gpu
}  // namespace bipp
