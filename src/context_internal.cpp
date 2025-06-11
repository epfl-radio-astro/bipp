#include "context_internal.hpp"

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "bipp/enums.h"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/blas_api.hpp"
#include "gpu/util/device_guard.hpp"
#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

#ifdef BIPP_MPI
#include <mpi.h>

#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_init_guard.hpp"
#endif

namespace bipp {

ContextInternal::ContextInternal(BippProcessingUnit pu)
    : hostAlloc_(AllocatorFactory::host()), deviceId_(0) {
#ifdef BIPP_MPI
  initialize_mpi_init_guard();
  int myRank = 0;
  mpi_check_status(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
#else
  int myRank = 0;
#endif

  if (pu == BIPP_PU_AUTO) {
    // select GPU if available
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    try {
      int count = 0;
      gpu::api::get_device_count(&count);

      if (count) {
        pu_ = BIPP_PU_GPU;
      } else {
        pu_ = BIPP_PU_CPU;
      }
    } catch (const GPUError& e) {
      pu_ = BIPP_PU_CPU;
    }
#else
    pu_ = BIPP_PU_CPU;
#endif
  } else {
    pu_ = pu;
  }

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  if (pu_ == BIPP_PU_GPU) {
    int count = 0;
    gpu::api::get_device_count(&count);
    deviceId_ = myRank % count;
    gpu::DeviceGuard deviceGuard(deviceId_);
    queue_.emplace();
  }
#endif

  if (pu_ == BIPP_PU_CPU)
    globLogger.log(BIPP_LOG_LEVEL_INFO, "{} CPU context created", static_cast<const void*>(this));
  else
    globLogger.log(BIPP_LOG_LEVEL_INFO, "{} GPU context created", static_cast<const void*>(this));
}


}  // namespace bipp
