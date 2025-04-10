

#include "bipp/config.h"

#ifdef BIPP_MPI
#include <mpi.h>

#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_init_guard.hpp"

namespace bipp {

namespace {
struct MPIInitGuard {
  MPIInitGuard() = default;

  auto initialize() -> void {
    int initialized = 0;
    mpi_check_status(MPI_Initialized(&initialized));
    if (!initialized) {
      int provided = MPI_THREAD_FUNNELED;
      MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);
      call_finalize = true;
    }
  }

  ~MPIInitGuard() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && call_finalize) MPI_Finalize();
  }

  MPIInitGuard(const MPIInitGuard&) = delete;
  MPIInitGuard(MPIInitGuard&& g) = delete;
  auto operator=(const MPIInitGuard&) -> MPIInitGuard& = delete;
  auto operator=(MPIInitGuard&& g) -> MPIInitGuard& = delete;

  bool call_finalize = false;
};


MPIInitGuard mpiInitGuard;

}  // namespace

auto initialize_mpi_init_guard() -> void { mpiInitGuard.initialize(); }

}  // namespace bipp
#endif
