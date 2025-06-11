#pragma once

#include "bipp/config.h"
#include "bipp/exceptions.hpp"

#ifdef BIPP_MPI

#include <mpi.h>


namespace bipp {
inline auto mpi_check_status(int status) -> void {
  if (status != MPI_SUCCESS) {
    throw MPIError();
  }
}
}  // namespace bipp

#endif
