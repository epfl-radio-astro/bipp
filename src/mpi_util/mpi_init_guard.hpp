#pragma once


#include "bipp/config.h"

namespace bipp {

#ifdef BIPP_MPI
auto initialize_mpi_init_guard() -> void;
#endif

}  // namespace bipp
