#include "bipp/communicator.hpp"

#include "bipp/config.h"

#include "bipp/exceptions.hpp"

#ifdef BIPP_MPI
#include <mpi.h>
#endif

namespace bipp {

#ifdef BIPP_MPI
Communicator::Communicator(const MPI_Comm& comm) : comm_(comm) {
  int rank = 0;
  int size = 1;
  mpi_check_status(MPI_Comm_size(comm_.value(), &size));
  mpi_check_status(MPI_Comm_rank(comm_.value(), &rank));

  rank_ = rank;
  size_ = size;
}

auto Communicator::custom(const MPI_Comm& comm) -> Communicator { return Communicator(comm); }
#endif

auto Communicator::world() -> Communicator { 
#ifdef BIPP_MPI
  return Communicator(MPI_COMM_WORLD); 
#else
  return Communicator::local();
#endif
}

auto Communicator::local() -> Communicator { return Communicator(); }

#ifdef BIPP_MPI
auto Communicator::mpi_handle() const -> const MPI_Comm& {
  if (comm_.has_value())
    return comm_.value();
  else
    return MPI_COMM_SELF;
}
#endif

}  // namespace bipp
