#include "bipp/communicator.hpp"

#include "bipp/config.h"

#include "bipp/exceptions.hpp"
#include "communicator_internal.hpp"

#ifdef BIPP_MPI
#include <mpi.h>
#endif

namespace bipp {

#ifdef BIPP_MPI
Communicator::Communicator(const MPI_Comm& comm) : comm_(new CommunicatorInternal(comm)) {}

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

auto Communicator::is_root() const -> bool {
#ifdef BIPP_MPI
  if (comm_) return comm_->is_root();
#endif
  return true;
}

auto Communicator::rank() const -> std::size_t {
#ifdef BIPP_MPI
  if (comm_) return comm_->comm().rank();
#endif
  return 0;
}

auto Communicator::size() const -> std::size_t {
#ifdef BIPP_MPI
  if (comm_) return comm_->comm().size();
#endif
  return 1;
}

extern "C" {

#ifdef BIPP_MPI
BIPP_EXPORT BippError bipp_comm_create_custom(BippCommunicator* comm, MPI_Comm mpiComm) {
  try {
    *reinterpret_cast<Communicator**>(comm) = new Communicator(Communicator::custom(mpiComm));
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_GENERIC_ERROR;
  }
  return BIPP_SUCCESS;
}
#endif

BIPP_EXPORT BippError bipp_comm_create_world(BippCommunicator* comm) {
  try {
    *reinterpret_cast<Communicator**>(comm) = new Communicator(Communicator::world());
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_GENERIC_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_comm_create_local(BippCommunicator* comm) {
  try {
    *reinterpret_cast<Communicator**>(comm) = new Communicator(Communicator::local());
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_GENERIC_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_comm_is_root(BippCommunicator comm, bool* root) {
  if (!comm) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    auto& commRef = *reinterpret_cast<Communicator*>(comm);
    *root = commRef.is_root();
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_GENERIC_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_comm_destroy(BippCommunicator* comm) {
  if (!comm || !(*comm)) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete *reinterpret_cast<Communicator**>(comm);
    *reinterpret_cast<Communicator**>(comm) = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_GENERIC_ERROR;
  }
  return BIPP_SUCCESS;
}
}
}  // namespace bipp
