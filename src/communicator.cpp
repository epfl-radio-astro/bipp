#include "bipp/communicator.hpp"

#include "bipp/config.h"

#include "communicator_internal.hpp"


#ifdef BIPP_MPI
#include <mpi.h>
#endif

namespace bipp {

#ifdef BIPP_MPI
Communicator::Communicator(const MPI_Comm& comm) : comm_(new CommunicatorInternal(comm)) {}
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

// extern "C" {
// BIPP_EXPORT BippError bipp_ctx_create(BippProcessingUnit pu, BippCommunicator* ctx) {
//   try {
//     *reinterpret_cast<Communicator**>(ctx) = new Communicator(pu);
//   } catch (const bipp::GenericError& e) {
//     return e.error_code();
//   } catch (...) {
//     return BIPP_UNKNOWN_ERROR;
//   }
//   return BIPP_SUCCESS;
// }

// BIPP_EXPORT BippError bipp_ctx_destroy(BippCommunicator* ctx) {
//   if (!ctx || !(*ctx)) {
//     return BIPP_INVALID_HANDLE_ERROR;
//   }
//   try {
//     delete *reinterpret_cast<Communicator**>(ctx);
//     *reinterpret_cast<Communicator**>(ctx) = nullptr;
//   } catch (const bipp::GenericError& e) {
//     return e.error_code();
//   } catch (...) {
//     return BIPP_UNKNOWN_ERROR;
//   }
//   return BIPP_SUCCESS;
// }
// }
}  // namespace bipp
