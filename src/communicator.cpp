#include "bipp/communicator.hpp"

#include "bipp/config.h"
#include "communicator_internal.hpp"
#include "context_internal.hpp"

namespace bipp {

Communicator::Communicator(const MPI_Comm& comm) : comm_(new CommunicatorInternal(comm)) {}

auto Communicator::is_root() const -> bool { return comm_->is_root(); }

auto Communicator::attach_non_root(Context& ctx) -> void {
  comm_->attach_non_root(InternalContextAccessor::get(ctx));
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
