#include "bipp/context.hpp"

#include "bipp/config.h"
#include "context_internal.hpp"

#ifdef BIPP_MPI
#include "communicator_internal.hpp"
#endif

namespace bipp {

Context::Context(BippProcessingUnit pu)
    : ctx_(new ContextInternal(pu)), comm_(Communicator::local()) {}

Context::Context(BippProcessingUnit pu, Communicator comm)
    : ctx_(new ContextInternal(pu)), comm_(std::move(comm)) {}

auto Context::processing_unit() const -> BippProcessingUnit{ return ctx_->processing_unit(); }

auto Context::communicator() const -> const Communicator& { return comm_; }


auto Context::attach_non_root() -> bool {
#ifdef BIPP_MPI
  if (comm_.rank() > 0) {
    auto& commInt = InternalCommunicatorAccessor::get(comm_);
    commInt->attach_non_root(InternalContextAccessor::get(*this));
    return true;
  }
#endif

  return false;
}

extern "C" {
BIPP_EXPORT BippError bipp_ctx_create(BippProcessingUnit pu, BippContext* ctx) {
  try {
    *reinterpret_cast<Context**>(ctx) = new Context(pu);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ctx_destroy(BippContext* ctx) {
  if (!ctx || !(*ctx)) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete *reinterpret_cast<Context**>(ctx);
    *reinterpret_cast<Context**>(ctx) = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}
}
}  // namespace bipp
