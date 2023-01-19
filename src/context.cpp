#include "bipp/context.hpp"

#include "bipp/config.h"
#include "context_internal.hpp"

namespace bipp {

Context::Context(BippProcessingUnit pu) : ctx_(new ContextInternal(pu)) {}

BippProcessingUnit Context::processing_unit() const { return ctx_->processing_unit(); }

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
