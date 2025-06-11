#include "bipp/context.hpp"

#include "bipp/config.h"
#include "context_internal.hpp"

namespace bipp {

Context::Context(BippProcessingUnit pu)
    : ctx_(new ContextInternal(pu)), comm_(Communicator::local()) {}

Context::Context(BippProcessingUnit pu, Communicator comm)
    : ctx_(new ContextInternal(pu)), comm_(std::move(comm)) {}

auto Context::processing_unit() const -> BippProcessingUnit{ return ctx_->processing_unit(); }

auto Context::communicator() const -> const Communicator& { return comm_; }

}  // namespace bipp
