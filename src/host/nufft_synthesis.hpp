#pragma once
#include <memory>

#include "bipp/communicator.hpp"
#include "bipp/config.h"
#include "bipp/dataset.hpp"
#include "bipp/image.hpp"
#include "bipp/image_synthesis.hpp"
#include "context_internal.hpp"

namespace bipp {
namespace host {

template <typename T>
void nufft_synthesis(std::shared_ptr<ContextInternal> ctxPtr, const NufftSynthesisOptions& opt,
                     Dataset& dataset, ConstHostView<float, 2> pixelXYZ,
                     ConstHostView<std::size_t, 1> sampleIds, ConstHostView<float, 3> dScaled,
                     HostView<float, 2> images);

}  // namespace host
}  // namespace bipp
