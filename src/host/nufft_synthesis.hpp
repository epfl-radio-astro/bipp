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
void nufft_synthesis(std::shared_ptr<ContextInternal> ctx, const NufftSynthesisOptions& opt,
                     Dataset& dataset,
                     ConstHostView<std::pair<std::size_t, const float*>, 1> samples,
                     ConstHostView<float, 1> pixelX, ConstHostView<float, 1> pixelY,
                     ConstHostView<float, 1> pixelZ, const std::string& imageTag,
                     Image& imageWriter);

}  // namespace host
}  // namespace bipp
