#pragma once

#include "bipp/communicator.hpp"
#include "bipp/config.h"
#include "bipp/dataset.hpp"
#include "bipp/image_synthesis.hpp"
#include "context_internal.hpp"
#include "io/image_file_writer.hpp"

namespace bipp {
namespace host {
template <typename T>
void nufft_synthesis(ContextInternal& ctx, const NufftSynthesisOptions& opt, Dataset& dataset,
                     ConstHostView<std::pair<std::size_t, const float*>, 1> samples,
                     ConstHostView<float, 1> pixelX, ConstHostView<float, 1> pixelY,
                     ConstHostView<float, 1> pixelZ, const std::string& imageTag,
                     ImageFileWriter& imageWriter);

}  // namespace host
}  // namespace bipp
