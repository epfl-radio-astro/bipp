#pragma once

#include <complex>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "bipp/image_synthesis.hpp"
#include "context_internal.hpp"
#include "host/domain_partition.hpp"
#include "memory/array.hpp"
#include "io/dataset_file_reader.hpp"
#include "io/image_file_writer.hpp"
#include "bipp/communicator.hpp"

namespace bipp {
namespace host {
template <typename T>
void nufft_synthesis(const Communicator& comm, ContextInternal& ctx,
                     const NufftSynthesisOptions& opt, DatasetFileReader& datasetReader,
                     ConstHostView<std::pair<std::size_t, const float*>, 1> samples,
                     ConstHostView<float, 1> pixelX, ConstHostView<float, 1> pixelY,
                     ConstHostView<float, 1> pixelZ, const std::string& imageTag,
                     ImageFileWriter& imageWriter);

}  // namespace host
}  // namespace bipp
