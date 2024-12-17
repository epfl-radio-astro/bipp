#include "bipp/image_synthesis.hpp"

#include <algorithm>
#include <chrono>
#include <complex>
#include <optional>
#include <variant>

#include "bipp/config.h"
#include "bipp/dataset.hpp"
#include "bipp/exceptions.hpp"
#include "bipp/image.hpp"
#include "context_internal.hpp"
#include "host/nufft_synthesis.hpp"
#include "memory/array.hpp"

#ifdef BIPP_MPI
#include "communicator_internal.hpp"
#endif

namespace bipp {
void image_synthesis(
    Context& ctx, const std::variant<NufftSynthesisOptions, StandardSynthesisOptions>& opt,
    Dataset& dataset,
    std::unordered_map<std::string, std::vector<std::pair<std::size_t, const float*>>> selection,
    Image& image) {
  // Check selection and sort selected samples for better performance when reading file
  for(auto& [name, samples] : selection) {
    // name must contain '/', '.', ' ' to avoid issues with hdf5 dataset names
    if (name.find('/') != std::string::npos) {
      throw InvalidParameterError("Selection name must not contain '/'");
    }
    if (name.find('.') != std::string::npos) {
      throw InvalidParameterError("Selection name must not contain '.'");
    }
    if (name.find(' ') != std::string::npos) {
      throw InvalidParameterError("Selection name must not contain spaces");
    }

    if(samples.empty()) {
      throw InvalidParameterError("Empty selection");
    }

    // sort by sample id
    std::sort(samples.begin(), samples.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    // check for duplicates
    if (samples.end() != std::adjacent_find(
                         samples.begin(), samples.end(),
                         [](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first; })) {
      throw InvalidParameterError("Selection sample id duplicate");
    }

    // make sure sample id does not exceed maximum
    if (samples.back().first >= dataset.num_samples()) {
      throw InvalidParameterError("Selection sample id exceeds number of dataset samples");
    }
  }

  ContextInternal& ctxInternal = *InternalContextAccessor::get(ctx);

  const auto numPixel = image.num_pixel();

  HostArray<float, 2> pixelXYZ(ctxInternal.host_alloc(), {numPixel, 3});

  image.pixel_lmn(pixelXYZ.data(), pixelXYZ.strides(1));

  ConstHostView<float, 1> pixelViewX = pixelXYZ.slice_view(0);
  ConstHostView<float, 1> pixelViewY = pixelXYZ.slice_view(1);
  ConstHostView<float, 1> pixelViewZ = pixelXYZ.slice_view(2);

  if (ctxInternal.processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    gpu::DeviceGuard deviceGuard(ctxInternal.device_id());  // TODO: remove.
    throw NotImplementedError();
#else
    throw GPUSupportError();
#endif
  } else {
    if (std::holds_alternative<NufftSynthesisOptions>(opt)) {
      auto nufftOpt = std::get<NufftSynthesisOptions>(opt);
      for (const auto& [tag, samples] : selection) {
        if (nufftOpt.precision == BIPP_PRECISION_SINGLE) {
          host::nufft_synthesis<float>(ctxInternal, nufftOpt, dataset,
                                       ConstHostView<std::pair<std::size_t, const float*>, 1>(
                                           samples.data(), samples.size(), 1),
                                       pixelViewX, pixelViewY, pixelViewZ, tag, image);
        } else {
          host::nufft_synthesis<double>(ctxInternal, nufftOpt, dataset,
                                        ConstHostView<std::pair<std::size_t, const float*>, 1>(
                                            samples.data(), samples.size(), 1),
                                        pixelViewX, pixelViewY, pixelViewZ, tag, image);
        }
      }

    } else {
      throw NotImplementedError();
    }
  }
}

}  // namespace bipp
