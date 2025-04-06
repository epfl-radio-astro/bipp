#include "bipp/image_synthesis.hpp"

#include <algorithm>
#include <chrono>
#include <complex>
#include <memory>
#include <optional>
#include <variant>

#include "bipp/config.h"
#include "bipp/dataset.hpp"
#include "bipp/exceptions.hpp"
#include "bipp/image.hpp"
#include "context_internal.hpp"
#include "host/nufft_synthesis.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"

#ifdef BIPP_MPI
#include "communicator_internal.hpp"
#endif

namespace bipp {
void image_synthesis(
    Context& ctx, const std::variant<NufftSynthesisOptions, StandardSynthesisOptions>& opt,
    Dataset& dataset,
    std::unordered_map<std::string, std::vector<std::pair<std::size_t, const float*>>> selection,
    Image& image) {
  std::shared_ptr<ContextInternal> ctxInternal = InternalContextAccessor::get(ctx);

  // Check selection and sort selected samples for better performance when reading file
  std::vector<std::string> imageNames;
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

    imageNames.emplace_back(name);
  }

  // make sure sample ids are the same for all imageArray
  // TODO: group by matching ids
  const auto nSamples = selection.begin()->second.size();
  for(const auto& name : imageNames) {
    if(selection[name].size() != nSamples) {
      throw InvalidParameterError("Selection must contain the same sample ids for all imageArray");
    }
  }


  // check and store sample ids
  HostArray<std::size_t, 1> sampleIds(ctxInternal->host_alloc(), nSamples);
  {
    const auto& sourceIds = selection.begin()->second;

    for (std::size_t idx = 0; idx < nSamples; ++idx) {
      sampleIds[idx] = sourceIds[idx].first;
    }
  }

  for (const auto& name : imageNames) {
    const auto& sourceIds = selection[name];
    for (std::size_t idx = 0; idx < nSamples; ++idx) {
      if (sourceIds[idx].first != sampleIds[idx]) {
        throw InvalidParameterError("Selection must contain the same sample ids for all imageArray");
      }
    }
  }


  // store all scaling values for all samples and imageArray
  const auto nBeam = dataset.num_beam();
  const auto nImages = imageNames.size();
  HostArray<float, 3> dScaled(ctxInternal->host_alloc(), {nBeam, nSamples, nImages});

  for (std::size_t idxImg = 0; idxImg < nImages; ++idxImg) {
    const auto& sourceIds = selection[imageNames[idxImg]];
    auto dScaledSlice = dScaled.slice_view(idxImg);
    for (std::size_t idx = 0; idx < nSamples; ++idx) {
      copy(ConstHostView<float, 1>(sourceIds[idx].second, nBeam, 1), dScaledSlice.slice_view(idx));
    }
    ctxInternal->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "scaled eigenvalues", dScaledSlice);
  }

  const auto nPixel = image.num_pixel();

  HostArray<float, 2> pixelXYZ(ctxInternal->host_alloc(), {nPixel, 3});

  image.pixel_lmn(pixelXYZ.data(), pixelXYZ.strides(1));


  HostArray<float, 2> imageArray(ctxInternal->host_alloc(), {nPixel, nImages});
  imageArray.zero();

  if (std::holds_alternative<NufftSynthesisOptions>(opt)) {
    auto nufftOpt = std::get<NufftSynthesisOptions>(opt);
    if (nufftOpt.precision == BIPP_PRECISION_SINGLE) {
      host::nufft_synthesis<float>(ctxInternal, nufftOpt, dataset, pixelXYZ, sampleIds, dScaled,
                                   imageArray);
    } else {
      host::nufft_synthesis<double>(ctxInternal, nufftOpt, dataset, pixelXYZ, sampleIds, dScaled,
                                    imageArray);
    }

    if (nufftOpt.normalizeImage) {
      ctxInternal->logger().start_timing(BIPP_LOG_LEVEL_INFO, "scale image");
      const float scale = 1.f / float(nSamples);

      for (std::size_t idxImg = 0; idxImg < nImages; ++idxImg) {
        auto imageSlice = imageArray.slice_view(idxImg);
        for (std::size_t j = 0; j < nPixel; ++j) {
          imageSlice[j] *= scale;
        }
      }
      ctxInternal->logger().stop_timing(BIPP_LOG_LEVEL_INFO, "scale image");
    }

  } else {
    throw NotImplementedError();
  }

  // write image
  ctxInternal->logger().start_timing(BIPP_LOG_LEVEL_INFO, "write imageArray");
  for (std::size_t idxImg = 0; idxImg < nImages; ++idxImg) {
    ctxInternal->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image", imageArray.slice_view(idxImg));
    image.set(imageNames[idxImg], &imageArray[{0, idxImg}]);
  }
  ctxInternal->logger().stop_timing(BIPP_LOG_LEVEL_INFO, "write imageArray");
}

}  // namespace bipp
