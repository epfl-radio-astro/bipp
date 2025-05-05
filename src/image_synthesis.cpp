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
#include "bipp/image_prop.hpp"
#include "bipp/image_data.hpp"
#include "bipp/image_data_file.hpp"
#include "bipp/communicator.hpp"
#include "context_internal.hpp"
#include "nufft_synthesis.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"

#ifdef BIPP_MPI
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_data_type.hpp"
#endif

namespace bipp {
void image_synthesis(
    Context& ctx, const NufftSynthesisOptions& opt, Dataset& dataset,
    std::unordered_map<std::string, std::vector<std::pair<std::size_t, const float*>>> selection,
    ImageProp& imageProp, const std::string& imageFileName) {
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

  const auto& comm = ctx.communicator();

  // make sure sample ids are the same for all imageArray
  // TODO: group by matching ids
  const auto nTotalSamples = selection.begin()->second.size();

  const auto nSamplePerRank = (nTotalSamples + comm.size() - 1) / comm.size();
  const auto localSampleBegin = comm.rank() * nSamplePerRank;
  const auto nLocalSamples = std::min<std::size_t>(
      nSamplePerRank, std::max<std::size_t>(nTotalSamples - localSampleBegin, 0));

  for(const auto& name : imageNames) {
    if (selection[name].size() != nTotalSamples) {
      throw InvalidParameterError("Selection must contain the same sample ids for all images.");
    }
  }


  // check and store sample ids
  HostArray<std::size_t, 1> localSampleIds(ctxInternal->host_alloc(), nLocalSamples);
  {
    const auto& sourceIds = selection.begin()->second;

    for (std::size_t idx = 0; idx < nLocalSamples; ++idx) {
      localSampleIds[idx] = sourceIds[idx + localSampleBegin].first;
    }
  }

  for (const auto& name : imageNames) {
    const auto& sourceIds = selection[name];
    for (std::size_t idx = 0; idx < nLocalSamples; ++idx) {
      if (sourceIds[idx + localSampleBegin].first != localSampleIds[idx]) {
        throw InvalidParameterError("Selection must contain the same sample ids for all images.");
      }
    }
  }


  // store all scaling values for all samples and imageArray
  const auto nBeam = dataset.num_beam();
  const auto nImages = imageNames.size();
  HostArray<float, 3> dScaled(ctxInternal->host_alloc(), {nBeam, nLocalSamples, nImages});

  for (std::size_t idxImg = 0; idxImg < nImages; ++idxImg) {
    const auto& sourceIds = selection[imageNames[idxImg]];
    auto dScaledSlice = dScaled.slice_view(idxImg);
    for (std::size_t idx = 0; idx < nLocalSamples; ++idx) {
      copy(ConstHostView<float, 1>(sourceIds[idx + localSampleBegin].second, nBeam, 1),
           dScaledSlice.slice_view(idx));
    }
    globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "scaled eigenvalues", dScaledSlice);
  }

  unsigned long long nPixel = imageProp.num_pixel();

  HostArray<float, 2> pixelXYZ(ctxInternal->host_alloc(), {nPixel, 3});
  imageProp.pixel_lmn(pixelXYZ.data(), pixelXYZ.strides(1));

  HostArray<float, 2> imageArray(ctxInternal->host_alloc(), {nPixel, nImages});
  imageArray.zero();

  if (opt.precision == BIPP_PRECISION_SINGLE) {
    nufft_synthesis<float>(ctxInternal, opt, dataset, pixelXYZ, localSampleIds, dScaled,
                           imageArray);
  } else {
    nufft_synthesis<double>(ctxInternal, opt, dataset, pixelXYZ, localSampleIds, dScaled,
                            imageArray);
  }

#ifdef BIPP_MPI
  if(comm.size() > 1) {
    assert(imageArray.is_contiguous());
    void* sendPtr = comm.rank() == 0 ? MPI_IN_PLACE : imageArray.data();
    void* recvPtr = comm.rank() == 0 ? imageArray.data() : nullptr;
    mpi_check_status(MPI_Reduce(sendPtr, recvPtr, imageArray.size(),
                                MPIType<float>::get(), MPI_SUM, 0, comm.mpi_handle()));
  }
#endif

  if (comm.rank() == 0) {

    if (opt.normalizeImage) {
      globLogger.start_timing(BIPP_LOG_LEVEL_INFO, "scale image");
      const float scale = 1.f / float(nTotalSamples);

      for (std::size_t idxImg = 0; idxImg < nImages; ++idxImg) {
        auto imageSlice = imageArray.slice_view(idxImg);
        for (std::size_t j = 0; j < nPixel; ++j) {
          imageSlice[j] *= scale;
        }
      }
      globLogger.stop_timing(BIPP_LOG_LEVEL_INFO, "scale image");
    }

    // write image
    auto imageData = ImageDataFile::create(imageFileName, nPixel);
    globLogger.start_timing(BIPP_LOG_LEVEL_INFO, "write image");
    for (std::size_t idxImg = 0; idxImg < nImages; ++idxImg) {
      globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "image", imageArray.slice_view(idxImg));
      imageData.set(imageNames[idxImg], &imageArray[{0, idxImg}]);
    }
    globLogger.stop_timing(BIPP_LOG_LEVEL_INFO, "write image");
  }

#ifdef BIPP_MPI
    if (comm.size() > 1) {
      MPI_Barrier(comm.mpi_handle());
    }
#endif
}

}  // namespace bipp
