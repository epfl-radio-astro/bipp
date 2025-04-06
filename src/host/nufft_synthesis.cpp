#include "host/nufft_synthesis.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstring>
#include <limits>
#include <memory>
#include <neonufft/plan.hpp>
#include <numeric>
#include <type_traits>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "host/domain_partition.hpp"
#include "host/nufft.hpp"
#include "host/uvw_partition.hpp"
#include "host/virtual_vis.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "nufft_util.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/nufft.hpp"
#endif

namespace bipp {
namespace host {

static auto system_memory() -> unsigned long long {
  unsigned long long pages = sysconf(_SC_PHYS_PAGES);
  unsigned long long pageSize = sysconf(_SC_PAGE_SIZE);
  unsigned long long memory = pages * pageSize;
  return memory > 0 ? memory : 8ull * 1024ull * 1024ull * 1024ull;
}

auto read_eig_val(ContextInternal& ctx, Dataset& dataset, std::size_t index, HostView<float, 1> d) {
  dataset.eig_val(index, d.data());
}

auto read_eig_val(ContextInternal& ctx, Dataset& dataset, std::size_t index,
                  HostView<double, 1> d) {
  HostArray<float, 1> buffer(ctx.host_alloc(), d.shape());
  dataset.eig_val(index, buffer.data());
  copy(buffer, d);
}

auto read_eig_vec(ContextInternal& ctx, Dataset& dataset, std::size_t index,
                  HostView<std::complex<float>, 2> v) {
  dataset.eig_vec(index, v.data(), v.strides(1));
}

auto read_eig_vec(ContextInternal& ctx, Dataset& dataset, std::size_t index,
                  HostView<std::complex<double>, 2> v) {
  HostArray<std::complex<float>, 2> buffer(ctx.host_alloc(), v.shape());
  dataset.eig_vec(index, buffer.data(), buffer.strides(1));
  copy(buffer, v);
}

auto read_uvw(ContextInternal& ctx, Dataset& dataset, std::size_t index, HostView<float, 2> uvw) {
  dataset.uvw(index, uvw.data(), uvw.strides(1));
}

auto read_uvw(ContextInternal& ctx, Dataset& dataset, std::size_t index, HostView<double, 2> uvw) {
  HostArray<float, 2> buffer(ctx.host_alloc(), uvw.shape());
  dataset.uvw(index, buffer.data(), buffer.strides(1));
  copy(buffer, uvw);
}

template <typename T>
void nufft_synthesis(std::shared_ptr<ContextInternal> ctxPtr, const NufftSynthesisOptions& opt,
                     Dataset& dataset, ConstHostView<float, 2> pixelXYZ,
                     ConstHostView<std::size_t, 1> sampleIds, ConstHostView<float, 3> dScaled,
                     HostView<float, 2> images) {
  auto& ctx = *ctxPtr;
  auto funcTimer = ctx.logger().scoped_timing(BIPP_LOG_LEVEL_INFO, "host::nufft_synthesis");

  const auto nPixel = pixelXYZ.shape(0);
  const auto nImages = images.shape(1);
  const auto nBeam = dataset.num_beam();
  const auto nAntenna = dataset.num_antenna();
  const auto nBaselines = nAntenna * nAntenna;

  assert(images.shape(0) == nPixel);

  assert(sampleIds.size() == dScaled.shape(1));
  assert(images.shape(1) == dScaled.shape(2));
  assert(nBeam == dScaled.shape(0));

  //TODO: decide based on host memory size
  const std::size_t maxCollectGroupSize = std::max<std::size_t>(200, sampleIds.size());


  // copy pixel values to double precision if required
  HostArray<double, 2> pixelArray;
  ConstHostView<T, 2> pixelXYZConverted;
  if constexpr(std::is_same_v<T, double>) {
    pixelArray = HostArray<double, 2>(ctx.host_alloc(), {pixelXYZ.shape()});
    copy(pixelXYZ, pixelArray);
    pixelXYZConverted = pixelArray;
  }else {
    pixelXYZConverted = pixelXYZ;
  }

  std::unique_ptr<NUFFTInterface<T>> nufft;
  if (ctx.processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    nufft.reset(new gpu::NUFFT<T>(ctxPtr, opt, pixelXYZConverted, nImages, nBaselines,
                                  maxCollectGroupSize));
#else
    throw GPUSupportError();
#endif
  } else {
    nufft.reset(new host::NUFFT<T>(ctxPtr, opt, pixelXYZConverted, nImages, nBaselines,
                                   maxCollectGroupSize));
  }

  HostArray<T, 2> uvw(ctx.host_alloc(), {nBaselines, 3});
  HostArray<std::complex<T>, 2> virtualVis(ctx.host_alloc(), {nBaselines, nImages});
  HostArray<std::complex<T>, 2> eigVec(ctx.host_alloc(), {nAntenna, nBeam});
  HostArray<T, 1> dSlice(ctx.host_alloc(), nBeam);
  for (std::size_t i = 0; i < sampleIds.size(); ++i) {
    const auto id = sampleIds[i];

    ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "sample id: {}", id);
    ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "read uvw");
    read_uvw(ctx, dataset, id, uvw);
    ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "read uvw");
    ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "read eig vec");
    read_eig_vec(ctx, dataset, id, eigVec);
    ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "read eig vec");
    ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", eigVec);

    for (std::size_t imageIdx = 0; imageIdx < nImages; ++imageIdx) {
      copy(dScaled.slice_view(imageIdx).slice_view(i), dSlice);
      ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "scaled eigenvalues", dSlice);
      virtual_vis<T>(ctx, dSlice, eigVec, virtualVis.slice_view(imageIdx));
      ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "virtual vis", virtualVis.slice_view(imageIdx));
    }

    nufft->add(uvw, virtualVis);
  }

  for (std::size_t imageIdx = 0; imageIdx < nImages; ++imageIdx) {
    nufft->get_image(imageIdx, images.slice_view(imageIdx));
  }

  //---------------

  /*
  for (std::size_t sampleStartIdx = 0; sampleStartIdx < sampleIds.size();
       sampleStartIdx += maxCollectGroupSize) {
    const std::size_t collectGroupSize =
        std::min<std::size_t>(sampleIds.size() - sampleStartIdx, maxCollectGroupSize);

    // read uvw
    HostArray<T, 2> uvwCollection(ctx.host_alloc(), {collectGroupSize * nBaselines, 3});

    HostArray<T, 2> uvw(ctx.host_alloc(), {nAntenna * nAntenna, 3});
    for (std::size_t i = 0; i < collectGroupSize; ++i) {
      ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "read uvw");
      read_uvw(ctx, dataset, sampleIds[sampleStartIdx + i], uvw);
      ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "read uvw");
      copy(uvw, uvwCollection.sub_view({i * nBaselines, 0}, {nBaselines, 3}));
    }

    // partition uvw
    DomainPartition uvwPartition = std::visit(
        [&](auto&& arg) -> DomainPartition {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
            ctx.logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: grid ({}, {}, {})",
                             arg.dimensions[0], arg.dimensions[1], arg.dimensions[2]);
            return DomainPartition::grid<T, 3>(
                ctx.host_alloc(), arg.dimensions,
                {uvwCollection.slice_view(0), uvwCollection.slice_view(1),
                 uvwCollection.slice_view(2)});
          } else if constexpr (std::is_same_v<ArgType, Partition::None> ||
                               std::is_same_v<ArgType, Partition::Auto>) {
            // TODO: AUTO partitioning
            ctx.logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: none");
            return DomainPartition::none(ctx.host_alloc(), uvwCollection.shape(0));
          }
        },
        opt.localUVWPartition.method);

    uvwPartition.apply(uvwCollection.slice_view(0));
    uvwPartition.apply(uvwCollection.slice_view(1));
    uvwPartition.apply(uvwCollection.slice_view(2));

    // compute virtual visibilities
    HostArray<std::complex<T>, 2> virtualVisCollection(ctx.host_alloc(),
                                                       {collectGroupSize * nBaselines, nImages});

    HostArray<std::complex<T>, 2> eigVec(ctx.host_alloc(), {nAntenna, nBeam});
    HostArray<T, 1> dBuffer(ctx.host_alloc(), nBeam);

    for (std::size_t i = 0; i < collectGroupSize; ++i) {
      const auto id = sampleIds[sampleStartIdx + i];
      ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "read eig vec");
      read_eig_vec(ctx, dataset, id, eigVec);
      ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "read eig vec");
      ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", eigVec);

      ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "virtual vis");
      for (std::size_t imageIdx = 0; imageIdx < nImages; ++imageIdx) {
        copy(dScaled.slice_view(imageIdx).slice_view(sampleStartIdx + i), dBuffer);
        ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "scaled eigenvalues", dBuffer);
        virtual_vis<T>(
            ctx, dBuffer, eigVec,
            virtualVisCollection.slice_view(imageIdx).sub_view(i * nBaselines, nBaselines));
        ctx.logger().log_matrix(
            BIPP_LOG_LEVEL_DEBUG, "virtual vis",
            virtualVisCollection.slice_view(imageIdx).sub_view(i * nBaselines, nBaselines));
      }
      ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "virtual vis");
    }

    for (std::size_t imageIdx = 0; imageIdx < nImages; ++imageIdx) {
      uvwPartition.apply(virtualVisCollection.slice_view(imageIdx));
    }

    for (const auto& [uvwBegin, uvwSize] : uvwPartition.groups()) {
      if (!uvwSize) continue;

      auto uvwPart = uvwCollection.sub_view({uvwBegin, 0}, {uvwSize, 3});

      auto uView = uvwPart.slice_view(0);
      auto vView = uvwPart.slice_view(1);
      auto wView = uvwPart.slice_view(2);

      auto uMinMax = std::minmax_element(uView.data(), uView.data() + uView.size());
      auto vMinMax = std::minmax_element(vView.data(), vView.data() + vView.size());
      auto wMinMax = std::minmax_element(wView.data(), wView.data() + wView.size());

      std::array<T, 3> uvwMin = {*uMinMax.first, *vMinMax.first, *wMinMax.first};
      std::array<T, 3> uvwMax = {*uMinMax.second, *vMinMax.second, *wMinMax.second};

      ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "prepare nufft");
      std::unique_ptr<NUFFTInterface<T>> nufft;
      if (ctx.processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
        ctx.gpu_queue().sync();  // make sure previous memory is freed
        nufft.reset(new gpu::NUFFT<T>(ctxPtr, neoOpt, 1, uvwMin, uvwMax, uvwPart, pixelMin,
                                      pixelMax, pixelX, pixelY, pixelZ));
#else
        throw GPUSupportError();
#endif
      } else {
        nufft.reset(new host::NUFFT<T>(ctxPtr, neoOpt, 1, uvwMin, uvwMax, uvwPart, pixelMin,
                                       pixelMax, pixelX, pixelY, pixelZ));
      }
      ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "prepare nufft");

      auto virtualVisPart =
          virtualVisCollection.sub_view({uvwBegin, 0}, {uvwSize, virtualVisCollection.shape(1)});
      for (std::size_t imageIdx = 0; imageIdx < nImages; ++imageIdx) {
        ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "compute nufft");
        ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft input u", uvwCollection.slice_view(0));
        ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft input v", uvwCollection.slice_view(1));
        ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft input w", uvwCollection.slice_view(2));
        ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft input values",
                                virtualVisCollection.slice_view(imageIdx));
        nufft->transform_and_add(virtualVisPart.slice_view(imageIdx), images.slice_view(imageIdx));
        ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "compute nufft");
        // TODO call reset if api changed in neonufft
      }
    }
  }
*/
}

template void nufft_synthesis<float>(std::shared_ptr<ContextInternal> ctxPtr,
                                     const NufftSynthesisOptions& opt, Dataset& dataset,
                                     ConstHostView<float, 2> pixelXYZ,
                                     ConstHostView<std::size_t, 1> sampleIds,
                                     ConstHostView<float, 3> dScaled, HostView<float, 2> images);

template void nufft_synthesis<double>(std::shared_ptr<ContextInternal> ctxPtr,
                                      const NufftSynthesisOptions& opt, Dataset& dataset,
                                      ConstHostView<float, 2> pixelXYZ,
                                      ConstHostView<std::size_t, 1> sampleIds,
                                      ConstHostView<float, 3> dScaled, HostView<float, 2> images);

}  // namespace host
}  // namespace bipp
