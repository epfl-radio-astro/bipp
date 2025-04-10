#include "nufft_synthesis.hpp"

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
#include "host/virtual_vis.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "nufft_util.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/nufft.hpp"
#endif

namespace bipp {

namespace {
auto system_memory() -> unsigned long long {
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
}  // namespace

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
      host::virtual_vis<T>(ctx, dSlice, eigVec, virtualVis.slice_view(imageIdx));
      ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "virtual vis", virtualVis.slice_view(imageIdx));
    }

    nufft->add(uvw, virtualVis);
  }

  for (std::size_t imageIdx = 0; imageIdx < nImages; ++imageIdx) {
    nufft->get_image(imageIdx, images.slice_view(imageIdx));
  }
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

}  // namespace bipp
