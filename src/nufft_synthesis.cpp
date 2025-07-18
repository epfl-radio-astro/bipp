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
#include "host/blas_api.hpp"
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
  auto funcTimer = globLogger.scoped_timing(BIPP_LOG_LEVEL_INFO, "host::nufft_synthesis");

  const auto nPixel = pixelXYZ.shape(0);
  const auto nImages = images.shape(1);
  const auto nBeam = dataset.num_beam();
  const auto nAntenna = dataset.num_antenna();
  const auto nBaselines = nAntenna * nAntenna;

  globLogger.log(BIPP_LOG_LEVEL_INFO, "nPixel: {}, nImages: {}, nBeam: {}, nAntenna: {}", nPixel,
                 nImages, nBeam, nAntenna);

  assert(images.shape(0) == nPixel);

  assert(sampleIds.size() == dScaled.shape(1));
  assert(images.shape(1) == dScaled.shape(2));
  assert(nBeam == dScaled.shape(0));

  // Use at most 2GB for batching. Estimate the memory usage at 20 bytes per baseline independent of
  // precision, to ensure consistency.
  const std::size_t sampleBatchSize =
      std::min<std::size_t>((opt.sampleBatchSize.has_value() && opt.sampleBatchSize.value() > 0)
                                ? opt.sampleBatchSize.value()
                                : (2 * 1000 * 1000 * 1000) / (20 * nBaselines),
                            sampleIds.size());
  globLogger.log(BIPP_LOG_LEVEL_INFO, "sampleBatchSize: {}", sampleBatchSize);

  // copy pixel values to double precision if required
  HostArray<double, 2> pixelArray;
  ConstHostView<T, 2> pixelXYZConverted;
  if constexpr (std::is_same_v<T, double>) {
    pixelArray = HostArray<double, 2>(ctx.host_alloc(), {pixelXYZ.shape()});
    copy(pixelXYZ, pixelArray);
    pixelXYZConverted = pixelArray;
  } else {
    pixelXYZConverted = pixelXYZ;
  }

  std::unique_ptr<NUFFTInterface<T>> nufft;
  if (ctx.processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    nufft.reset(
        new gpu::NUFFT<T>(ctxPtr, opt, pixelXYZConverted, nImages, nBaselines, sampleBatchSize));
#else
    throw GPUSupportError();
#endif
  } else {
    nufft.reset(
        new host::NUFFT<T>(ctxPtr, opt, pixelXYZConverted, nImages, nBaselines, sampleBatchSize));
  }

  HostArray<T, 2> uvw(ctx.host_alloc(), {nBaselines, 3});
  HostArray<std::complex<T>, 2> virtualVis(ctx.host_alloc(), {nBaselines, nImages});
  HostArray<std::complex<T>, 2> eigVec(ctx.host_alloc(), {nAntenna, nBeam});
  HostArray<T, 1> dSlice(ctx.host_alloc(), nBeam);
  for (std::size_t i = 0; i < sampleIds.size(); ++i) {
    const auto id = sampleIds[i];

    const T wl = dataset.wl(id);
    const T scale = dataset.scale(id);

    globLogger.log(BIPP_LOG_LEVEL_DEBUG, "sample id: {}", id);
    globLogger.start_timing(BIPP_LOG_LEVEL_INFO, "read uvw");
    read_uvw(ctx, dataset, id, uvw);
    globLogger.stop_timing(BIPP_LOG_LEVEL_INFO, "read uvw");
    globLogger.start_timing(BIPP_LOG_LEVEL_INFO, "read eig vec");
    read_eig_vec(ctx, dataset, id, eigVec);
    globLogger.stop_timing(BIPP_LOG_LEVEL_INFO, "read eig vec");
    globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", eigVec);

    // scale uvw
    globLogger.start_timing(BIPP_LOG_LEVEL_INFO, "scale uvw");
    constexpr auto twoPi = T(2 * 3.14159265358979323846);
    const T uvwScale = twoPi / wl;
    assert(uvw.is_contiguous());
    host::blas::scal(uvw.size(), uvwScale, uvw.data(), 1);
    globLogger.stop_timing(BIPP_LOG_LEVEL_INFO, "scale uvw");

    for (std::size_t imageIdx = 0; imageIdx < nImages; ++imageIdx) {
      copy(dScaled.slice_view(imageIdx).slice_view(i), dSlice);
      globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "scaled eigenvalues", dSlice);
      host::virtual_vis<T>(ctx, scale, dSlice, eigVec, virtualVis.slice_view(imageIdx));
      globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "virtual vis", virtualVis.slice_view(imageIdx));
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
