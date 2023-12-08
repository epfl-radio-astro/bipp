#include "imager.hpp"

#include <cassert>
#include <complex>
#include <cstddef>
#include <memory>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "host/collector.hpp"
#include "host/eigensolver.hpp"
#include "host/nufft_synthesis.hpp"
#include "host/standard_synthesis.hpp"
#include "memory/copy.hpp"
#include "synthesis_factory.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/collector.hpp"
#include "gpu/eigensolver.hpp"
#include "gpu/kernels/center_vector.hpp"
#include "gpu/nufft_synthesis.hpp"
#include "gpu/standard_synthesis.hpp"
#include "gpu/util/device_accessor.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

namespace bipp {
template <typename T>
static auto center_vector(std::size_t n, const T* __restrict__ in, T* __restrict__ out) -> void {
  T mean = 0;
  for (std::size_t i = 0; i < n; ++i) {
    mean += in[i];
  }
  mean /= n;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = in[i] - mean;
  }
}

template <typename T>
auto Imager<T>::standard_synthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nLevel,
                                   ConstView<BippFilter, 1> filter, ConstView<T, 1> pixelX,
                                   ConstView<T, 1> pixelY, ConstView<T, 1> pixelZ) -> Imager<T> {
  assert(ctx);
  return Imager(ctx, SynthesisFactory<T>::create_standard_synthesis(ctx, nLevel, filter, pixelX,
                                                                    pixelY, pixelZ));
}

template <typename T>
auto Imager<T>::nufft_synthesis(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
                                std::size_t nLevel, ConstView<BippFilter, 1> filter,
                                ConstView<T, 1> pixelX, ConstView<T, 1> pixelY,
                                ConstView<T, 1> pixelZ) -> Imager<T> {
  assert(ctx);
  return Imager(ctx, SynthesisFactory<T>::create_nufft_synthesis(ctx, std::move(opt), nLevel,
                                                                 filter, pixelX, pixelY, pixelZ));
}

#ifdef BIPP_MPI
template <typename T>
auto Imager<T>::distributed_standard_synthesis(std::shared_ptr<CommunicatorInternal> comm,
                                             std::shared_ptr<ContextInternal> ctx,
                                             std::size_t nLevel, ConstView<BippFilter, 1> filter,
                                             ConstView<T, 1> pixelX, ConstView<T, 1> pixelY,
                                             ConstView<T, 1> pixelZ) -> Imager<T> {
  assert(comm);
  assert(ctx);
  return Imager(ctx, SynthesisFactory<T>::create_distributed_standard_synthesis(
      std::move(comm), ctx, nLevel, filter, pixelX, pixelY, pixelZ));
}

template <typename T>
auto Imager<T>::distributed_nufft_synthesis(std::shared_ptr<CommunicatorInternal> comm,
                                          std::shared_ptr<ContextInternal> ctx,
                                          NufftSynthesisOptions opt, std::size_t nLevel,
                                          ConstView<BippFilter, 1> filter, ConstView<T, 1> pixelX,
                                          ConstView<T, 1> pixelY, ConstView<T, 1> pixelZ) -> Imager<T> {
  assert(comm);
  assert(ctx);
  return Imager(ctx,
                SynthesisFactory<T>::create_distributed_nufft_synthesis(
                    std::move(comm), ctx, std::move(opt), nLevel, filter, pixelX, pixelY, pixelZ));
}
#endif

template <typename T>
Imager<T>::Imager(std::shared_ptr<ContextInternal> ctx, std::unique_ptr<SynthesisInterface<T>> syn)
    : synthesis_(std::move(syn)) {
  if (ctx->processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    collector_ = std::make_unique<gpu::Collector<T>>(std::move(ctx));
#else
    throw GPUSupportError();
#endif
  } else {
    collector_ = std::make_unique<host::Collector<T>>(std::move(ctx));
  }
}

template <typename T>
auto Imager<T>::collect(T wl, const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc,
                        ConstView<std::complex<T>, 2> s, ConstView<std::complex<T>, 2> w,
                        ConstView<T, 2> xyz, ConstView<T, 2> uvw) -> void {
  if (!synthesis_) throw InternalError();

  const auto nAntenna = w.shape(0);
  const auto nBeam = w.shape(1);
  const auto nLevel = synthesis_->image().shape(1);

  auto& ctx = *synthesis_->context();

  auto t =
      ctx.logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO, pointer_to_string(this) + " collect");

  if (synthesis_->gpu_enabled()) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    gpu::Queue& queue = ctx.gpu_queue();
    // Syncronize with default stream.
    queue.sync_with_stream(nullptr);
    // syncronize with stream to be synchronous with host before exiting
    auto syncGuard = queue.sync_guard();

    ConstDeviceAccessor<gpu::api::ComplexType<T>, 2> sDevice(
        queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(s.data()), s.shape(), s.strides());

    ConstDeviceAccessor<gpu::api::ComplexType<T>, 2> wDevice(
        queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(w.data()), w.shape(), w.strides());

    ConstDeviceAccessor<T, 2> uvwDevice(queue, uvw);

    auto vUnbeamArray = queue.create_device_array<gpu::api::ComplexType<T>, 2>({nAntenna, nBeam});
    auto dArray = queue.create_device_array<T, 1>(nBeam);

    // Center coordinates for much better performance of cos / sin
    auto xyzCentered = queue.create_device_array<T, 2>(xyz.shape());
    copy(queue, xyz, xyzCentered);

    for (std::size_t i = 0; i < xyzCentered.shape(1); ++i) {
      center_vector<T>(queue, nAntenna, xyzCentered.slice_view(i).data());
    }

    const auto nEig =
        gpu::eigh<T>(ctx, wl, sDevice.view(), wDevice.view(), xyzCentered, dArray, vUnbeamArray);

    auto d = dArray.sub_view(0, nEig);
    auto vUnbeam = vUnbeamArray.sub_view({0, 0}, {nAntenna, nEig});
    auto vUnbeamCast =
        ConstView<std::complex<T>, 2>(reinterpret_cast<const std::complex<T>*>(vUnbeam.data()),
                                      vUnbeam.shape(), vUnbeam.strides());

    auto dHostArray = queue.create_pinned_array<T, 1>(nEig);

    copy(queue, d, dHostArray);
    queue.sync();  // make sure d is on host

    auto dMaskedArray = HostArray<T, 2>(ctx.host_alloc(), {d.size(), nLevel});

    for (std::size_t idxLevel = 0; idxLevel < nLevel; ++idxLevel) {
      copy(dHostArray, dMaskedArray.slice_view(idxLevel));
      eigMaskFunc(idxLevel, nEig, dMaskedArray.slice_view(idxLevel).data());
    }

    collector_->collect(
        wl, vUnbeamCast, dMaskedArray,
        synthesis_->type() == SynthesisType::Standard ? xyzCentered : uvwDevice.view());

    if (collector_->size() >= 10) {
      synthesis_->process(*collector_);
      collector_->clear();
    }
#else
    throw GPUSupportError();
#endif

  } else {
    auto sHost = ConstHostView<std::complex<T>, 2>(s);
    auto wHost = ConstHostView<std::complex<T>, 2>(w);
    auto xyzHost = ConstHostView<T, 2>(xyz);
    auto uvwHost = ConstHostView<T, 2>(uvw);

    auto vUnbeamArray = HostArray<std::complex<T>, 2>(ctx.host_alloc(), {nAntenna, nBeam});

    auto dArray = HostArray<T, 1>(ctx.host_alloc(), nBeam);

    // Center coordinates for much better performance of cos / sin in standard synthesis
    auto xyzCentered = HostArray<T, 2>();
    if (synthesis_->type() == SynthesisType::Standard) {
      xyzCentered = HostArray<T, 2>(ctx.host_alloc(), {nAntenna, 3});
      center_vector(nAntenna, xyzHost.slice_view(0).data(), xyzCentered.data());
      center_vector(nAntenna, xyzHost.slice_view(1).data(), xyzCentered.slice_view(1).data());
      center_vector(nAntenna, xyzHost.slice_view(2).data(), xyzCentered.slice_view(2).data());

      xyzHost = xyzCentered;
    }

    const auto nEig = host::eigh<T>(ctx, wl, sHost, wHost, xyzHost, dArray, vUnbeamArray);

    auto d = dArray.sub_view(0, nEig);
    auto vUnbeam = vUnbeamArray.sub_view({0, 0}, {nAntenna, nEig});

    auto dMaskedArray = HostArray<T, 2>(ctx.host_alloc(), {d.size(), nLevel});

    for (std::size_t idxLevel = 0; idxLevel < nLevel; ++idxLevel) {
      copy(d, dMaskedArray.slice_view(idxLevel));
      eigMaskFunc(idxLevel, nEig, dMaskedArray.slice_view(idxLevel).data());
    }

    collector_->collect(wl, vUnbeam, dMaskedArray,
                        synthesis_->type() == SynthesisType::Standard ? xyzHost : uvwHost);

    if (collector_->size() >= 10) {
      synthesis_->process(*collector_);
      collector_->clear();
    }
  }
}

template <typename T>
auto Imager<T>::get(BippFilter f, T* out, std::size_t ld) -> void {
  if(collector_->size()) {
    synthesis_->process(*collector_);
    collector_->clear();
  }


  auto img = synthesis_->image();
  auto& ctx = *synthesis_->context();
  auto t =
      ctx.logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO, pointer_to_string(this) + " get");

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  if (synthesis_->gpu_enabled()) {
    auto& queue = ctx.gpu_queue();
    // Syncronize with default stream.
    queue.sync_with_stream(nullptr);
  }
#endif

  synthesis_->get(f, View<T, 2>(out, {img.shape(0), img.shape(1)}, {1, ld}));

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  if (synthesis_->gpu_enabled()) {
    gpu::Queue& queue = synthesis_->context()->gpu_queue();
    queue.sync();
  }
#endif
}

template class Imager<float>;
template class Imager<double>;

}  // namespace bipp
