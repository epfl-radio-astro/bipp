#include "gpu/nufft_synthesis.hpp"

#include <complex>
#include <cstring>
#include <functional>
#include <memory>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "gpu/eigensolver.hpp"
#include "gpu/gram_matrix.hpp"
#include "gpu/kernels/add_vector.hpp"
#include "gpu/kernels/min_max_element.hpp"
#include "gpu/nufft_3d3.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#include "gpu/virtual_vis.hpp"
#include "nufft_util.hpp"

namespace bipp {
namespace gpu {

template <typename T>
NufftSynthesis<T>::NufftSynthesis(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
                                  std::size_t nAntenna, std::size_t nBeam, std::size_t nIntervals,
                                  std::size_t nFilter, const BippFilter* filterHost,
                                  std::size_t nPixel, const T* lmnX, const T* lmnY, const T* lmnZ)
    : ctx_(std::move(ctx)),
      opt_(std::move(opt)),
      nIntervals_(nIntervals),
      nFilter_(nFilter),
      nPixel_(nPixel),
      nAntenna_(nAntenna),
      nBeam_(nBeam),
      imgPartition_(DomainPartition::none(ctx_, nPixel)),
      collectCount_(0) {
  auto& queue = ctx_->gpu_queue();
  filterHost_ = queue.create_host_buffer<BippFilter>(nFilter_);
  std::memcpy(filterHost_.get(), filterHost, sizeof(BippFilter) * nFilter_);

  std::visit(
      [&](auto&& arg) -> void {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
          imgPartition_ =
              DomainPartition::grid<T>(ctx_, arg.dimensions, nPixel_, {lmnX, lmnY, lmnZ});
        }
      },
      opt_.localImagePartition.method);

  lmnX_ = queue.create_device_buffer<T>(nPixel_);
  lmnY_ = queue.create_device_buffer<T>(nPixel_);
  lmnZ_ = queue.create_device_buffer<T>(nPixel_);

  imgPartition_.apply(lmnX, lmnX_.get());
  imgPartition_.apply(lmnY, lmnY_.get());
  imgPartition_.apply(lmnZ, lmnZ_.get());

  if (opt_.collectGroupSize && opt_.collectGroupSize.value() > 0) {
    nMaxInputCount_ = opt_.collectGroupSize.value();
  } else {
    // use at most 25% of memory more accumulation, but not more than 200
    // iterations.
    std::size_t freeMem, totalMem;
    api::mem_get_info(&freeMem, &totalMem);
    nMaxInputCount_ =
        (totalMem / 4) / (nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ * sizeof(std::complex<T>));
    nMaxInputCount_ = std::min<std::size_t>(std::max<std::size_t>(1, nMaxInputCount_), 200);
  }

  const auto virtualVisBufferSize =
      nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ * nMaxInputCount_;
  virtualVis_ = queue.create_device_buffer<api::ComplexType<T>>(virtualVisBufferSize);
  uvwX_ = queue.create_device_buffer<T>(nAntenna_ * nAntenna_ * nMaxInputCount_);
  uvwY_ = queue.create_device_buffer<T>(nAntenna_ * nAntenna_ * nMaxInputCount_);
  uvwZ_ = queue.create_device_buffer<T>(nAntenna_ * nAntenna_ * nMaxInputCount_);

  img_ = queue.create_device_buffer<T>(nPixel_ * nIntervals_ * nFilter_);
  api::memset_async(img_.get(), 0, nPixel_ * nIntervals_ * nFilter_ * sizeof(T), queue.stream());
}

template <typename T>
auto NufftSynthesis<T>::collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
                                const api::ComplexType<T>* s, std::size_t lds,
                                const api::ComplexType<T>* w, std::size_t ldw, const T* xyz,
                                std::size_t ldxyz, const T* uvw, std::size_t lduvw) -> void {
  auto& queue = ctx_->gpu_queue();

  // store coordinates
  api::memcpy_async(uvwX_.get() + collectCount_ * nAntenna_ * nAntenna_, uvw,
                    sizeof(T) * nAntenna_ * nAntenna_, api::flag::MemcpyDeviceToDevice,
                    queue.stream());
  api::memcpy_async(uvwY_.get() + collectCount_ * nAntenna_ * nAntenna_, uvw + lduvw,
                    sizeof(T) * nAntenna_ * nAntenna_, api::flag::MemcpyDeviceToDevice,
                    queue.stream());
  api::memcpy_async(uvwZ_.get() + collectCount_ * nAntenna_ * nAntenna_, uvw + 2 * lduvw,
                    sizeof(T) * nAntenna_ * nAntenna_, api::flag::MemcpyDeviceToDevice,
                    queue.stream());

  auto v = queue.create_device_buffer<api::ComplexType<T>>(nBeam_ * nEig);
  auto d = queue.create_device_buffer<T>(nEig);

  {
    auto g = queue.create_device_buffer<api::ComplexType<T>>(nBeam_ * nBeam_);

    gram_matrix<T>(*ctx_, nAntenna_, nBeam_, w, ldw, xyz, ldxyz, wl, g.get(), nBeam_);

    std::size_t nEigOut = 0;
    // Note different order of s and g input
    if (s)
      eigh<T>(*ctx_, nBeam_, nEig, s, lds, g.get(), nBeam_, &nEigOut, d.get(), v.get(), nBeam_);
    else
      eigh<T>(*ctx_, nBeam_, nEig, g.get(), nBeam_, nullptr, 0, &nEigOut, d.get(), v.get(), nBeam_);
  }

  auto virtVisPtr = virtualVis_.get() + collectCount_ * nAntenna_ * nAntenna_;

  virtual_vis(*ctx_, nFilter_, filterHost_.get(), nIntervals_, intervals, ldIntervals, nEig,
              d.get(), nAntenna_, v.get(), nBeam_, nBeam_, w, ldw, virtVisPtr,
              nMaxInputCount_ * nIntervals_ * nAntenna_ * nAntenna_,
              nMaxInputCount_ * nAntenna_ * nAntenna_, nAntenna_);

  ++collectCount_;
  if (collectCount_ >= nMaxInputCount_) {
    computeNufft();
  }
}

template <typename T>
auto NufftSynthesis<T>::computeNufft() -> void {
  auto& queue = ctx_->gpu_queue();

  if (collectCount_) {
    auto output = queue.create_device_buffer<api::ComplexType<T>>(nPixel_);
    auto outputPtr = output.get();

    const auto nInputPoints = nAntenna_ * nAntenna_ * collectCount_;

    auto inputPartition = std::visit(
        [&](auto&& arg) -> DomainPartition {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
            return DomainPartition::grid<T>(ctx_, arg.dimensions, nInputPoints,
                                            {uvwX_.get(), uvwY_.get(), uvwZ_.get()});

          } else if constexpr (std::is_same_v<ArgType, Partition::None>) {
            return DomainPartition::none(ctx_, nInputPoints);

          } else if constexpr (std::is_same_v<ArgType, Partition::Auto>) {
            auto minMaxBuffer = queue.create_device_buffer<T>(12);
            min_element(queue, nInputPoints, uvwX_.get(), minMaxBuffer.get());
            max_element(queue, nInputPoints, uvwX_.get(), minMaxBuffer.get() + 1);

            min_element(queue, nInputPoints, uvwY_.get(), minMaxBuffer.get() + 2);
            max_element(queue, nInputPoints, uvwY_.get(), minMaxBuffer.get() + 3);

            min_element(queue, nInputPoints, uvwZ_.get(), minMaxBuffer.get() + 4);
            max_element(queue, nInputPoints, uvwZ_.get(), minMaxBuffer.get() + 5);

            min_element(queue, nPixel_, lmnX_.get(), minMaxBuffer.get() + 6);
            max_element(queue, nPixel_, lmnX_.get(), minMaxBuffer.get() + 7);

            min_element(queue, nPixel_, lmnY_.get(), minMaxBuffer.get() + 8);
            max_element(queue, nPixel_, lmnY_.get(), minMaxBuffer.get() + 9);

            min_element(queue, nPixel_, lmnZ_.get(), minMaxBuffer.get() + 10);
            max_element(queue, nPixel_, lmnZ_.get(), minMaxBuffer.get() + 11);

            auto minMaxHostBuffer = queue.create_host_buffer<T>(minMaxBuffer.size());

            api::memcpy_async(minMaxHostBuffer.get(), minMaxBuffer.get(),
                              minMaxBuffer.size_in_bytes(), api::flag::MemcpyDeviceToHost,
                              queue.stream());

            queue.sync();

            const T* minMaxPtr = minMaxHostBuffer.get();

            std::array<double, 3> uvwExtent = {minMaxPtr[1] - minMaxPtr[0],
                                               minMaxPtr[3] - minMaxPtr[2],
                                               minMaxPtr[5] - minMaxPtr[4]};
            std::array<double, 3> imgExtent = {minMaxPtr[7] - minMaxPtr[6],
                                               minMaxPtr[9] - minMaxPtr[8],
                                               minMaxPtr[11] - minMaxPtr[10]};

            // Use at most 25% of total memory for fft grid
            const auto gridSize = optimal_nufft_input_partition(
                uvwExtent, imgExtent,
                queue.device_prop().totalGlobalMem / (4 * sizeof(api::ComplexType<T>)));

            // set partition method to grid and create grid partition
            opt_.localUVWPartition.method = Partition::Grid{gridSize};
            return DomainPartition::grid<T>(ctx_, gridSize, nInputPoints,
                                            {uvwX_.get(), uvwY_.get(), uvwZ_.get()});
          }
        },
        opt_.localUVWPartition.method);

    const auto ldVirtVis3 = nAntenna_;
    const auto ldVirtVis2 = nMaxInputCount_ * nAntenna_ * ldVirtVis3;
    const auto ldVirtVis1 = nIntervals_ * ldVirtVis2;

    for (std::size_t i = 0; i < nFilter_; ++i) {
      for (std::size_t j = 0; j < nIntervals_; ++j) {
        inputPartition.apply(virtualVis_.get() + i * ldVirtVis1 + j * ldVirtVis2);
      }
    }

    inputPartition.apply(uvwX_.get());
    inputPartition.apply(uvwY_.get());
    inputPartition.apply(uvwZ_.get());

    queue.signal_stream(nullptr);  // cufinufft uses default stream

    for (const auto& [inputBegin, inputSize] : inputPartition.groups()) {
      if (!inputSize) continue;
      for (const auto& [imgBegin, imgSize] : imgPartition_.groups()) {
        if (!imgSize) continue;

        Nufft3d3<T> transform(1, opt_.tolerance, 1, inputSize, uvwX_.get() + inputBegin,
                              uvwY_.get() + inputBegin, uvwZ_.get() + inputBegin, imgSize,
                              lmnX_.get() + imgBegin, lmnY_.get() + imgBegin,
                              lmnZ_.get() + imgBegin);

        for (std::size_t i = 0; i < nFilter_; ++i) {
          for (std::size_t j = 0; j < nIntervals_; ++j) {
            auto imgPtr = img_.get() + (j + i * nIntervals_) * nPixel_ + imgBegin;
            transform.execute(virtualVis_.get() + i * ldVirtVis1 + j * ldVirtVis2 + inputBegin,
                              outputPtr);

            // add dependency on default stream for correct ordering
            queue.sync_with_stream(nullptr);
            add_vector_real_to_complex<T>(queue, imgSize, outputPtr, imgPtr);
            queue.signal_stream(nullptr);
          }
        }
      }
    }
  }

  collectCount_ = 0;
}

template <typename T>
auto NufftSynthesis<T>::get(BippFilter f, T* outHostOrDevice, std::size_t ld) -> void {
  computeNufft();  // make sure all input has been processed

  auto& queue = ctx_->gpu_queue();
  std::size_t index = nFilter_;
  const BippFilter* filterPtr = filterHost_.get();
  for (std::size_t i = 0; i < nFilter_; ++i) {
    if (filterPtr[i] == f) {
      index = i;
      break;
    }
  }
  if (index == nFilter_) throw InvalidParameterError();

  if (is_device_ptr(outHostOrDevice)) {
    for (std::size_t i = 0; i < nIntervals_; ++i) {
      imgPartition_.reverse(img_.get() + index * nIntervals_ * nPixel_ + i * nPixel_,
                            outHostOrDevice + i * ld);
    }
  } else {
    auto workBuffer = queue.create_device_buffer<T>(nPixel_);
    for (std::size_t i = 0; i < nIntervals_; ++i) {
      imgPartition_.reverse(img_.get() + index * nIntervals_ * nPixel_ + i * nPixel_,
                            workBuffer.get());
      gpu::api::memcpy_async(outHostOrDevice + i * ld, workBuffer.get(), workBuffer.size_in_bytes(),
                             gpu::api::flag::MemcpyDefault, queue.stream());
    }
  }
}

template class NufftSynthesis<float>;
template class NufftSynthesis<double>;

}  // namespace gpu
}  // namespace bipp
