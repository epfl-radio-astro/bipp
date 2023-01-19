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
#include "gpu/nufft_3d3.hpp"
#include "gpu/util/runtime_api.hpp"
#include "gpu/virtual_vis.hpp"

namespace bipp {
namespace gpu {

template <typename T>
NufftSynthesis<T>::NufftSynthesis(std::shared_ptr<ContextInternal> ctx, T tol, std::size_t nAntenna,
                                  std::size_t nBeam, std::size_t nIntervals, std::size_t nFilter,
                                  const BippFilter* filterHost, std::size_t nPixel, const T* lmnX,
                                  const T* lmnY, const T* lmnZ)
    : ctx_(std::move(ctx)),
      tol_(tol),
      nIntervals_(nIntervals),
      nFilter_(nFilter),
      nPixel_(nPixel),
      nAntenna_(nAntenna),
      nBeam_(nBeam),
      inputCount_(0) {
  auto& queue = ctx_->gpu_queue();
  filterHost_ = queue.create_host_buffer<BippFilter>(nFilter_);
  std::memcpy(filterHost_.get(), filterHost, sizeof(BippFilter) * nFilter_);
  lmnX_ = queue.create_device_buffer<T>(nPixel_);
  api::memcpy_async(lmnX_.get(), lmnX, sizeof(T) * nPixel_, api::flag::MemcpyDeviceToDevice,
                    queue.stream());
  lmnY_ = queue.create_device_buffer<T>(nPixel_);
  api::memcpy_async(lmnY_.get(), lmnY, sizeof(T) * nPixel_, api::flag::MemcpyDeviceToDevice,
                    queue.stream());
  lmnZ_ = queue.create_device_buffer<T>(nPixel_);
  api::memcpy_async(lmnZ_.get(), lmnZ, sizeof(T) * nPixel_, api::flag::MemcpyDeviceToDevice,
                    queue.stream());

  // use at most 33% of memory more accumulation, but not more than 200
  // iterations. TODO: find optimum
  std::size_t freeMem, totalMem;
  api::mem_get_info(&freeMem, &totalMem);
  nMaxInputCount_ =
      (totalMem / 3) / (nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ * sizeof(std::complex<T>));
  nMaxInputCount_ = std::min<std::size_t>(std::max<std::size_t>(1, nMaxInputCount_), 200);

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
  api::memcpy_async(uvwX_.get() + inputCount_ * nAntenna_ * nAntenna_, uvw,
                    sizeof(T) * nAntenna_ * nAntenna_, api::flag::MemcpyDeviceToDevice,
                    queue.stream());
  api::memcpy_async(uvwY_.get() + inputCount_ * nAntenna_ * nAntenna_, uvw + lduvw,
                    sizeof(T) * nAntenna_ * nAntenna_, api::flag::MemcpyDeviceToDevice,
                    queue.stream());
  api::memcpy_async(uvwZ_.get() + inputCount_ * nAntenna_ * nAntenna_, uvw + 2 * lduvw,
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

  auto virtVisPtr = virtualVis_.get() + inputCount_ * nAntenna_ * nAntenna_;

  virtual_vis(*ctx_, nFilter_, filterHost_.get(), nIntervals_, intervals, ldIntervals, nEig,
              d.get(), nAntenna_, v.get(), nBeam_, nBeam_, w, ldw, virtVisPtr,
              nMaxInputCount_ * nIntervals_ * nAntenna_ * nAntenna_,
              nMaxInputCount_ * nAntenna_ * nAntenna_, nAntenna_);

  ++inputCount_;
  if (inputCount_ >= nMaxInputCount_) {
    computeNufft();
  }
}

template <typename T>
auto NufftSynthesis<T>::computeNufft() -> void {
  auto& queue = ctx_->gpu_queue();

  if (inputCount_) {
    auto output = queue.create_device_buffer<api::ComplexType<T>>(nPixel_);
    auto outputPtr = output.get();
    queue.sync();  // cufinufft cannot be asigned a stream
    Nufft3d3<T> transform(1, tol_, 1, nAntenna_ * nAntenna_ * inputCount_, uvwX_.get(), uvwY_.get(),
                          uvwZ_.get(), nPixel_, lmnX_.get(), lmnY_.get(), lmnZ_.get());

    const auto ldVirtVis3 = nAntenna_;
    const auto ldVirtVis2 = nMaxInputCount_ * nAntenna_ * ldVirtVis3;
    const auto ldVirtVis1 = nIntervals_ * ldVirtVis2;

    for (std::size_t i = 0; i < nFilter_; ++i) {
      for (std::size_t j = 0; j < nIntervals_; ++j) {
        auto imgPtr = img_.get() + (j + i * nIntervals_) * nPixel_;
        transform.execute(virtualVis_.get() + i * ldVirtVis1 + j * ldVirtVis2, outputPtr);

        // use default stream to match cufiNUFFT
        queue.sync_with_stream(nullptr);
        add_vector_real_to_complex<T>(queue, nPixel_, outputPtr, imgPtr);
        queue.signal_stream(nullptr);
      }
    }
  }

  api::stream_synchronize(nullptr);  // cufinufft cannot be asigned a stream
  inputCount_ = 0;
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

  api::memcpy_2d_async(outHostOrDevice, ld * sizeof(T), img_.get() + index * nIntervals_ * nPixel_,
                       nPixel_ * sizeof(T), nPixel_ * sizeof(T), nIntervals_,
                       api::flag::MemcpyDefault, queue.stream());
}

template class NufftSynthesis<float>;
template class NufftSynthesis<double>;

}  // namespace gpu
}  // namespace bipp
