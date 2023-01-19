#include "host/nufft_synthesis.hpp"

#include <unistd.h>

#include <algorithm>
#include <complex>
#include <cstring>
#include <functional>
#include <memory>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "host/eigensolver.hpp"
#include "host/gram_matrix.hpp"
#include "host/nufft_3d3.hpp"
#include "host/virtual_vis.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace host {

static auto system_memory() {
  auto pages = sysconf(_SC_PHYS_PAGES);
  auto pageSize = sysconf(_SC_PAGE_SIZE);
  auto memory = pages * pageSize;
  return memory > 0 ? memory : 1024;
}

template <typename T>
NufftSynthesis<T>::NufftSynthesis(std::shared_ptr<ContextInternal> ctx, T tol, std::size_t nAntenna,
                                  std::size_t nBeam, std::size_t nIntervals, std::size_t nFilter,
                                  const BippFilter* filter, std::size_t nPixel, const T* lmnX,
                                  const T* lmnY, const T* lmnZ)
    : ctx_(std::move(ctx)),
      tol_(tol),
      nIntervals_(nIntervals),
      nFilter_(nFilter),
      nPixel_(nPixel),
      nAntenna_(nAntenna),
      nBeam_(nBeam),
      inputCount_(0) {
  filter_ = Buffer<BippFilter>(ctx_->host_alloc(), nFilter_);
  std::memcpy(filter_.get(), filter, sizeof(BippFilter) * nFilter_);
  lmnX_ = Buffer<T>(ctx_->host_alloc(), nPixel_);
  std::memcpy(lmnX_.get(), lmnX, sizeof(T) * nPixel_);
  lmnY_ = Buffer<T>(ctx_->host_alloc(), nPixel_);
  std::memcpy(lmnY_.get(), lmnY, sizeof(T) * nPixel_);
  lmnZ_ = Buffer<T>(ctx_->host_alloc(), nPixel_);
  std::memcpy(lmnZ_.get(), lmnZ, sizeof(T) * nPixel_);

  // use at most 33% of memory more accumulation, but not more than 200
  // iterations. TODO: find optimum
  nMaxInputCount_ = (system_memory() / 3) /
                    (nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ * sizeof(std::complex<T>));
  nMaxInputCount_ = std::min<std::size_t>(std::max<std::size_t>(1, nMaxInputCount_), 200);

  const auto virtualVisBufferSize =
      nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ * nMaxInputCount_;
  virtualVis_ = Buffer<std::complex<T>>(ctx_->host_alloc(), virtualVisBufferSize);
  uvwX_ = Buffer<T>(ctx_->host_alloc(), nAntenna_ * nAntenna_ * nMaxInputCount_);
  uvwY_ = Buffer<T>(ctx_->host_alloc(), nAntenna_ * nAntenna_ * nMaxInputCount_);
  uvwZ_ = Buffer<T>(ctx_->host_alloc(), nAntenna_ * nAntenna_ * nMaxInputCount_);

  img_ = Buffer<T>(ctx_->host_alloc(), nPixel_ * nIntervals_ * nFilter_);
  std::memset(img_.get(), 0, nPixel_ * nIntervals_ * nFilter_ * sizeof(T));
}

template <typename T>
auto NufftSynthesis<T>::collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
                                const std::complex<T>* s, std::size_t lds, const std::complex<T>* w,
                                std::size_t ldw, const T* xyz, std::size_t ldxyz, const T* uvw,
                                std::size_t lduvw) -> void {
  // store coordinates
  std::memcpy(uvwX_.get() + inputCount_ * nAntenna_ * nAntenna_, uvw,
              sizeof(T) * nAntenna_ * nAntenna_);
  std::memcpy(uvwY_.get() + inputCount_ * nAntenna_ * nAntenna_, uvw + lduvw,
              sizeof(T) * nAntenna_ * nAntenna_);
  std::memcpy(uvwZ_.get() + inputCount_ * nAntenna_ * nAntenna_, uvw + 2 * lduvw,
              sizeof(T) * nAntenna_ * nAntenna_);

  auto v = Buffer<std::complex<T>>(ctx_->host_alloc(), nBeam_ * nEig);
  auto d = Buffer<T>(ctx_->host_alloc(), nEig);

  {
    auto g = Buffer<std::complex<T>>(ctx_->host_alloc(), nBeam_ * nBeam_);

    gram_matrix<T>(*ctx_, nAntenna_, nBeam_, w, ldw, xyz, ldxyz, wl, g.get(), nBeam_);

    std::size_t nEigOut = 0;
    // Note different order of s and g input
    if (s)
      eigh<T>(*ctx_, nBeam_, nEig, s, lds, g.get(), nBeam_, &nEigOut, d.get(), v.get(), nBeam_);
    else {
      eigh<T>(*ctx_, nBeam_, nEig, g.get(), nBeam_, nullptr, 0, &nEigOut, d.get(), v.get(), nBeam_);
    }
  }

  auto virtVisPtr = virtualVis_.get() + inputCount_ * nAntenna_ * nAntenna_;

  virtual_vis(*ctx_, nFilter_, filter_.get(), nIntervals_, intervals, ldIntervals, nEig, d.get(),
              nAntenna_, v.get(), nBeam_, nBeam_, w, ldw, virtVisPtr,
              nMaxInputCount_ * nIntervals_ * nAntenna_ * nAntenna_,
              nMaxInputCount_ * nAntenna_ * nAntenna_, nAntenna_);

  ++inputCount_;
  if (inputCount_ >= nMaxInputCount_) {
    computeNufft();
  }
}

template <typename T>
auto NufftSynthesis<T>::computeNufft() -> void {
  if (inputCount_) {
    auto output = Buffer<std::complex<T>>(ctx_->host_alloc(), nPixel_);
    auto outputPtr = output.get();
    Nufft3d3<T> transform(1, tol_, 1, nAntenna_ * nAntenna_ * inputCount_, uvwX_.get(), uvwY_.get(),
                          uvwZ_.get(), nPixel_, lmnX_.get(), lmnY_.get(), lmnZ_.get());

    const auto ldVirtVis3 = nAntenna_;
    const auto ldVirtVis2 = nMaxInputCount_ * nAntenna_ * ldVirtVis3;
    const auto ldVirtVis1 = nIntervals_ * ldVirtVis2;

    for (std::size_t i = 0; i < nFilter_; ++i) {
      for (std::size_t j = 0; j < nIntervals_; ++j) {
        auto imgPtr = img_.get() + (j + i * nIntervals_) * nPixel_;
        transform.execute(virtualVis_.get() + i * ldVirtVis1 + j * ldVirtVis2, outputPtr);

        for (std::size_t k = 0; k < nPixel_; ++k) {
          imgPtr[k] += outputPtr[k].real();
        }
      }
    }
  }

  inputCount_ = 0;
}

template <typename T>
auto NufftSynthesis<T>::get(BippFilter f, T* out, std::size_t ld) -> void {
  computeNufft();  // make sure all input has been processed

  std::size_t index = nFilter_;
  const BippFilter* filterPtr = filter_.get();
  for (std::size_t i = 0; i < nFilter_; ++i) {
    if (filterPtr[i] == f) {
      index = i;
      break;
    }
  }
  if (index == nFilter_) throw InvalidParameterError();

  for (std::size_t i = 0; i < nIntervals_; ++i) {
    std::memcpy(out + i * ld, img_.get() + index * nIntervals_ * nPixel_ + i * nPixel_,
                sizeof(T) * nPixel_);
  }
}

template class NufftSynthesis<float>;
template class NufftSynthesis<double>;

}  // namespace host
}  // namespace bipp
