#include "host/kernels/gemmexp.hpp"

#include <cmath>
#include <complex>
#include <vector>

#include "host/omp_definitions.hpp"

#ifdef BIPP_VC
#include <Vc/Vc>
#endif

namespace bipp {
namespace host {

template <typename T>
auto gemmexp(std::size_t nEig, std::size_t nPixel, std::size_t nAntenna, T alpha,
             const std::complex<T>* __restrict__ vUnbeam, std::size_t ldv,
             const T* __restrict__ xyz, std::size_t ldxyz, const T* __restrict__ pixelX,
             const T* __restrict__ pixelY, const T* __restrict__ pixelZ, T* __restrict__ out,
             std::size_t ldout) -> void {

  BIPP_OMP_PRAGMA("omp parallel") {
    std::vector<std::complex<T> > pixSumVec(nEig);

    BIPP_OMP_PRAGMA("omp for schedule(static)")
    for (std::size_t idxPix = 0; idxPix < nPixel; ++idxPix) {
      const auto pX = pixelX[idxPix];
      const auto pY = pixelY[idxPix];
      const auto pZ = pixelZ[idxPix];
      for (std::size_t idxAnt = 0; idxAnt < nAntenna; ++idxAnt) {
        const auto imag =
            alpha * (pX * xyz[idxAnt] + pY * xyz[idxAnt + ldxyz] + pZ * xyz[idxAnt + 2 * ldxyz]);
        const std::complex<T> ie{std::cos(imag), std::sin(imag)};
        for (std::size_t idxEig = 0; idxEig < nEig; ++idxEig) {
          pixSumVec[idxEig] += vUnbeam[idxEig * ldv + idxAnt] * ie;
        }
      }

      for (std::size_t idxEig = 0; idxEig < nEig; ++idxEig) {
        const auto pv = pixSumVec[idxEig];
        pixSumVec[idxEig] = 0;
        out[idxEig * ldout + idxPix] = pv.real() * pv.real() + pv.imag() * pv.imag();
      }
    }
  }
}

template auto gemmexp<float>(std::size_t nEig, std::size_t nPixel, std::size_t nAntenna,
                             float alpha, const std::complex<float>* __restrict__ vUnbeam,
                             std::size_t ldv, const float* __restrict__ xyz, std::size_t ldxyz,
                             const float* __restrict__ pixelX, const float* __restrict__ pixelY,
                             const float* __restrict__ pixelZ, float* __restrict__ out,
                             std::size_t ldout) -> void;

template auto gemmexp<double>(std::size_t nEig, std::size_t nPixel, std::size_t nAntenna,
                              double alpha, const std::complex<double>* __restrict__ vUnbeam,
                              std::size_t ldv, const double* __restrict__ xyz, std::size_t ldxyz,
                              const double* __restrict__ pixelX, const double* __restrict__ pixelY,
                              const double* __restrict__ pixelZ, double* __restrict__ out,
                              std::size_t ldout) -> void;
}  // namespace host
}  // namespace bipp
