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
#ifdef BIPP_VC

  using simdType = Vc::Vector<T>;
  constexpr std::size_t simdSize = simdType::size();

  const simdType alphaVec = alpha;
  const typename simdType::IndexType indexComplex([](auto&& n) { return 2 * n; });

  BIPP_OMP_PRAGMA("omp parallel for schedule(static)")
  for (std::size_t idxPix = 0; idxPix < nPixel; ++idxPix) {
    const simdType pX = pixelX[idxPix];
    const simdType pY = pixelY[idxPix];
    const simdType pZ = pixelZ[idxPix];

    for (std::size_t idxEig = 0; idxEig < nEig; ++idxEig) {
      simdType sumReal = 0;
      simdType sumImag = 0;

      std::size_t idxAnt = 0;
      for (; idxAnt + simdSize <= nAntenna; idxAnt += simdSize) {
        simdType x(xyz + idxAnt, Vc::Unaligned);
        simdType y(xyz + ldxyz + idxAnt, Vc::Unaligned);
        simdType z(xyz + 2 * ldxyz + idxAnt, Vc::Unaligned);

        const auto imag = alphaVec * Vc::fma(pX, x, Vc::fma(pY, y, pZ * z));

        simdType cosValue, sinValue;
        Vc::sincos(imag, &sinValue, &cosValue);

        const T* vUnbeamScalarPtr = reinterpret_cast<const T*>(vUnbeam + idxEig * ldv + idxAnt);
        simdType vValueReal(vUnbeamScalarPtr, indexComplex);
        simdType vValueImag(vUnbeamScalarPtr + 1, indexComplex);

        sumReal += vValueReal * cosValue - vValueImag * sinValue;
        sumImag += Vc::fma(vValueReal, sinValue, vValueImag * cosValue);
      }

      const auto tail = nAntenna - idxAnt;
      if (tail) {
        simdType x, y, z;
        x.setZero();
        y.setZero();
        z.setZero();
        for (std::size_t i = 0; i < tail; ++i) {
          x[i] = xyz[idxAnt + i];
          y[i] = xyz[idxAnt + ldxyz + i];
          z[i] = xyz[idxAnt + 2 * ldxyz + i];
        }
        const auto imag = alphaVec * Vc::fma(pX, x, Vc::fma(pY, y, pZ * z));

        simdType cosValue, sinValue;
        Vc::sincos(imag, &sinValue, &cosValue);

        simdType vValueReal;
        simdType vValueImag;
        for (std::size_t i = 0; i < tail; ++i) {
          const auto vValue = vUnbeam[idxEig * ldv + idxAnt + i];
          vValueReal[i] = vValue.real();
          vValueImag[i] = vValue.imag();
        }
        auto tailSumReal = vValueReal * cosValue - vValueImag * sinValue;
        auto tailSumImag = Vc::fma(vValueReal, sinValue, vValueImag * cosValue);

        for (std::size_t i = 0; i < tail; ++i) {
          sumReal[i] += tailSumReal[i];
          sumImag[i] += tailSumImag[i];
        }
      }

      const T sumRealScalar = sumReal.sum();
      const T sumImagScalar = sumImag.sum();
      out[idxEig * ldout + idxPix] = sumRealScalar * sumRealScalar + sumImagScalar * sumImagScalar;
    }
  }

#else

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

#endif
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
