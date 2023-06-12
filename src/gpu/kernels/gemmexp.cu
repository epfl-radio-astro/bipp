#include <algorithm>

#include "bipp/config.h"
#include "gpu/kernels/gemmexp.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

static __device__ __forceinline__ void calc_sincos(float x, float* sptr, float* cptr) {
  sincosf(x, sptr, cptr);
}

static __device__ __forceinline__ void calc_sincos(double x, double* sptr, double* cptr) {
  sincos(x, sptr, cptr);
}

namespace {
template <typename T>
struct ComplexOp {
  ComplexOp() = default;
  __device__ __forceinline__ ComplexOp(T x, T y) : value{x, y} {}
  __device__ __forceinline__ ComplexOp(const api::ComplexType<T>& c) : value(c) {}

  __device__ __forceinline__ ComplexOp<T> operator-(const ComplexOp<T>& other) const {
    return ComplexOp{value.x - other.value.x, value.y - other.value.y};
  }

  __device__ __forceinline__ ComplexOp<T> operator+(const ComplexOp<T>& other) const {
    return ComplexOp{value.x + other.value.x, value.y + other.value.y};
  }

  __device__ __forceinline__ ComplexOp<T> operator*(const ComplexOp<T>& other) const {
    return ComplexOp{value.x * other.value.x - value.y * other.value.y,
                     value.x * other.value.y + other.value.x * value.y};
  }

  api::ComplexType<T> value;
};
}  // namespace

template <typename T, size_t BLOCK_THREADS>
static __global__ __launch_bounds__(BLOCK_THREADS) void gemmexp_kernel(
    size_t nEig, size_t nPixel, size_t nAntenna, T alpha,
    const api::ComplexType<T>* __restrict__ vUnbeam, size_t ldv, const T* __restrict__ xyz,
    size_t ldxyz, const T* __restrict__ pixelX, const T* __restrict__ pixelY,
    const T* __restrict__ pixelZ, T* __restrict__ out, size_t ldout) {
  __shared__ ComplexOp<T> tmpStorage[BLOCK_THREADS];

  for (size_t idxPix = blockIdx.x; idxPix < nPixel; idxPix += gridDim.x) {
    const auto pX = pixelX[idxPix];
    const auto pY = pixelY[idxPix];
    const auto pZ = pixelZ[idxPix];
    for (size_t idxEigStart = blockIdx.y * BLOCK_THREADS; idxEigStart < nEig;
         idxEigStart += gridDim.y * BLOCK_THREADS) {
      // each thread computes pixel value for different eigenvalue
      const size_t idxEig = idxEigStart + threadIdx.x;

      ComplexOp<T> localSum{0, 0};
      // iterate over antenna indices in blocks such that all threads always do the full loop count
      for (size_t idxAntStart = 0; idxAntStart < nAntenna; idxAntStart += BLOCK_THREADS) {
        // each thread reads a different value
        const size_t idxAnt = idxAntStart + threadIdx.x;
        if (idxAnt < nAntenna) {
          ComplexOp<T> sc;
          const auto imag =
              alpha * (pX * xyz[idxAnt] + pY * xyz[idxAnt + ldxyz] + pZ * xyz[idxAnt + 2 * ldxyz]);
          calc_sincos(imag, &(sc.value.y), &(sc.value.x));
          tmpStorage[threadIdx.x] = sc;
        }
        __syncthreads();

        if (idxEig < nEig) {
          for (size_t storageIdx = 0; storageIdx < min(BLOCK_THREADS, nAntenna - idxAntStart);
               ++storageIdx) {
            localSum =
                localSum + tmpStorage[storageIdx] *
                               ComplexOp<T>(vUnbeam[idxEig * ldv + idxAntStart + storageIdx]);
          }
        }
        __syncthreads();
      }

      if (idxEig < nEig) {
        out[idxEig * ldout + idxPix] =
            localSum.value.x * localSum.value.x + localSum.value.y * localSum.value.y;
      }
    }
  }
}

template <typename T, int BLOCK_THREADS>
static auto gemmexp_launch(Queue& q, std::size_t nEig, std::size_t nPixel, std::size_t nAntenna,
                           T alpha, const api::ComplexType<T>* vUnbeam, std::size_t ldv,
                           const T* xyz, std::size_t ldxyz, const T* pixelX, const T* pixelY,
                           const T* pixelZ, T* out, std::size_t ldout) -> void {
  const dim3 block(BLOCK_THREADS, 1, 1);

  const auto grid = kernel_launch_grid(q.device_prop(), {nPixel / 2, 1, 1}, block);

  api::launch_kernel(gemmexp_kernel<T, BLOCK_THREADS>, grid, block, 0, q.stream(), nEig, nPixel,
                     nAntenna, alpha, vUnbeam, ldv, xyz, ldxyz, pixelX, pixelY, pixelZ, out, ldout);
}

template <typename T>
auto gemmexp(Queue& q, std::size_t nEig, std::size_t nPixel, std::size_t nAntenna, T alpha,
             const api::ComplexType<T>* vUnbeam, std::size_t ldv, const T* xyz, std::size_t ldxyz,
             const T* pixelX, const T* pixelY, const T* pixelZ, T* out, std::size_t ldout) -> void {
  if (nEig >= 1024 && q.device_prop().maxThreadsDim[0] >= 1024) {
    gemmexp_launch<T, 1024>(q, nEig, nPixel, nAntenna, alpha, vUnbeam, ldv, xyz, ldxyz, pixelX,
                            pixelY, pixelZ, out, ldout);
  } else if (nEig >= 512 && q.device_prop().maxThreadsDim[0] >= 512) {
    gemmexp_launch<T, 512>(q, nEig, nPixel, nAntenna, alpha, vUnbeam, ldv, xyz, ldxyz, pixelX,
                           pixelY, pixelZ, out, ldout);
  } else if (nEig >= 256 && q.device_prop().maxThreadsDim[0] >= 512) {
    gemmexp_launch<T, 256>(q, nEig, nPixel, nAntenna, alpha, vUnbeam, ldv, xyz, ldxyz, pixelX,
                           pixelY, pixelZ, out, ldout);
  } else {
    gemmexp_launch<T, 128>(q, nEig, nPixel, nAntenna, alpha, vUnbeam, ldv, xyz, ldxyz, pixelX,
                           pixelY, pixelZ, out, ldout);
  }
}

template auto gemmexp<float>(Queue& q, std::size_t nEig, std::size_t nPixel, std::size_t nAntenna,
                             float alpha, const api::ComplexType<float>* __restrict__ vUnbeam,
                             std::size_t ldv, const float* __restrict__ xyz, std::size_t ldxyz,
                             const float* __restrict__ pixelX, const float* __restrict__ pixelY,
                             const float* __restrict__ pixelZ, float* __restrict__ out,
                             std::size_t ldout) -> void;

template auto gemmexp<double>(Queue& q, std::size_t nEig, std::size_t nPixel, std::size_t nAntenna,
                              double alpha, const api::ComplexType<double>* __restrict__ vUnbeam,
                              std::size_t ldv, const double* __restrict__ xyz, std::size_t ldxyz,
                              const double* __restrict__ pixelX, const double* __restrict__ pixelY,
                              const double* __restrict__ pixelZ, double* __restrict__ out,
                              std::size_t ldout) -> void;
}  // namespace gpu
}  // namespace bipp
