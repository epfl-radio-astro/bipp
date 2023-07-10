#include <algorithm>
#include <cstddef>

#include "bipp/config.h"
#include "gpu/kernels/nuft_sum.hpp"
#include "gpu/util/cub_api.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/queue.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
namespace {

__device__ __forceinline__ void calc_sincos(float x, float* sptr, float* cptr) {
  sincosf(x, sptr, cptr);
}

__device__ __forceinline__ void calc_sincos(double x, double* sptr, double* cptr) {
  sincos(x, sptr, cptr);
}

template <typename T, int BLOCK_THREADS, api::cub::BlockReduceAlgorithm ALGORITHM>
__global__ __launch_bounds__(BLOCK_THREADS) void nuft_sum_kernel(
    T alpha, std::size_t nIn, const api::ComplexType<T>* __restrict__ input,
    const T* __restrict__ u, const T* __restrict__ v, const T* __restrict__ w, std::size_t nOut,
    const T* __restrict__ x, const T* __restrict__ y, const T* __restrict__ z, T* out) {
  using BlockReduceType = api::cub::BlockReduce<T, BLOCK_THREADS, ALGORITHM>;
  __shared__ typename BlockReduceType::TempStorage tmpStorage;

  for (std::size_t idxOut = blockIdx.x; idxOut < nOut; idxOut += gridDim.x) {
    const auto xVal = x[idxOut];
    const auto yVal = y[idxOut];
    const auto zVal = z[idxOut];

    T localSum = 0;
    for (std::size_t idxIn = threadIdx.x; idxIn < nIn; idxIn += BLOCK_THREADS) {
      const auto imag = alpha * (xVal * u[idxIn] + yVal * v[idxIn] + zVal * w[idxIn]);
      api::ComplexType<T> sc;
      calc_sincos(imag, &(sc.y), &(sc.x));
      const auto inVal = input[idxIn];
      localSum += inVal.x * sc.x - inVal.y * sc.y;
    }

    auto totalSum = BlockReduceType(tmpStorage).Sum(localSum);
    if (threadIdx.x == 0) {
      out[idxOut] += totalSum;
    }
  }
}

template <typename T, int BLOCK_THREADS>
auto nuft_sum_launch(const api::DevicePropType& prop, const api::StreamType& stream, T alpha,
                     std::size_t nIn, const api::ComplexType<T>* __restrict__ input, const T* u,
                     const T* v, const T* w, std::size_t nOut, const T* x, const T* y, const T* z,
                     T* out) -> void {
  const dim3 block(BLOCK_THREADS, 1, 1);
  const auto grid = kernel_launch_grid(prop, {nOut, 1, 1}, block);

  api::launch_kernel(nuft_sum_kernel<T, BLOCK_THREADS,
                                     api::cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>,
                     grid, block, 0, stream, alpha, nIn, input, u, v, w, nOut, x, y, z, out);
}

}  // namespace

template <typename T>
auto nuft_sum(const api::DevicePropType& prop, const api::StreamType& stream, T alpha,
              std::size_t nIn, const api::ComplexType<T>* __restrict__ input, const T* u,
              const T* v, const T* w, std::size_t nOut, const T* x, const T* y, const T* z, T* out)
    -> void {
  if (nIn >= 1024 && prop.maxThreadsDim[0] >= 1024) {
    nuft_sum_launch<T, 1024>(prop, stream, alpha, nIn, input, u, v, w, nOut, x, y, z, out);
  } else if (nIn >= 512 && prop.maxThreadsDim[0] >= 512) {
    nuft_sum_launch<T, 512>(prop, stream, alpha, nIn, input, u, v, w, nOut, x, y, z, out);
  } else if (nIn >= 256 && prop.maxThreadsDim[0] >= 256) {
    nuft_sum_launch<T, 256>(prop, stream, alpha, nIn, input, u, v, w, nOut, x, y, z, out);
  } else if (nIn >= 128 && prop.maxThreadsDim[0] >= 128) {
    nuft_sum_launch<T, 128>(prop, stream, alpha, nIn, input, u, v, w, nOut, x, y, z, out);
  } else {
    nuft_sum_launch<T, 64>(prop, stream, alpha, nIn, input, u, v, w, nOut, x, y, z, out);
  }
}

template auto nuft_sum<float>(const api::DevicePropType& prop, const api::StreamType& stream,
                              float alpha, std::size_t nIn,
                              const api::ComplexType<float>* __restrict__ input, const float* u,
                              const float* v, const float* w, std::size_t nOut, const float* x,
                              const float* y, const float* z, float* out) -> void;

template auto nuft_sum<double>(const api::DevicePropType& prop, const api::StreamType& stream,
                               double alpha, std::size_t nIn,
                               const api::ComplexType<double>* __restrict__ input, const double* u,
                               const double* v, const double* w, std::size_t nOut, const double* x,
                               const double* y, const double* z, double* out) -> void;
}  // namespace gpu
}  // namespace bipp
