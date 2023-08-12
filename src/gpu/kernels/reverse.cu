#include <algorithm>

#include "bipp/config.h"
#include "gpu/kernels/reverse.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
__global__ void reverse_1d_kernel(std::size_t n, T* __restrict__ x) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n / 2;
       i += gridDim.x * blockDim.x) {
    T x1 = x[i];
    T x2 = x[n - 1 - i];
    x[n - 1 - i] = x1;
    x[i] = x2;
  }
}

template <typename T>
__global__ void reverse_2d_columns_kernel(std::size_t m, std::size_t n, T* __restrict__ x,
                                          std::size_t ld) {
  int n_2 = n % 2 == 0 ? n / 2 : (n - 1) / 2;
  std::size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix < m && iy < n_2) {
    std::size_t i1 = iy * m + ix;
    std::size_t i2 = (n - 1 - iy) * m + ix;
    T x1 = x[i1];
    T x2 = x[i2];
    x[i2] = x1;
    x[i1] = x2;
  }
}

template <typename T>
auto reverse_1d(Queue& q, std::size_t n, T* x) -> void {
  constexpr int blockSize = 256;
  const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(q.device_prop(), {n, 1, 1}, block);
  api::launch_kernel(reverse_1d_kernel<T>, grid, block, 0, q.stream(), n, x);
}

template <typename T>
auto reverse_2d(Queue& q, std::size_t m, std::size_t n, T* x, std::size_t ld) -> void {
  constexpr int blockSize = 256;
  const dim3 block(std::min<int>(blockSize / 8, q.device_prop().maxThreadsDim[0]), 8, 1);
  const auto grid = kernel_launch_grid(q.device_prop(), {m, n, 1}, block);
  api::launch_kernel(reverse_2d_columns_kernel<T>, grid, block, 0, q.stream(), m, n, x, ld);
  cudaDeviceSynchronize();
}

template auto reverse_1d<float>(Queue& q, std::size_t n, float* x) -> void;

template auto reverse_1d<double>(Queue& q, std::size_t n, double* x) -> void;

template auto reverse_1d<api::ComplexType<float>>(Queue& q, std::size_t n,
                                                  api::ComplexType<float>* x) -> void;

template auto reverse_1d<api::ComplexType<double>>(Queue& q, std::size_t n,
                                                   api::ComplexType<double>* x) -> void;

template auto reverse_2d<float>(Queue& q, std::size_t m, std::size_t n, float* x, std::size_t ld)
    -> void;

template auto reverse_2d<double>(Queue& q, std::size_t m, std::size_t n, double* x, std::size_t ld)
    -> void;

template auto reverse_2d<api::ComplexType<float>>(Queue& q, std::size_t m, std::size_t n,
                                                  api::ComplexType<float>* x, std::size_t ld)
    -> void;

template auto reverse_2d<api::ComplexType<double>>(Queue& q, std::size_t m, std::size_t n,
                                                   api::ComplexType<double>* x, std::size_t ld)
    -> void;

}  // namespace gpu
}  // namespace bipp
