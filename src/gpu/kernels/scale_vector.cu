#include <algorithm>

#include "bipp//config.h"
#include "gpu/kernels/scale_vector.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
__global__ static void scale_vector_kernel(std::size_t n, const T* __restrict__ a, T alpha,
                                           T* __restrict__ b) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    b[i] = alpha * a[i];
  }
}

template <typename T>
__global__ static void scale_vector_inplace_kernel(std::size_t n, T alpha, T* a) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    a[i] *= alpha;
  }
}

template <typename T>
auto scale_vector(const api::DevicePropType& prop, const api::StreamType& stream, std::size_t n,
                  const T* a, T alpha, T* b) -> void {
  constexpr int blockSize = 256;

  const dim3 block(std::min<int>(blockSize, prop.maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(prop, {n, 1, 1}, block);
  api::launch_kernel(scale_vector_kernel<T>, grid, block, 0, stream, n, a, alpha, b);
}

template <typename T>
auto scale_vector(const api::DevicePropType& prop, const api::StreamType& stream, std::size_t n,
                  T alpha, T* a) -> void {
  constexpr int blockSize = 256;

  const dim3 block(std::min<int>(blockSize, prop.maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(prop, {n, 1, 1}, block);
  api::launch_kernel(scale_vector_inplace_kernel<T>, grid, block, 0, stream, n, alpha, a);
}

template auto scale_vector<float>(const api::DevicePropType& prop, const api::StreamType& stream,
                                  std::size_t n, const float* a, float alpha, float* b) -> void;

template auto scale_vector<double>(const api::DevicePropType& prop, const api::StreamType& stream,
                                   std::size_t n, const double* a, double alpha, double* b) -> void;

template auto scale_vector<float>(const api::DevicePropType& prop, const api::StreamType& stream,
                                  std::size_t n, float alpha, float* a) -> void;

template auto scale_vector<double>(const api::DevicePropType& prop, const api::StreamType& stream,
                                   std::size_t n, double alpha, double* a) -> void;

}  // namespace gpu
}  // namespace bipp
