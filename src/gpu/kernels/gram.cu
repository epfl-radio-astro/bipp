#include <algorithm>

#include "bipp//config.h"
#include "gpu/kernels/gram.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

static __device__ __forceinline__ float calc_sqrt(float x) { return sqrtf(x); }
static __device__ __forceinline__ double calc_sqrt(double x) { return sqrt(x); }

static __device__ __forceinline__ float calc_pi_sinc(float x) {
    constexpr auto pi = float(3.14159265358979323846);
    return x ? sinf(pi * x) / (pi * x) : float(1.0);
}
static __device__ __forceinline__ double calc_pi_sinc(double x) {
    constexpr auto pi = double(3.14159265358979323846);
    return x ? sin(pi * x) / (pi * x) : double(1.0);
}

template <typename T>
static __global__ void gram_kernel(std::size_t n, const T* __restrict__ x, const T* __restrict__ y,
                                   const T* __restrict__ z, T wl,
                                   api::ComplexType<T>* __restrict__ g, std::size_t ldg) {

  for (std::size_t j = threadIdx.y + blockIdx.y * blockDim.y; j < n; j += gridDim.y * blockDim.y) {
    T x1 = x[j];
    T y1 = y[j];
    T z1 = z[j];
    for (std::size_t i = j + threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += gridDim.x * blockDim.x) {
      T diffX = x1 - x[i];
      T diffY = y1 - y[i];
      T diffZ = z1 - z[i];

      T norm = calc_sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);
      g[i + j * ldg] = { calc_pi_sinc(T(2) * norm / wl), 0};
    }
  }
}

template <typename T>
auto gram(Queue& q, std::size_t n, const T* x, const T* y, const T* z, T wl, api::ComplexType<T>* g,
          std::size_t ldg) -> void {
  const int blockSizeX = std::min<int>(16, q.device_prop().maxThreadsDim[0]);
  const int blockSizeY = std::min<int>(16, q.device_prop().maxThreadsDim[1]);
  const dim3 block(blockSizeX, blockSizeY, 1);
  const auto grid = kernel_launch_grid(q.device_prop(), {n, n, 1}, block);
  api::launch_kernel(gram_kernel<T>, grid, block, 0, q.stream(), n, x, y, z, wl, g, ldg);
}

template auto gram<float>(Queue& q, std::size_t n, const float* x, const float* y, const float* z,
                          float wl, api::ComplexType<float>* g, std::size_t ldg) -> void;

template auto gram<double>(Queue& q, std::size_t n, const double* x, const double* y,
                           const double* z, double wl, api::ComplexType<double>* g, std::size_t ldg)
    -> void;
}  // namespace gpu
}  // namespace bipp
