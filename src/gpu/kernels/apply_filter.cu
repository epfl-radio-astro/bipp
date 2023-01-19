#include <algorithm>

#include "bipp//config.h"
#include "gpu/kernels/apply_filter.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

static __device__ auto calc_sqrt(float x) -> float { return sqrtf(x); }

static __device__ auto calc_sqrt(double x) -> double { return sqrt(x); }

template <typename T>
__global__ void apply_filter_std_kernel(std::size_t n, T* __restrict__ out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    out[i] = 1;
  }
}

template <typename T>
__global__ void apply_filter_sqrt_kernel(std::size_t n, const T* __restrict__ in,
                                         T* __restrict__ out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    out[i] = calc_sqrt(in[i]);
  }
}

template <typename T>
__global__ void apply_filter_inv_kernel(std::size_t n, const T* __restrict__ in,
                                        T* __restrict__ out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    const auto value = in[i];
    if (value)
      out[i] = static_cast<T>(1) / value;
    else
      out[i] = 0;
  }
}

template <typename T>
__global__ void apply_filter_inv_sq_kernel(std::size_t n, const T* __restrict__ in,
                                           T* __restrict__ out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    const auto value = in[i];
    if (value)
      out[i] = static_cast<T>(1) / (value * value);
    else
      out[i] = 0;
  }
}

template <typename T>
auto apply_filter(Queue& q, BippFilter filter, std::size_t n, const T* in, T* out) -> void {
  constexpr int blockSize = 256;

  const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(q.device_prop(), {n, 1, 1}, block);

  switch (filter) {
    case BIPP_FILTER_STD: {
      api::launch_kernel(apply_filter_std_kernel<T>, grid, block, 0, q.stream(), n, out);
      break;
    }
    case BIPP_FILTER_SQRT: {
      api::launch_kernel(apply_filter_sqrt_kernel<T>, grid, block, 0, q.stream(), n, in, out);
      break;
    }
    case BIPP_FILTER_INV: {
      api::launch_kernel(apply_filter_inv_kernel<T>, grid, block, 0, q.stream(), n, in, out);
      break;
    }
    case BIPP_FILTER_INV_SQ: {
      api::launch_kernel(apply_filter_inv_sq_kernel<T>, grid, block, 0, q.stream(), n, in, out);
      break;
    }
    case BIPP_FILTER_LSQ: {
      api::memcpy_async(out, in, n * sizeof(T), api::flag::MemcpyDeviceToDevice, q.stream());
      break;
    }
  }
}

template auto apply_filter<float>(Queue& q, BippFilter filter, std::size_t n, const float* in,
                                  float* out) -> void;

template auto apply_filter<double>(Queue& q, BippFilter filter, std::size_t n, const double* in,
                                   double* out) -> void;

}  // namespace gpu
}  // namespace bipp
