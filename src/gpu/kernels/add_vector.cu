#include <algorithm>

#include "bipp//config.h"
#include "gpu/kernels/add_vector.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
__global__ static void add_vector_real_of_complex_kernel(std::size_t n,
                                                         const api::ComplexType<T>* __restrict__ a,
                                                         float* __restrict__ b) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    b[i] += a[i].x;
  }
}

template <typename T>
auto add_vector_real_of_complex(const api::DevicePropType& prop, const api::StreamType& stream,
                                std::size_t n, const api::ComplexType<T>* a, float* b) -> void {
  constexpr int blockSize = 256;

  const dim3 block(std::min<int>(blockSize, prop.maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(prop, {n, 1, 1}, block);
  api::launch_kernel(add_vector_real_of_complex_kernel<T>, grid, block, 0, stream, n, a, b);
}

template auto add_vector_real_of_complex<float>(const api::DevicePropType& prop,
                                                const api::StreamType& stream, std::size_t n,
                                                const api::ComplexType<float>* a, float* b) -> void;

template auto add_vector_real_of_complex<double>(const api::DevicePropType& prop,
                                                 const api::StreamType& stream, std::size_t n,
                                                 const api::ComplexType<double>* a, float* b)
    -> void;

}  // namespace gpu
}  // namespace bipp
