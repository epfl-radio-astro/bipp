#include <algorithm>

#include "bipp//config.h"
#include "gpu/kernels/init_vector.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
__global__ static void init_vector_kernel(std::size_t n, T value, T* __restrict__ a) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    a[i] = value;
  }
}

template <typename T>
auto init_vector(const api::DevicePropType& prop, const api::StreamType& stream, std::size_t n,
                 T value, T* a) -> void {
  constexpr int blockSize = 256;

  const dim3 block(std::min<int>(blockSize, prop.maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(prop, {n, 1, 1}, block);
  api::launch_kernel(init_vector_kernel<T>, grid, block, 0, stream, n, value, a);
}

template auto init_vector<float>(const api::DevicePropType& prop, const api::StreamType& stream,
                                 std::size_t n, float value, float* a) -> void;

template auto init_vector<double>(const api::DevicePropType& prop, const api::StreamType& stream,
                                  std::size_t n, double value, double* a) -> void;

template auto init_vector<api::ComplexType<float>>(const api::DevicePropType& prop,
                                                   const api::StreamType& stream, std::size_t n,
                                                   api::ComplexType<float> value,
                                                   api::ComplexType<float>* a) -> void;

template auto init_vector<api::ComplexType<double>>(const api::DevicePropType& prop,
                                                    const api::StreamType& stream, std::size_t n,
                                                    api::ComplexType<double> value,
                                                    api::ComplexType<double>* a) -> void;

}  // namespace gpu
}  // namespace bipp
