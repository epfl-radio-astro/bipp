#include <algorithm>

#include "bipp/config.h"
#include "gpu/kernels/center_vector.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"
#include "gpu/util/cub_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
static __global__ void sub_from_vector_kernel(std::size_t n, const T* __restrict__ value,
                                              T* __restrict__ vec) {
  const T mean = *value / n;
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    vec[i] -= mean;
  }
}

template <typename T>
auto center_vector(Queue& q, std::size_t n, T* vec) -> void {
  std::size_t worksize = 0;
  api::check_status(
      api::cub::DeviceReduce::Sum<const T*, T*>(nullptr, worksize, nullptr, nullptr, n, q.stream()));

  auto workBuffer = q.create_device_array<char,1>(sizeof(T) + worksize);

  // To avoid alignment issues for type T, sum up at beginning of work array and
  // provide remaining memory to reduce function
  T* sumPtr = reinterpret_cast<T*>(workBuffer.data());

  api::check_status(api::cub::DeviceReduce::Sum<const T*, T*>(
      reinterpret_cast<T*>(workBuffer.data()) + 1, worksize, vec, sumPtr, n, q.stream()));

  constexpr int blockSize = 256;
  const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(q.device_prop(), {n, 1, 1}, block);

  api::launch_kernel(sub_from_vector_kernel<T>, grid, block, 0, q.stream(), n, sumPtr, vec);
}

template auto center_vector<float>(Queue& q, std::size_t n, float* vec) -> void;

template auto center_vector<double>(Queue& q, std::size_t n, double* vec) -> void;
}  // namespace gpu
}  // namespace bipp
