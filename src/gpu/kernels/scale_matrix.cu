#include <algorithm>

#include "bipp//config.h"
#include "gpu/kernels/scale_matrix.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
__global__ void scale_matrix_kernel(std::size_t m, std::size_t n,
                                    const api::ComplexType<T>* __restrict__ A, std::size_t lda,
                                    const T* __restrict__ x, api::ComplexType<T>* __restrict__ B,
                                    std::size_t ldb) {
  for (std::size_t j = blockIdx.y; j < n; j += gridDim.y) {
    const auto valX = x[j];
    for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < m;
         i += gridDim.x * blockDim.x) {
      const auto valA = A[j * lda + i];
      B[j * ldb + i] = {valA.x * valX, valA.y * valX};
    }
  }
}

template <typename T>
auto scale_matrix(Queue& q, std::size_t m, std::size_t n, const api::ComplexType<T>* A,
                  std::size_t lda, const T* x, api::ComplexType<T>* B, std::size_t ldb) -> void {
  constexpr int blockSize = 256;
  const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(q.device_prop(), {m, n, 1}, block);

  api::launch_kernel(scale_matrix_kernel<T>, grid, block, 0, q.stream(), m, n, A, lda, x, B, ldb);
}

template auto scale_matrix<float>(Queue& q, std::size_t m, std::size_t n,
                                  const api::ComplexType<float>* A, std::size_t lda, const float* x,
                                  api::ComplexType<float>* B, std::size_t ldb) -> void;

template auto scale_matrix<double>(Queue& q, std::size_t m, std::size_t n,
                                   const api::ComplexType<double>* A, std::size_t lda,
                                   const double* x, api::ComplexType<double>* B, std::size_t ldb)
    -> void;

}  // namespace gpu
}  // namespace bipp
