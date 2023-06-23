#include <algorithm>

#include "bipp//config.h"
#include "gpu/kernels/copy_at_indices.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
__global__ static void copy_matrix_from_indices_kernel(std::size_t n, const std::size_t* __restrict__ indices,
                                                         const T* __restrict__ a, std::size_t lda,
                                                         T* __restrict__ b, std::size_t ldb) {
  for (std::size_t col = blockIdx.x; col < n; col += gridDim.x) {
    const auto colIdx = indices[col];
    for (std::size_t row = threadIdx.x; row < n;
         row += blockDim.x) {
      const auto rowIdx = indices[row];
      b[col * ldb + row] = a[colIdx * ldb + rowIdx];
    }
  }
}

template <typename T>
__global__ static void copy_matrix_rows_to_indices_kernel(
    std::size_t nRows, std::size_t nCols, const std::size_t* __restrict__ rowIndices,
    const T* __restrict__ a, std::size_t lda, T* __restrict__ b, std::size_t ldb) {
  for (std::size_t col = blockIdx.x; col < nCols; col += gridDim.x) {
    for (std::size_t row = threadIdx.x; row < nRows; row += blockDim.x) {
      const auto rowIdx = rowIndices[row];
      b[col * ldb + rowIdx] = a[col * ldb + row];
    }
  }
}

template <typename T>
auto copy_matrix_rows_to_indices(const api::DevicePropType& prop, const api::StreamType& stream,
                                 std::size_t nRows, std::size_t nCols, const std::size_t* rowIndices, const T* a,
                                 std::size_t lda, T* b, std::size_t ldb) -> void {
  constexpr int blockSize = 256;

  const dim3 block(std::min<int>(blockSize, prop.maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(prop, {nCols, 1, 1}, {1, 1, 1});
  api::launch_kernel(copy_matrix_rows_to_indices_kernel<T>, grid, block, 0, stream, nRows, nCols,
                     rowIndices, a, lda, b, ldb);
}

template <typename T>
auto copy_matrix_from_indices(const api::DevicePropType& prop, const api::StreamType& stream,
                                     std::size_t n, const std::size_t* indices, const T* a,
                                     std::size_t lda, T* b, std::size_t ldb) -> void {
  constexpr int blockSize = 256;

  const dim3 block(std::min<int>(blockSize, prop.maxThreadsDim[0]), 1, 1);
  const auto grid = kernel_launch_grid(prop, {n, 1, 1}, {1, 1, 1});
  api::launch_kernel(copy_matrix_from_indices_kernel<T>, grid, block, 0, stream, n, indices, a, lda,
                     b, ldb);
}

template auto copy_matrix_rows_to_indices<float>(const api::DevicePropType& prop, const api::StreamType& stream,
                                 std::size_t nRows, std::size_t nCols, const std::size_t* rowIndices, const float* a,
                                 std::size_t lda, float* b, std::size_t ldb) -> void;
template auto copy_matrix_rows_to_indices<double>(const api::DevicePropType& prop, const api::StreamType& stream,
                                 std::size_t nRows, std::size_t nCols, const std::size_t* rowIndices, const double* a,
                                 std::size_t lda, double* b, std::size_t ldb)
    -> void;

template auto copy_matrix_rows_to_indices<api::ComplexFloatType>(
    const api::DevicePropType& prop, const api::StreamType& stream, std::size_t nRows,
    std::size_t nCols, const std::size_t* rowIndices, const api::ComplexFloatType* a,
    std::size_t lda, api::ComplexFloatType* b, std::size_t ldb) -> void;

template auto copy_matrix_rows_to_indices<api::ComplexDoubleType>(
    const api::DevicePropType& prop, const api::StreamType& stream, std::size_t nRows,
    std::size_t nCols, const std::size_t* rowIndices, const api::ComplexDoubleType* a,
    std::size_t lda, api::ComplexDoubleType* b, std::size_t ldb) -> void;

template auto copy_matrix_from_indices<float>(const api::DevicePropType& prop,
                                                   const api::StreamType& stream, std::size_t n,
                                                   const std::size_t* indices, const float* a,
                                                   std::size_t lda, float* b, std::size_t ldb)
    -> void;
template auto copy_matrix_from_indices<double>(const api::DevicePropType& prop,
                                                   const api::StreamType& stream, std::size_t n,
                                                   const std::size_t* indices, const double* a,
                                                   std::size_t lda, double* b, std::size_t ldb)
    -> void;

template auto copy_matrix_from_indices<api::ComplexFloatType>(
    const api::DevicePropType& prop, const api::StreamType& stream, std::size_t n,
    const std::size_t* indices, const api::ComplexFloatType* a, std::size_t lda,
    api::ComplexFloatType* b, std::size_t ldb) -> void;

template auto copy_matrix_from_indices<api::ComplexDoubleType>(
    const api::DevicePropType& prop, const api::StreamType& stream, std::size_t n,
    const std::size_t* indices, const api::ComplexDoubleType* a, std::size_t lda,
    api::ComplexDoubleType* b, std::size_t ldb) -> void;

}  // namespace gpu
}  // namespace bipp
