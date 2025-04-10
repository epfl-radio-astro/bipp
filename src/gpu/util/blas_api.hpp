#pragma once

#include <stdexcept>
#include <utility>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/view.hpp"

#if defined(BIPP_CUDA)
#include <cublas_v2.h>

#elif defined(BIPP_ROCM)
#include <rocblas/rocblas.h>

#else
#error Either BIPP_CUDA or BIPP_ROCM must be defined!
#endif

namespace bipp {
namespace gpu {
namespace api {
namespace blas {

#if defined(BIPP_CUDA)
using HandleType = cublasHandle_t;
using StatusType = cublasStatus_t;
using OperationType = cublasOperation_t;
using SideModeType = cublasSideMode_t;
using FillModeType = cublasFillMode_t;
using DiagType = cublasDiagType_t;
#endif

#if defined(BIPP_ROCM)
using HandleType = rocblas_handle;
using StatusType = rocblas_status;
using OperationType = rocblas_operation;
using SideModeType = rocblas_side;
using FillModeType = rocblas_fill;
using DiagType = rocblas_diagonal;
#endif

namespace operation {
#if defined(BIPP_CUDA)
constexpr auto None = CUBLAS_OP_N;
constexpr auto Transpose = CUBLAS_OP_T;
constexpr auto ConjugateTranspose = CUBLAS_OP_C;
#endif

#if defined(BIPP_ROCM)
constexpr auto None = rocblas_operation_none;
constexpr auto Transpose = rocblas_operation_transpose;
constexpr auto ConjugateTranspose = rocblas_operation_conjugate_transpose;
#endif
}  // namespace operation

namespace side {
#if defined(BIPP_CUDA)
constexpr auto left = CUBLAS_SIDE_LEFT;
constexpr auto right = CUBLAS_SIDE_RIGHT;
#endif

#if defined(BIPP_ROCM)
constexpr auto left = rocblas_side_left;
constexpr auto right = rocblas_side_right;
#endif
}  // namespace side

namespace fill {
#if defined(BIPP_CUDA)
constexpr auto upper = CUBLAS_FILL_MODE_UPPER;
constexpr auto lower = CUBLAS_FILL_MODE_LOWER;
constexpr auto full = CUBLAS_FILL_MODE_FULL;
#endif

#if defined(BIPP_ROCM)
constexpr auto upper = rocblas_fill_upper;
constexpr auto lower = rocblas_fill_lower;
constexpr auto full = rocblas_fill_full;
#endif
}  // namespace fill


namespace diag {
#if defined(BIPP_CUDA)
constexpr auto unit = CUBLAS_DIAG_UNIT;
constexpr auto nonunit = CUBLAS_DIAG_NON_UNIT;
#endif

#if defined(BIPP_ROCM)
constexpr auto unit = rocblas_diagonal_unit;
constexpr auto nonunit = rocblas_diagonal_non_unit;
#endif
}  // namespace fill

namespace status {
#if defined(BIPP_CUDA)
constexpr auto Success = CUBLAS_STATUS_SUCCESS;
#endif

#if defined(BIPP_ROCM)
constexpr auto Success = rocblas_status_success;
#endif

static const char* get_string(StatusType error) {
#if defined(BIPP_CUDA)
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "CUBLAS_ERROR";
  }
#endif

#if defined(BIPP_ROCM)
  switch (error) {
    case rocblas_status_success:
      return "rocblas_status_success";

    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";

    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";

    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";

    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";

    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";

    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";

    case rocblas_status_perf_degraded:
      return "rocblas_status_perf_degraded";

    case rocblas_status_size_query_mismatch:
      return "rocblas_status_size_query_mismatch";

    case rocblas_status_size_increased:
      return "rocblas_status_size_increased";

    case rocblas_status_size_unchanged:
      return "rocblas_status_size_unchanged";
    default:
      return "rocblas_error";
  }
#endif

  return "gpu_blas_unknown_error";
}
}  // namespace status

inline auto check_status(StatusType error) -> void {
  if (error != status::Success) {
    throw GPUBlasError(status::get_string(error));
  }
}

// ========================================================
// Forwarding functions of to GPU BLAS API with error check
// ========================================================
template <typename... ARGS>
inline auto create(ARGS&&... args) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasCreate(std::forward<ARGS>(args)...));
#else
  check_status(rocblas_create_handle(std::forward<ARGS>(args)...));
#endif
}

template <typename... ARGS>
inline auto set_stream(ARGS&&... args) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasSetStream(std::forward<ARGS>(args)...));
#else
  check_status(rocblas_set_stream(std::forward<ARGS>(args)...));
#endif
}

template <typename... ARGS>
inline auto get_stream(ARGS&&... args) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasGetStream(std::forward<ARGS>(args)...));
#else
  check_status(rocblas_get_stream(std::forward<ARGS>(args)...));
#endif
}

inline auto gemm(HandleType handle, OperationType transa, OperationType transb, int m, int n, int k,
                 const float* alpha, const float* A, int lda, const float* B, int ldb,
                 const float* beta, float* C, int ldc) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
#else
  check_status(rocblas_sgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
#endif  // BIPP_CUDA
}

inline auto gemm(HandleType handle, OperationType transa, OperationType transb, int m, int n, int k,
                 const double* alpha, const double* A, int lda, const double* B, int ldb,
                 const double* beta, double* C, int ldc) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
#else
  check_status(rocblas_dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
#endif  // BIPP_CUDA
}

inline auto gemm(HandleType handle, OperationType transa, OperationType transb, int m, int n, int k,
                 const ComplexFloatType* alpha, const ComplexFloatType* A, int lda,
                 const ComplexFloatType* B, int ldb, const ComplexFloatType* beta,
                 ComplexFloatType* C, int ldc) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
#else
  check_status(rocblas_cgemm(handle, transa, transb, m, n, k,
                             reinterpret_cast<const rocblas_float_complex*>(alpha),
                             reinterpret_cast<const rocblas_float_complex*>(A), lda,
                             reinterpret_cast<const rocblas_float_complex*>(B), ldb,
                             reinterpret_cast<const rocblas_float_complex*>(beta),
                             reinterpret_cast<rocblas_float_complex*>(C), ldc));
#endif  // BIPP_CUDA
}

inline auto gemm(HandleType handle, OperationType transa, OperationType transb, int m, int n, int k,
                 const ComplexDoubleType* alpha, const ComplexDoubleType* A, int lda,
                 const ComplexDoubleType* B, int ldb, const ComplexDoubleType* beta,
                 ComplexDoubleType* C, int ldc) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
#else
  check_status(rocblas_zgemm(handle, transa, transb, m, n, k,
                             reinterpret_cast<const rocblas_double_complex*>(alpha),
                             reinterpret_cast<const rocblas_double_complex*>(A), lda,
                             reinterpret_cast<const rocblas_double_complex*>(B), ldb,
                             reinterpret_cast<const rocblas_double_complex*>(beta),
                             reinterpret_cast<rocblas_double_complex*>(C), ldc));
#endif  // BIPP_CUDA
}

template <typename T>
auto gemm(HandleType handle, OperationType transa, OperationType transb, T alpha,
          ConstDeviceView<T, 2> a, ConstDeviceView<T, 2> b, T beta, DeviceView<T, 2> c) {
  const auto m = transa == operation::None ? a.shape(0) : a.shape(1);
  const auto n = transb == operation::None ? b.shape(1) : b.shape(0);
  const auto k = transa == operation::None ? a.shape(1) : a.shape(0);

  assert(c.shape(0) == m);
  assert(c.shape(1) == n);
  assert(!(b.shape(0) != k && transb == operation::None));
  assert(!(b.shape(1) != k && transb != operation::None));

  gemm(handle, transa, transb, m, n, k, &alpha, a.data(), a.strides(1), b.data(), b.strides(1),
       &beta, c.data(), c.strides(1));
}

inline auto gemm_batched(HandleType handle, OperationType transa, OperationType transb, int m,
                         int n, int k, const ComplexFloatType* alpha,
                         const ComplexFloatType* const A[], int lda,
                         const ComplexFloatType* const B[], int ldb, const ComplexFloatType* beta,
                         ComplexFloatType* const C[], int ldc, int batchCount) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                  ldc, batchCount));
#else
  check_status(rocblas_cgemm_batched(
      handle, transa, transb, m, n, k, reinterpret_cast<const rocblas_float_complex*>(alpha),
      reinterpret_cast<const rocblas_float_complex* const*>(A), lda,
      reinterpret_cast<const rocblas_float_complex* const*>(B), ldb,
      reinterpret_cast<const rocblas_float_complex*>(beta),
      reinterpret_cast<rocblas_float_complex* const*>(C), ldc, batchCount));
#endif  // BIPP_CUDA
}

inline auto gemm_batched(HandleType handle, OperationType transa, OperationType transb, int m,
                         int n, int k, const ComplexDoubleType* alpha,
                         const ComplexDoubleType* const A[], int lda,
                         const ComplexDoubleType* const B[], int ldb, const ComplexDoubleType* beta,
                         ComplexDoubleType* const C[], int ldc, int batchCount) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasZgemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                  ldc, batchCount));
#else
  check_status(rocblas_zgemm_batched(
      handle, transa, transb, m, n, k, reinterpret_cast<const rocblas_double_complex*>(alpha),
      reinterpret_cast<const rocblas_double_complex* const*>(A), lda,
      reinterpret_cast<const rocblas_double_complex* const*>(B), ldb,
      reinterpret_cast<const rocblas_double_complex*>(beta),
      reinterpret_cast<rocblas_double_complex* const*>(C), ldc, batchCount));
#endif  // BIPP_CUDA
}

inline auto dgmm(HandleType handle, SideModeType mode, int m, int n, const float* A, int lda,
                 const float* x, int incx, float* C, int ldc) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc));
#else
  check_status(rocblas_sdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc));
#endif  // BIPP_CUDA
}

inline auto dgmm(HandleType handle, SideModeType mode, int m, int n, const double* A, int lda,
                 const double* x, int incx, double* C, int ldc) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc));
#else
  check_status(rocblas_ddgmm(handle, mode, m, n, A, lda, x, incx, C, ldc));
#endif  // BIPP_CUDA
}

inline auto dgmm(HandleType handle, SideModeType mode, int m, int n, const ComplexFloatType* A,
                 int lda, const ComplexFloatType* x, int incx, ComplexFloatType* C, int ldc)
    -> void {
#if defined(BIPP_CUDA)
  check_status(cublasCdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc));
#else
  check_status(rocblas_cdgmm(handle, mode, m, n, reinterpret_cast<const rocblas_float_complex*>(A),
                             lda, reinterpret_cast<const rocblas_float_complex*>(x), incx,
                             reinterpret_cast<rocblas_float_complex*>(C), ldc));
#endif  // BIPP_CUDA
}

inline auto dgmm(HandleType handle, SideModeType mode, int m, int n, const ComplexDoubleType* A,
                 int lda, const ComplexDoubleType* x, int incx, ComplexDoubleType* C, int ldc)
    -> void {
#if defined(BIPP_CUDA)
  check_status(cublasZdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc));
#else
  check_status(rocblas_zdgmm(handle, mode, m, n, reinterpret_cast<const rocblas_double_complex*>(A),
                             lda, reinterpret_cast<const rocblas_double_complex*>(x), incx,
                             reinterpret_cast<rocblas_double_complex*>(C), ldc));
#endif  // BIPP_CUDA
}

inline auto symm(HandleType handle, SideModeType side, FillModeType uplo, int m, int n,
                 const float* alpha, const float* A, int lda, const float* B, int ldb,
                 const float* beta, float* C, int ldc) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
#else
  check_status(rocblas_ssymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
#endif  // BIPP_CUDA
}

inline auto symm(HandleType handle, SideModeType side, FillModeType uplo, int m, int n,
                 const double* alpha, const double* A, int lda, const double* B, int ldb,
                 const double* beta, double* C, int ldc) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasDsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
#else
  check_status(rocblas_dsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
#endif  // BIPP_CUDA
}

inline auto symm(HandleType handle, SideModeType side, FillModeType uplo, int m, int n,
                 const ComplexFloatType* alpha, const ComplexFloatType* A, int lda,
                 const ComplexFloatType* B, int ldb, const ComplexFloatType* beta,
                 ComplexFloatType* C, int ldc) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasCsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
#else
  check_status(rocblas_csymm(handle, side, uplo, m, n,
                             reinterpret_cast<const rocblas_float_complex*>(alpha),
                             reinterpret_cast<const rocblas_float_complex*>(A), lda,
                             reinterpret_cast<const rocblas_float_complex*>(B), ldb,
                             reinterpret_cast<const rocblas_float_complex*>(beta),
                             reinterpret_cast<rocblas_float_complex*>(C), ldc));
#endif  // BIPP_CUDA
}

inline auto symm(HandleType handle, SideModeType side, FillModeType uplo, int m, int n,
                 const ComplexDoubleType* alpha, const ComplexDoubleType* A, int lda,
                 const ComplexDoubleType* B, int ldb, const ComplexDoubleType* beta,
                 ComplexDoubleType* C, int ldc) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasZsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
#else
  check_status(rocblas_zsymm(handle, side, uplo, m, n,
                             reinterpret_cast<const rocblas_double_complex*>(alpha),
                             reinterpret_cast<const rocblas_double_complex*>(A), lda,
                             reinterpret_cast<const rocblas_double_complex*>(B), ldb,
                             reinterpret_cast<const rocblas_double_complex*>(beta),
                             reinterpret_cast<rocblas_double_complex*>(C), ldc));
#endif  // BIPP_CUDA
}

template <typename T>
auto symm(HandleType handle, SideModeType side, FillModeType uplo, T alpha, ConstDeviceView<T, 2> A,
          ConstDeviceView<T, 2> B, T beta, DeviceView<T, 2> C) {
  const auto m = C.shape(0);
  const auto n = C.shape(1);

  assert(side == side::left ? (A.shape(0) == m) : (A.shape(0) == n));
  assert(side == side::left ? (A.shape(1) == m) : (A.shape(1) == n));

  assert(B.shape(0) == m);
  assert(B.shape(1) == n);

  symm(handle, side, uplo, m, n, &alpha, A.data(), A.strides(1), B.data(), B.strides(1), &beta,
       C.data(), C.strides(1));
}

inline auto axpy(HandleType handle, int n, const float* alpha, const float* x, int incx, float* y,
                 int incy) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasSaxpy(handle, n, alpha, x, incx, y, incy));
#else
  check_status(rocblas_saxpy(handle, n, alpha, x, incx, y, incy));
#endif  // BIPP_CUDA
}

inline auto axpy(HandleType handle, int n, const double* alpha, const double* x, int incx,
                 double* y, int incy) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasDaxpy(handle, n, alpha, x, incx, y, incy));
#else
  check_status(rocblas_daxpy(handle, n, alpha, x, incx, y, incy));
#endif  // BIPP_CUDA
}

inline auto trmv(HandleType handle, FillModeType uplo, OperationType trans, DiagType diag, int n,
                 const ComplexFloatType* A, int lda, ComplexFloatType* x, int incx) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx));
#else
  check_status(rocblas_ctrmv(handle, uplo, trans, diag, n,
                             reinterpret_cast<const rocblas_float_complex*>(A), lda,
                             reinterpret_cast<rocblas_float_complex*>(x), incx));
#endif  // BIPP_CUDA
}

inline auto trmv(HandleType handle, FillModeType uplo, OperationType trans, DiagType diag, int n,
                 const ComplexDoubleType* A, int lda, ComplexDoubleType* x, int incx) -> void {
#if defined(BIPP_CUDA)
  check_status(cublasZtrmv(handle, uplo, trans, diag, n, A, lda, x, incx));
#else
  check_status(rocblas_ztrmv(handle, uplo, trans, diag, n,
                             reinterpret_cast<const rocblas_double_complex*>(A), lda,
                             reinterpret_cast<rocblas_double_complex*>(x), incx));
#endif  // BIPP_CUDA
}

// ==========================================================
// Forwarding functions of to GPU BLAS API with status return
// ==========================================================

template <typename... ARGS>
inline auto destroy(ARGS&&... args) -> StatusType {
#if defined(BIPP_CUDA)
  return cublasDestroy(std::forward<ARGS>(args)...);
#else
  return rocblas_destroy_handle(std::forward<ARGS>(args)...);
#endif
}

}  // namespace blas
}  // namespace api
}  // namespace gpu
}  // namespace bipp
