#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <cassert>
#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/kernels/reverse.hpp"
#include "gpu/util/blas_api.hpp"
#include "gpu/util/runtime_api.hpp"
#include "gpu/util/solver_api.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto eigh(ContextInternal& ctx, std::size_t m, std::size_t nEig, const api::ComplexType<T>* a,
          std::size_t lda, const api::ComplexType<T>* b, std::size_t ldb, const char range,
          std::size_t* nEigOut, T* d, api::ComplexType<T>* v, std::size_t ldv) -> void {

  // TODO: add fill mode
  using ComplexType = api::ComplexType<T>;
  using ScalarType = T;
  auto& queue = ctx.gpu_queue();

  auto aBuffer = queue.create_device_buffer<ComplexType>(m * m);  // Matrix A
  auto dBuffer = queue.create_device_buffer<T>(m);                // Matrix D

  api::memcpy_2d_async(aBuffer.get(), m * sizeof(ComplexType), a, lda * sizeof(ComplexType),
                       m * sizeof(ComplexType), m, api::flag::MemcpyDeviceToDevice, queue.stream());
  int hMeig = 0;

  // compute eigenvalues
  eigensolver::solve(ctx, 'V', range, 'L', m, aBuffer.get(), m, std::numeric_limits<T>::epsilon(),
                     std::numeric_limits<T>::max(), 1, m, &hMeig, dBuffer.get());

  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvalues", hMeig, 1, dBuffer.get(), hMeig);
  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", m, m, aBuffer.get(), m);
  if (b) {
    auto bBuffer = queue.create_device_buffer<ComplexType>(m * m);  // Matrix B
    api::memcpy_2d_async(bBuffer.get(), m * sizeof(ComplexType), b, ldb * sizeof(ComplexType),
                         m * sizeof(ComplexType), m, api::flag::MemcpyDeviceToDevice,
                         queue.stream());
    if (static_cast<std::size_t>(hMeig) != m) {
      // reconstuct 'a' without negative eigenvalues (v * diag(d) * v^H)
      auto dComplexD = queue.create_device_buffer<ComplexType>(m);
      auto cD = queue.create_device_buffer<ComplexType>(m * m);
      auto newABuffer = queue.create_device_buffer<ComplexType>(m * m);

      // copy scalar eigenvalues to complex for multiplication
      api::memset_async(dComplexD.get(), 0, hMeig * sizeof(ComplexType), queue.stream());
      api::memcpy_2d_async(dComplexD.get(), sizeof(ComplexType), dBuffer.get(), sizeof(ScalarType),
                           sizeof(ScalarType), hMeig, api::flag::MemcpyDeviceToDevice,
                           queue.stream());

      api::blas::dgmm(queue.blas_handle(), api::blas::side::right, m, hMeig, aBuffer.get(), m,
                      dComplexD.get(), 1, cD.get(), m);
      ComplexType alpha{1, 0};
      ComplexType beta{0, 0};
      api::blas::gemm(queue.blas_handle(), api::blas::operation::None,
                      api::blas::operation::ConjugateTranspose, m, m, hMeig, &alpha, cD.get(), m,
                      aBuffer.get(), m, &beta, newABuffer.get(), m);
      std::swap(newABuffer, aBuffer);
    } else {
      // a was overwritten by eigensolver
      api::memcpy_2d_async(aBuffer.get(), m * sizeof(ComplexType), a, lda * sizeof(ComplexType),
                           m * sizeof(ComplexType), m, api::flag::MemcpyDeviceToDevice,
                           queue.stream());
    }

    // compute general eigenvalues
    eigensolver::solve(ctx, 'V', range, 'L', m, aBuffer.get(), m, bBuffer.get(), m,
                       std::numeric_limits<T>::epsilon(), std::numeric_limits<T>::max(), 1, m,
                       &hMeig, dBuffer.get());
    ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gen. eigenvalues", hMeig, 1, dBuffer.get(),
                            hMeig);
    ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gen. eigenvectors", m, m, aBuffer.get(), m);
  }

  if (range == 'A')
      assert(nEig == hMeig);

  if (hMeig > 1) {
    // reverse order, such that large eigenvalues are first
    reverse_1d<ScalarType>(queue, hMeig, dBuffer.get());
    reverse_2d(queue, m, hMeig, aBuffer.get(), m);
  }

  if (static_cast<std::size_t>(hMeig) < nEig) {
    // fewer positive eigenvalues found than requested. Setting others to 0.
    api::memset_async(dBuffer.get() + hMeig, 0, (nEig - hMeig) * sizeof(ScalarType),
                      queue.stream());
    api::memset_async(aBuffer.get() + hMeig * m, 0, (nEig - hMeig) * m * sizeof(ComplexType),
                      queue.stream());
  }

  // copy results to output
  api::memcpy_async(d, dBuffer.get(), nEig * sizeof(ScalarType), api::flag::MemcpyDeviceToDevice,
                    queue.stream());
  api::memcpy_2d_async(v, ldv * sizeof(ComplexType), aBuffer.get(), m * sizeof(ComplexType),
                       m * sizeof(ComplexType), nEig, api::flag::MemcpyDeviceToDevice,
                       queue.stream());

  *nEigOut = std::min<std::size_t>(hMeig, nEig);
}

template auto eigh<float>(ContextInternal& ctx, std::size_t m, std::size_t nEig,
                          const api::ComplexType<float>* a, std::size_t lda,
                          const api::ComplexType<float>* b, std::size_t ldb, const char range,
                          std::size_t* nEigOut, float* d, api::ComplexType<float>* v, std::size_t ldv) -> void;

template auto eigh<double>(ContextInternal& ctx, std::size_t m, std::size_t nEig,
                           const api::ComplexType<double>* a, std::size_t lda,
                           const api::ComplexType<double>* b, std::size_t ldb, const char range,
                           std::size_t* nEigOut, double* d, api::ComplexType<double>* v, std::size_t ldv) -> void;

}  // namespace gpu
}  // namespace bipp
