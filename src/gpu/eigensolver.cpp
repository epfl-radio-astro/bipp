#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/kernels//copy_at_indices.hpp"
#include "gpu/util/blas_api.hpp"
#include "gpu/util/runtime_api.hpp"
#include "gpu/util/solver_api.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto eigh(ContextInternal& ctx, std::size_t m, std::size_t nEig, const api::ComplexType<T>* a,
          std::size_t lda, const api::ComplexType<T>* b, std::size_t ldb, T* d,
          api::ComplexType<T>* v, std::size_t ldv) -> void {
  // TODO: add fill mode
  using ComplexType = api::ComplexType<T>;
  using ScalarType = T;
  auto& queue = ctx.gpu_queue();

  auto aBuffer = queue.create_device_buffer<ComplexType>(m * m);
  auto dBuffer = queue.create_device_buffer<T>(m);

  auto colSumBufferHost = queue.create_pinned_buffer<ComplexType>(m);
  auto colSumPtr = colSumBufferHost.get();

  for(std::size_t i = 0; i < m; ++i) {
    colSumPtr[i] = {1, 0};
  }

  api::memcpy_async(aBuffer.get(), colSumBufferHost.get(), sizeof(ComplexType) * m,
                    api::flag::MemcpyHostToDevice, queue.stream());
  api::blas::trmv(queue.blas_handle(), api::blas::fill::lower, api::blas::operation::Transpose,
                  api::blas::diag::nonunit, m, a, lda, aBuffer.get(), 1);

  api::memcpy_async(colSumBufferHost.get(), aBuffer.get(), sizeof(ComplexType) * m,
                    api::flag::MemcpyDeviceToHost, queue.stream());

  queue.sync();

  auto indexBufferHost = queue.create_pinned_buffer<std::size_t>(m);
  auto indexBuffer = queue.create_device_buffer<std::size_t>(m);

  auto indexHostPtr = indexBufferHost.get();

  std::size_t mReduced = 0;
  for(std::size_t i = 0; i < m; ++i) {
    const auto sum = colSumPtr[i];
    if (sum.x * sum.x + sum.y * sum.y > 1e-8) {
      indexHostPtr[mReduced] = i;
      ++mReduced;
    }
  }

  if(false) {
    api::memcpy_2d_async(aBuffer.get(), m * sizeof(ComplexType), a, lda * sizeof(ComplexType),
                         m * sizeof(ComplexType), m, api::flag::MemcpyDeviceToDevice,
                         queue.stream());
  } else {
    api::memset_async(aBuffer.get(), 0, mReduced * mReduced * sizeof(ComplexType), queue.stream());
    api::memcpy_async(indexBuffer.get(), indexBufferHost.get(), sizeof(std::size_t) * mReduced,
                      api::flag::MemcpyHostToDevice, queue.stream());
    copy_matrix_from_indices(queue.device_prop(), queue.stream(), mReduced, indexBuffer.get(), a,
                             lda, aBuffer.get(), mReduced);
  }

  int hMeig = 0;

  const auto firstEigIndexFortran = mReduced - std::min(mReduced, nEig) + 1;
  if(b) {
    auto bBuffer = queue.create_device_buffer<ComplexType>(m * m);  

  if(m == mReduced) {
    api::memcpy_2d_async(bBuffer.get(), m * sizeof(ComplexType), b, ldb * sizeof(ComplexType),
                         m * sizeof(ComplexType), m, api::flag::MemcpyDeviceToDevice,
                         queue.stream());
  } else {
    api::memset_async(bBuffer.get(), 0, mReduced * mReduced * sizeof(ComplexType), queue.stream());
    copy_matrix_from_indices(queue.device_prop(), queue.stream(), mReduced, indexBuffer.get(), b,
                             ldb, bBuffer.get(), mReduced);
  }


    eigensolver::solve(ctx, 'V', 'I', 'L', mReduced, aBuffer.get(), mReduced, bBuffer.get(),
                       mReduced, 0, 0, firstEigIndexFortran, mReduced, &hMeig, dBuffer.get());
  } else{
    eigensolver::solve(ctx, 'V', 'I', 'L', mReduced, aBuffer.get(), mReduced, 0, 0,
                       firstEigIndexFortran, mReduced, &hMeig, dBuffer.get());
  }

  const auto nEigOut = std::min<std::size_t>(hMeig, nEig);

  if (nEigOut < nEig) api::memset_async(d, 0, nEig * sizeof(ScalarType), queue.stream());

  if (nEigOut < nEig || m != mReduced)
    api::memset_async(v, 0, nEig * m * sizeof(ComplexType), queue.stream());

  // copy results to output
  api::memcpy_async(d, dBuffer.get() + hMeig - nEigOut, nEigOut * sizeof(ScalarType),
                    api::flag::MemcpyDeviceToDevice, queue.stream());

  if (m == mReduced) {
    api::memcpy_2d_async(v, ldv * sizeof(ComplexType), aBuffer.get() + (hMeig - nEigOut) * m,
                         m * sizeof(ComplexType), m * sizeof(ComplexType), nEigOut,
                         api::flag::MemcpyDeviceToDevice, queue.stream());
  } else {
    copy_matrix_rows_to_indices(queue.device_prop(), queue.stream(), mReduced, nEigOut,
                                indexBuffer.get(), aBuffer.get() + (hMeig - nEigOut) * mReduced, mReduced,
                                v, ldv);
  }

  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvalues", nEig, 1, d, nEig);
  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", m, nEig, v, m);
}

template auto eigh<float>(ContextInternal& ctx, std::size_t m, std::size_t nEig,
                          const api::ComplexType<float>* a, std::size_t lda,
                          const api::ComplexType<float>* b, std::size_t ldb, float* d,
                          api::ComplexType<float>* v, std::size_t ldv) -> void;

template auto eigh<double>(ContextInternal& ctx, std::size_t m, std::size_t nEig,
                           const api::ComplexType<double>* a, std::size_t lda,
                           const api::ComplexType<double>* b, std::size_t ldb, double* d,
                           api::ComplexType<double>* v, std::size_t ldv) -> void;

}  // namespace gpu
}  // namespace bipp
