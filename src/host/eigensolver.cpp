#include "host/eigensolver.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "host/blas_api.hpp"
#include "host/lapack_api.hpp"
#include "memory/allocator.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace host {

template <typename T>
static auto copy_lower_triangle_at_indices(std::size_t m, const std::vector<std::size_t>& indices,
                                           const T* a, std::size_t lda, T* b, std::size_t ldb) {
  const std::size_t mReduced = indices.size();
  if (mReduced == m) {
    for (std::size_t col = 0; col < mReduced; ++col) {
      std::memcpy(b + col * ldb + col, a + col * lda + col, (mReduced - col) * sizeof(T));
    }
  } else {
    for (std::size_t col = 0; col < mReduced; ++col) {
      const auto colIdx = indices[col];
      for (std::size_t row = col; row < mReduced; ++row) {
        const auto rowIdx = indices[row];
        b[col * mReduced + row] = a[colIdx * lda + rowIdx];
      }
    }
  }
}

template <typename T>
auto eigh(ContextInternal& ctx, std::size_t m, std::size_t nEig, const std::complex<T>* a,
          std::size_t lda, const std::complex<T>* b, std::size_t ldb, T* d, std::complex<T>* v,
          std::size_t ldv) -> void {
  auto bufferA = Buffer<std::complex<T>>(ctx.host_alloc(), m * m);
  auto bufferV = Buffer<std::complex<T>>(ctx.host_alloc(), m * m);
  auto bufferD = Buffer<T>(ctx.host_alloc(), m);
  auto bufferIfail = Buffer<int>(ctx.host_alloc(), m);
  std::complex<T>* aReduced = bufferA.get();

  std::vector<short> nonZeroIndexFlag(m, 0);

  // flag working coloumns / rows
  for (std::size_t col = 0; col < m; ++col) {
    for (std::size_t row = col; row < m; ++row) {
      const auto val =  a[col * lda + row];
      if (val.real() != 0 || val.imag() != 0) {
        nonZeroIndexFlag[col] |= 1;
        nonZeroIndexFlag[row] |= 1;
      }
    }
  }

  std::vector<std::size_t> indices;
  indices.reserve(m);
  for (std::size_t i = 0; i < m; ++i) {
    if (nonZeroIndexFlag[i]) indices.push_back(i);
  }

  const std::size_t mReduced = indices.size();

  ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "Eigensolver: removing {} coloumns / rows", m - mReduced);

  // copy lower triangle into buffer
  copy_lower_triangle_at_indices(m, indices, a, lda, aReduced, mReduced);

  const auto firstEigIndexFortran = mReduced - std::min(mReduced, nEig) + 1;

  int hMeig = 0;
  if (b) {
    auto bufferB = Buffer<std::complex<T>>(ctx.host_alloc(), m * m);
    std::complex<T>* bReduced = bufferB.get();

    copy_lower_triangle_at_indices(m, indices, b, ldb, bReduced, mReduced);

    if (lapack::eigh_solve(LapackeLayout::COL_MAJOR, 1, 'V', 'I', 'L', mReduced, aReduced, mReduced,
                           bReduced, mReduced, 0, 0, firstEigIndexFortran, mReduced,
                           &hMeig, bufferD.get(), bufferV.get(), mReduced, bufferIfail.get())) {
      throw EigensolverError();
    }
  } else {
    if (lapack::eigh_solve(LapackeLayout::COL_MAJOR, 'V', 'I', 'L', mReduced, aReduced, mReduced, 0,
                           0, firstEigIndexFortran, mReduced, &hMeig, bufferD.get(),
                           bufferV.get(), mReduced, bufferIfail.get())) {
      throw EigensolverError();
    }
  }

  const auto nEigOut = std::min<std::size_t>(hMeig, nEig);

  auto bufferPtrD = bufferD.get();
  auto bufferPtrV = bufferV.get();

  std::memset(d, 0, nEig * sizeof(T));
  for (std::size_t col = 0; col < nEig; ++col) {
    std::memset(v + col * ldv, 0, m * sizeof(std::complex<T>));
  }


  // copy in reverse order into output and pad to full size
  for (std::size_t col = 0; col < nEigOut; ++col) {
    d[col] = bufferPtrD[col + hMeig - nEigOut];
  }

  if (mReduced == m) {
  for (std::size_t col = 0; col < nEigOut; ++col) {
    std::memcpy(v + col * ldv, bufferPtrV + (col + hMeig - nEigOut) * m,
                m * sizeof(std::complex<T>));
    }
  } else {
    for (std::size_t col = 0; col < nEigOut; ++col) {
      const auto colPtr = bufferPtrV + (col + hMeig - nEigOut) * mReduced;
      for (std::size_t row = 0; row < mReduced; ++row) {
        v[col * ldv + indices[row]] = colPtr[row];
      }
    }
  }

  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvalues", nEig, 1, d, nEig);
  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", m, nEig, v, m);
}

template auto eigh<float>(ContextInternal& ctx, std::size_t m, std::size_t nEig,
                          const std::complex<float>* a, std::size_t lda,
                          const std::complex<float>* b, std::size_t ldb, float* d,
                          std::complex<float>* v, std::size_t ldv) -> void;

template auto eigh<double>(ContextInternal& ctx, std::size_t m, std::size_t nEig,
                           const std::complex<double>* a, std::size_t lda,
                           const std::complex<double>* b, std::size_t ldb, double* d,
                           std::complex<double>* v, std::size_t ldv) -> void;
}  // namespace host
}  // namespace bipp
