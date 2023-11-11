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
#include "memory/array.hpp"
#include "memory/copy.hpp"

namespace bipp {
namespace host {

template <typename T>
static auto copy_lower_triangle_at_indices(const std::vector<std::size_t>& indices,
                                           const ConstHostView<T, 2>& a, HostView<T, 2> b) {
  const std::size_t mReduced = indices.size();
  if (mReduced == a.shape(0)) {
    copy(a, b);
  } else {
    for (std::size_t col = 0; col < mReduced; ++col) {
      const auto colIdx = indices[col];
      auto bCol = b.slice_view(col);
      auto aCol = a.slice_view(colIdx);
      for (std::size_t row = col; row < mReduced; ++row) {
        const auto rowIdx = indices[row];
        bCol[{row}] = aCol[{rowIdx}];
      }
    }
  }
}

template <typename T>
auto eigh(ContextInternal& ctx, std::size_t nEig, const ConstHostView<std::complex<T>, 2>& a,
          const ConstHostView<std::complex<T>, 2>& b, HostView<T, 1> d,
          HostView<std::complex<T>, 2> v) -> void {
  const auto m = a.shape(0);

  std::vector<short> nonZeroIndexFlag(m, 0);

  // flag working coloumns / rows
  for (std::size_t col = 0; col < a.shape(1); ++col) {
    for (std::size_t row = col; row < a.shape(0); ++row) {
      const auto val = a[{row, col}];
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

  HostArray<std::complex<T>, 2> aReduced(ctx.host_alloc(), {mReduced, mReduced});
  HostArray<std::complex<T>, 2> vBuffer(ctx.host_alloc(), {mReduced, mReduced});
  HostArray<T, 1> dBuffer(ctx.host_alloc(), {mReduced});
  HostArray<int, 1> bufferIfail(ctx.host_alloc(), {mReduced});

  // copy lower triangle into buffer
  copy_lower_triangle_at_indices(indices, a, aReduced);

  const auto firstEigIndexFortran = mReduced - std::min(mReduced, nEig) + 1;

  int hMeig = 0;
  if (b.size()) {
    HostArray<std::complex<T>, 2> bReduced(ctx.host_alloc(), {mReduced, mReduced});
    copy_lower_triangle_at_indices(indices, b, bReduced);

    if (lapack::eigh_solve(LapackeLayout::COL_MAJOR, 1, 'V', 'I', 'L', mReduced, aReduced.data(),
                           aReduced.strides(1), bReduced.data(), bReduced.strides(1), 0, 0,
                           firstEigIndexFortran, mReduced, &hMeig, dBuffer.data(), vBuffer.data(),
                           vBuffer.strides(1), bufferIfail.data())) {
      throw EigensolverError();
    }
  } else {
    if (lapack::eigh_solve(LapackeLayout::COL_MAJOR, 'V', 'I', 'L', mReduced, aReduced.data(),
                           aReduced.strides(1), 0, 0, firstEigIndexFortran, mReduced, &hMeig,
                           dBuffer.data(), vBuffer.data(), vBuffer.strides(1),
                           bufferIfail.data())) {
      throw EigensolverError();
    }
  }

  const auto nEigOut = std::min<std::size_t>(hMeig, nEig);

  d.zero();
  v.zero();

  copy(dBuffer.sub_view({hMeig - nEigOut}, {nEigOut}), d.sub_view({0}, {nEigOut}));

  if (mReduced == m) {
    copy(vBuffer.sub_view({0, hMeig - nEigOut}, {m, nEigOut}), v.sub_view({0, 0}, {m, nEigOut}));
  } else {
    for (std::size_t col = 0; col < nEigOut; ++col) {
      auto sourceCol = vBuffer.slice_view(col + hMeig - nEigOut);
      for (std::size_t row = 0; row < mReduced; ++row) {
        v[{col, indices[row]}] = sourceCol[{row}];
      }
    }
  }

  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvalues", d);
  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", v);
}

template auto eigh<float>(ContextInternal& ctx, std::size_t nEig,
                          const ConstHostView<std::complex<float>, 2>& a,
                          const ConstHostView<std::complex<float>, 2>& b, HostView<float, 1> d,
                          HostView<std::complex<float>, 2> v) -> void;

template auto eigh<double>(ContextInternal& ctx, std::size_t nEig,
                           const ConstHostView<std::complex<double>, 2>& a,
                           const ConstHostView<std::complex<double>, 2>& b, HostView<double, 1> d,
                           HostView<std::complex<double>, 2> v) -> void;

}  // namespace host
}  // namespace bipp
