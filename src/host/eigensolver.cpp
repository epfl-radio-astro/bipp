#include "host/eigensolver.hpp"

#include <algorithm>
#include <cassert>
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
#include "host/gram_matrix.hpp"
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
auto eigh(ContextInternal& ctx, T wl, ConstHostView<std::complex<T>, 2> s,
          ConstHostView<std::complex<T>, 2> w, ConstHostView<T, 2> xyz, HostView<T, 1> d,
          HostView<std::complex<T>, 2> vUnbeam) -> std::pair<std::size_t, std::size_t> {
  const auto nAntenna = w.shape(0);
  const auto nBeam = w.shape(1);

  assert(xyz.shape(0) == nAntenna);
  assert(xyz.shape(1) == 3);
  assert(s.shape(0) == nBeam);
  assert(s.shape(1) == nBeam);
  assert(!vUnbeam.size() || vUnbeam.shape(0) == nAntenna);
  assert(!vUnbeam.size() || vUnbeam.shape(1) == nBeam);

  HostArray<short, 1> nonZeroIndexFlag(ctx.host_alloc(), nBeam);
  nonZeroIndexFlag.zero();

  // flag working columns / rows
  std::size_t nVis = 0;
  for (std::size_t col = 0; col < s.shape(1); ++col) {
    for (std::size_t row = col; row < s.shape(0); ++row) {
      const auto val = s[{row, col}];
      if (std::abs(val.real()) >= std::numeric_limits<T>::epsilon() ||
          std::abs(val.imag()) >= std::numeric_limits<T>::epsilon()) {
        nonZeroIndexFlag[col] |= 1;
        nonZeroIndexFlag[row] |= 1;
        nVis += 1 + (row != col);
      }
    }
  }
  ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "eigensolver (host) nVis = {}", nVis);

  std::vector<std::size_t> indices;
  indices.reserve(nBeam);
  for (std::size_t i = 0; i < nBeam; ++i) {
    if (nonZeroIndexFlag[i]) indices.push_back(i);
  }

  const std::size_t nBeamReduced = indices.size();

  ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "Eigensolver: removing {} columns / rows", nBeam - nBeamReduced);

  d.zero();
  vUnbeam.zero();

  HostArray<std::complex<T>, 2> v(ctx.host_alloc(), {nBeamReduced, nBeamReduced});

  const char mode = vUnbeam.size() ? 'V' : 'N';

  if (nBeamReduced == nBeam) {
    copy(s, v);

    // Compute gram matrix
    auto g = HostArray<std::complex<T>, 2>(ctx.host_alloc(), {nBeam, nBeam});
    gram_matrix<T>(ctx, w, xyz, wl, g);

    lapack::eigh_solve(LapackeLayout::COL_MAJOR, 1, mode, 'L', nBeam, v.data(), v.strides(1),
                       g.data(), g.strides(1), d.data());

    if (vUnbeam.size())
      blas::gemm<std::complex<T>>(CblasNoTrans, CblasNoTrans, {1, 0}, w, v, {0, 0}, vUnbeam);
  } else {
    // Remove broken beams from w and s
    HostArray<std::complex<T>, 2> wReduced(ctx.host_alloc(), {nAntenna, nBeamReduced});

    copy_lower_triangle_at_indices(indices, s, v);

    for(std::size_t i =0; i < nBeamReduced; ++i) {
      copy(w.slice_view(indices[i]), wReduced.slice_view(i));
    }

    // Compute gram matrix
    auto gReduced = HostArray<std::complex<T>, 2>(ctx.host_alloc(), {nBeamReduced, nBeamReduced});
    gram_matrix<T>(ctx, wReduced, xyz, wl, gReduced);

    lapack::eigh_solve(LapackeLayout::COL_MAJOR, 1, mode, 'L', nBeamReduced, v.data(), v.strides(1),
                       gReduced.data(), gReduced.strides(1), d.data());

    if (vUnbeam.size())
      blas::gemm<std::complex<T>>(CblasNoTrans, CblasNoTrans, {1, 0}, wReduced, v, {0, 0},
                                  vUnbeam.sub_view({0, 0}, {nAntenna, nBeamReduced}));
  }

  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvalues", d.sub_view(0, nBeamReduced));
  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", v);

  return std::make_pair(nBeamReduced, nVis);
}

template auto eigh<float>(ContextInternal& ctx, float wl, ConstHostView<std::complex<float>, 2> s,
                          ConstHostView<std::complex<float>, 2> w, ConstHostView<float, 2> xyz,
                          HostView<float, 1> d, HostView<std::complex<float>, 2> vUnbeam)
    -> std::pair<std::size_t, std::size_t>;

template auto eigh<double>(ContextInternal& ctx, double wl,
                           ConstHostView<std::complex<double>, 2> s,
                           ConstHostView<std::complex<double>, 2> w, ConstHostView<double, 2> xyz,
                           HostView<double, 1> d, HostView<std::complex<double>, 2> vUnbeam)
    -> std::pair<std::size_t, std::size_t>;

}  // namespace host
}  // namespace bipp
