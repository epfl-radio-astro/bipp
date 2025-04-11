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

#include "bipp/config.h"
#include "context_internal.hpp"
#include "host/blas_api.hpp"
#include "host/gram_matrix.hpp"
#include "host/lapack_api.hpp"
#include "memory/allocator.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/allocator_factory.hpp"

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
auto eigh(T wl, ConstHostView<std::complex<T>, 2> s, ConstHostView<std::complex<T>, 2> w,
          ConstHostView<T, 2> xyz, HostView<T, 1> d, HostView<std::complex<T>, 2> vUnbeam)
    -> std::pair<std::size_t, T> {
  auto funcTimer = globLogger.scoped_timing(BIPP_LOG_LEVEL_INFO, "host::eigh");
  const auto nAntenna = w.shape(0);
  const auto nBeam = w.shape(1);

  std::shared_ptr<Allocator> alloc = AllocatorFactory::simple_host();

  assert(xyz.shape(0) == nAntenna);
  assert(xyz.shape(1) == 3);
  assert(s.shape(0) == nBeam);
  assert(s.shape(1) == nBeam);
  assert(!vUnbeam.size() || vUnbeam.shape(0) == nAntenna);
  assert(!vUnbeam.size() || vUnbeam.shape(1) == nBeam);

  HostArray<short, 1> nonZeroIndexFlag(alloc, nBeam);
  nonZeroIndexFlag.zero();

  // flag working coloumns / rows
  std::size_t nVis = 0;
  for (std::size_t col = 0; col < s.shape(1); ++col) {
    for (std::size_t row = col; row < s.shape(0); ++row) {
      const auto val = s[{row, col}];
      if (std::norm(val) > std::numeric_limits<T>::epsilon()) {
        nonZeroIndexFlag[col] |= 1;
        nonZeroIndexFlag[row] |= 1;
        nVis += 1 + (row != col);
      }
    }
  }
  globLogger.log(BIPP_LOG_LEVEL_DEBUG, "eigensolver (host) nVis = {}", nVis);

  std::vector<std::size_t> indices;
  indices.reserve(nBeam);
  for (std::size_t i = 0; i < nBeam; ++i) {
    if (nonZeroIndexFlag[i]) indices.push_back(i);
  }

  const std::size_t nBeamReduced = indices.size();

  globLogger.log(BIPP_LOG_LEVEL_DEBUG, "Eigensolver: removing {} columns / rows", nBeam - nBeamReduced);

  d.zero();
  vUnbeam.zero();

  HostArray<std::complex<T>, 2> v(alloc, {nBeamReduced, nBeamReduced});

  const char mode = vUnbeam.size() ? 'V' : 'N';

  if (nBeamReduced == nBeam) {
    copy(s, v);

    // Compute gram matrix
    auto g = HostArray<std::complex<T>, 2>(alloc, {nBeam, nBeam});
    gram_matrix<T>(alloc, w, xyz, wl, g);

    globLogger.start_timing(BIPP_LOG_LEVEL_INFO, "lapack solve");
    lapack::eigh_solve(LapackeLayout::COL_MAJOR, 1, mode, 'L', nBeam, v.data(), v.strides(1),
                       g.data(), g.strides(1), d.data());
    globLogger.stop_timing(BIPP_LOG_LEVEL_INFO, "lapack solve");

    if (vUnbeam.size())
      blas::gemm<std::complex<T>>(CblasNoTrans, CblasNoTrans, {1, 0}, w, v, {0, 0}, vUnbeam);
  } else {
    // Remove broken beams from w and s
    HostArray<std::complex<T>, 2> wReduced(alloc, {nAntenna, nBeamReduced});

    copy_lower_triangle_at_indices(indices, s, v);

    for(std::size_t i =0; i < nBeamReduced; ++i) {
      copy(w.slice_view(indices[i]), wReduced.slice_view(i));
    }

    // Compute gram matrix
    auto gReduced = HostArray<std::complex<T>, 2>(alloc, {nBeamReduced, nBeamReduced});
    gram_matrix<T>(alloc, wReduced, xyz, wl, gReduced);

    globLogger.start_timing(BIPP_LOG_LEVEL_INFO, "lapack solve");
    lapack::eigh_solve(LapackeLayout::COL_MAJOR, 1, mode, 'L', nBeamReduced, v.data(), v.strides(1),
                       gReduced.data(), gReduced.strides(1), d.data());
    globLogger.stop_timing(BIPP_LOG_LEVEL_INFO, "lapack solve");

    if (vUnbeam.size())
      blas::gemm<std::complex<T>>(CblasNoTrans, CblasNoTrans, {1, 0}, wReduced, v, {0, 0},
                                  vUnbeam.sub_view({0, 0}, {nAntenna, nBeamReduced}));
  }

  globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvalues", d.sub_view(0, nBeamReduced));
  globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", v);

  const T scalingFactor = nVis ? T(1) / T(nVis) : T(0);

  return std::make_pair(nBeamReduced, scalingFactor);
}

template auto eigh<float>(float wl, ConstHostView<std::complex<float>, 2> s,
                          ConstHostView<std::complex<float>, 2> w, ConstHostView<float, 2> xyz,
                          HostView<float, 1> d, HostView<std::complex<float>, 2> vUnbeam)
    -> std::pair<std::size_t, float>;

template auto eigh<double>(double wl, ConstHostView<std::complex<double>, 2> s,
                           ConstHostView<std::complex<double>, 2> w, ConstHostView<double, 2> xyz,
                           HostView<double, 1> d, HostView<std::complex<double>, 2> vUnbeam)
    -> std::pair<std::size_t, double>;

}  // namespace host
}  // namespace bipp
