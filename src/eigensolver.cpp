#include "host/eigensolver.hpp"

#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

#include "bipp/bipp.hpp"
#include "bipp/config.h"
#include "bipp/context.hpp"
#include "context_internal.hpp"
#include "host/blas_api.hpp"
#include "host/gram_matrix.hpp"
#include "host/lapack_api.hpp"
#include "memory/view.hpp"
#include "memory/copy.hpp"
#include "memory/array.hpp"

namespace bipp {

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

template <typename T, typename>
BIPP_EXPORT auto eigh(T wl, std::size_t nAntenna, std::size_t nBeam, const std::complex<T>* s,
                      std::size_t lds, const std::complex<T>* w, std::size_t ldw, T* d,
                      std::complex<T>* v, std::size_t ldv) -> std::pair<std::size_t, T> {
  auto funcTimer = globLogger.scoped_timing(BIPP_LOG_LEVEL_INFO, "host::eigh");

  ConstHostView<std::complex<T>, 2> sView(s, {nBeam, nBeam}, {1, lds});

  ConstHostView<std::complex<T>, 2> wView(w, {nAntenna, nBeam}, {1, ldw});
  HostView<T, 1> dView(d, nBeam, 1);
  HostView<std::complex<T>, 2> vUnbeam(v, {nAntenna, nBeam}, {1, ldv});

  std::shared_ptr<Allocator> alloc = AllocatorFactory::simple_host();

  HostArray<short, 1> nonZeroIndexFlag(alloc, nBeam);
  nonZeroIndexFlag.zero();

  // flag working coloumns / rows
  std::size_t nVis = 0;
  for (std::size_t col = 0; col < sView.shape(1); ++col) {
    for (std::size_t row = col; row < sView.shape(0); ++row) {
      const auto val = sView[{row, col}];
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

  globLogger.log(BIPP_LOG_LEVEL_DEBUG, "Eigensolver: removing {} columns / rows",
                 nBeam - nBeamReduced);

  dView.zero();
  vUnbeam.zero();

  HostArray<std::complex<T>, 2> vArray(alloc, {nBeamReduced, nBeamReduced});

  const char mode = vUnbeam.size() ? 'V' : 'N';

  if (nBeamReduced == nBeam) {
    copy(sView, vArray);

    globLogger.start_timing(BIPP_LOG_LEVEL_INFO, "lapack solve");
    host::lapack::eigh_solve(LapackeLayout::COL_MAJOR, mode, 'L', nBeam, vArray.data(),
                             vArray.strides(1), dView.data());
    globLogger.stop_timing(BIPP_LOG_LEVEL_INFO, "lapack solve");

    if (vUnbeam.size())
      host::blas::gemm<std::complex<T>>(CblasNoTrans, CblasNoTrans, {1, 0}, wView, vArray, {0, 0},
                                        vUnbeam);
  } else {
    // Remove broken beams from wView and sView
    HostArray<std::complex<T>, 2> wReduced(alloc, {nAntenna, nBeamReduced});

    copy_lower_triangle_at_indices(indices, sView, vArray);

    for (std::size_t i = 0; i < nBeamReduced; ++i) {
      copy(wView.slice_view(indices[i]), wReduced.slice_view(i));
    }

    globLogger.start_timing(BIPP_LOG_LEVEL_INFO, "lapack solve");
    host::lapack::eigh_solve(LapackeLayout::COL_MAJOR, mode, 'L', nBeamReduced, vArray.data(),
                             vArray.strides(1), dView.data());
    globLogger.stop_timing(BIPP_LOG_LEVEL_INFO, "lapack solve");

    if (vUnbeam.size())
      host::blas::gemm<std::complex<T>>(CblasNoTrans, CblasNoTrans, {1, 0}, wReduced, vArray,
                                        {0, 0}, vUnbeam.sub_view({0, 0}, {nAntenna, nBeamReduced}));
  }

  globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvalues", dView.sub_view(0, nBeamReduced));
  globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", vArray);

  const T scalingFactor = nVis ? T(1) / T(nVis) : T(0);

  return std::make_pair(nBeamReduced, scalingFactor);
}

template <typename T, typename>
BIPP_EXPORT auto eigh_gram(T wl, std::size_t nAntenna, std::size_t nBeam, const std::complex<T>* s,
                           std::size_t lds, const std::complex<T>* w, std::size_t ldw, const T* xyz,
                           std::size_t ldxyz, T* d, std::complex<T>* v, std::size_t ldv)
    -> std::pair<std::size_t, T> {
  auto funcTimer = globLogger.scoped_timing(BIPP_LOG_LEVEL_INFO, "host::eigh");

  ConstHostView<std::complex<T>, 2> sView(s, {nBeam, nBeam}, {1, lds});

  ConstHostView<std::complex<T>, 2> wView(w, {nAntenna, nBeam}, {1, ldw});
  ConstHostView<T, 2> xyzView(xyz, {nAntenna, 3}, {1, ldxyz});
  HostView<T, 1> dView(d, nBeam, 1);
  HostView<std::complex<T>, 2> vUnbeam(v, {nAntenna, nBeam}, {1, ldv});

  std::shared_ptr<Allocator> alloc = AllocatorFactory::simple_host();

  HostArray<short, 1> nonZeroIndexFlag(alloc, nBeam);
  nonZeroIndexFlag.zero();

  // flag working coloumns / rows
  std::size_t nVis = 0;
  for (std::size_t col = 0; col < sView.shape(1); ++col) {
    for (std::size_t row = col; row < sView.shape(0); ++row) {
      const auto val = sView[{row, col}];
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

  globLogger.log(BIPP_LOG_LEVEL_DEBUG, "Eigensolver: removing {} columns / rows",
                 nBeam - nBeamReduced);

  dView.zero();
  vUnbeam.zero();

  HostArray<std::complex<T>, 2> vArray(alloc, {nBeamReduced, nBeamReduced});

  const char mode = vUnbeam.size() ? 'V' : 'N';

  if (nBeamReduced == nBeam) {
    copy(sView, vArray);

    // Compute gram matrix
    auto g = HostArray<std::complex<T>, 2>(alloc, {nBeam, nBeam});
    host::gram_matrix<T>(alloc, wView, xyzView, wl, g);

    globLogger.start_timing(BIPP_LOG_LEVEL_INFO, "lapack solve");
    host::lapack::eigh_solve(LapackeLayout::COL_MAJOR, 1, mode, 'L', nBeam, vArray.data(),
                             vArray.strides(1), g.data(), g.strides(1), dView.data());
    globLogger.stop_timing(BIPP_LOG_LEVEL_INFO, "lapack solve");

    if (vUnbeam.size())
      host::blas::gemm<std::complex<T>>(CblasNoTrans, CblasNoTrans, {1, 0}, wView, vArray, {0, 0},
                                        vUnbeam);
  } else {
    // Remove broken beams from wView and sView
    HostArray<std::complex<T>, 2> wReduced(alloc, {nAntenna, nBeamReduced});

    copy_lower_triangle_at_indices(indices, sView, vArray);

    for (std::size_t i = 0; i < nBeamReduced; ++i) {
      copy(wView.slice_view(indices[i]), wReduced.slice_view(i));
    }

    // Compute gram matrix
    auto gReduced = HostArray<std::complex<T>, 2>(alloc, {nBeamReduced, nBeamReduced});
    host::gram_matrix<T>(alloc, wReduced, xyzView, wl, gReduced);

    globLogger.start_timing(BIPP_LOG_LEVEL_INFO, "lapack solve");
    host::lapack::eigh_solve(LapackeLayout::COL_MAJOR, 1, mode, 'L', nBeamReduced, vArray.data(),
                             vArray.strides(1), gReduced.data(), gReduced.strides(1), dView.data());
    globLogger.stop_timing(BIPP_LOG_LEVEL_INFO, "lapack solve");

    if (vUnbeam.size())
      host::blas::gemm<std::complex<T>>(CblasNoTrans, CblasNoTrans, {1, 0}, wReduced, vArray,
                                        {0, 0}, vUnbeam.sub_view({0, 0}, {nAntenna, nBeamReduced}));
  }

  globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvalues", dView.sub_view(0, nBeamReduced));
  globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", vArray);

  const T scalingFactor = nVis ? T(1) / T(nVis) : T(0);

  return std::make_pair(nBeamReduced, scalingFactor);
}

template BIPP_EXPORT auto eigh<float, void>(float wl, std::size_t nAntenna, std::size_t nBeam,
                                            const std::complex<float>* s, std::size_t lds,
                                            const std::complex<float>* w, std::size_t ldw, float* d,
                                            std::complex<float>* v, std::size_t ldv)
    -> std::pair<std::size_t, float>;

template BIPP_EXPORT auto eigh<double, void>(double wl, std::size_t nAntenna, std::size_t nBeam,
                                             const std::complex<double>* s, std::size_t lds,
                                             const std::complex<double>* w, std::size_t ldw,
                                             double* d, std::complex<double>* v, std::size_t ldv)
    -> std::pair<std::size_t, double>;

template BIPP_EXPORT auto eigh_gram<float, void>(float wl, std::size_t nAntenna, std::size_t nBeam,
                                                 const std::complex<float>* s, std::size_t lds,
                                                 const std::complex<float>* w, std::size_t ldw,
                                                 const float* xyz, std::size_t ldxyz, float* d,
                                                 std::complex<float>* v, std::size_t ldv)
    -> std::pair<std::size_t, float>;

template BIPP_EXPORT auto eigh_gram<double, void>(double wl, std::size_t nAntenna,
                                                  std::size_t nBeam, const std::complex<double>* s,
                                                  std::size_t lds, const std::complex<double>* w,
                                                  std::size_t ldw, const double* xyz,
                                                  std::size_t ldxyz, double* d,
                                                  std::complex<double>* v, std::size_t ldv)
    -> std::pair<std::size_t, double>;

}  // namespace bipp
