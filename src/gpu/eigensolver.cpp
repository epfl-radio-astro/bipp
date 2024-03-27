#include "gpu/eigensolver.hpp"

#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/gram_matrix.hpp"
#include "gpu/kernels//copy_at_indices.hpp"
#include "gpu/util/blas_api.hpp"
#include "gpu/util/runtime_api.hpp"
#include "gpu/util/solver_api.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace gpu {
template <typename T>
auto eigh(ContextInternal& ctx, T wl, ConstDeviceView<api::ComplexType<T>, 2> s,
          ConstDeviceView<api::ComplexType<T>, 2> w, ConstDeviceView<T, 2> xyz, DeviceView<T, 1> d,
          DeviceView<api::ComplexType<T>, 2> vUnbeam) -> std::pair<std::size_t, std::size_t> {
  const auto nAntenna = w.shape(0);
  const auto nBeam = w.shape(1);
  auto& queue = ctx.gpu_queue();

  assert(xyz.shape(0) == nAntenna);
  assert(xyz.shape(1) == 3);
  assert(s.shape(0) == nBeam);
  assert(s.shape(1) == nBeam);
  assert(!vUnbeam.size() || vUnbeam.shape(0) == nAntenna);
  assert(!vUnbeam.size() || vUnbeam.shape(1) == nBeam);

  HostArray<short, 1> nonZeroIndexFlag(ctx.host_alloc(), nBeam);
  nonZeroIndexFlag.zero();

  auto sHost = queue.create_pinned_array<api::ComplexType<T>, 2>(s.shape());
  copy(queue, s, sHost);
  queue.sync();

  // flag working columns / rows
  for (std::size_t col = 0; col < s.shape(1); ++col) {
    for (std::size_t row = col; row < s.shape(0); ++row) {
      const auto val = sHost[{row, col}];
      if (val.x != 0 || val.y != 0) {
        nonZeroIndexFlag[col] |= 1;
        nonZeroIndexFlag[row] |= 1;
      }
    }
  }

  // Discard rows / columns with sum close to zero
  for (std::size_t col = 0; col < s.shape(1); ++col) {
    auto sum = sHost[{col, col}];
    for (std::size_t row = col+1; row < s.shape(0); ++row) {
      const auto val = sHost[{row, col}];
      sum.x += val.x;
      sum.y += val.y;
    }
    auto const row = col;
    for (std::size_t col = 0; col < row; ++col) {
      const auto val = sHost[{row, col}];
      sum.x += val.x;
      sum.y += -val.y; // Use conjugate in sum!
    }
    auto norm = std::sqrt(sum.x * sum.x + sum.y * sum.y);
    if (norm <= std::numeric_limits<T>::epsilon()) {
      ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "Eigensolver: removing column {} / row {} with ||sum|| = {}", col, row, norm);
      nonZeroIndexFlag[col] = 0;
    }
  }

  // Count the number of non-zero visibilities
  std::size_t nVis = {0};
  const std::complex<T> c0 = 0.0;
  nVis = s.shape(0) * s.shape(1);
  for (std::size_t col = 0; col < s.shape(1); ++col) {
    for (std::size_t row = col; row < s.shape(0); ++row) {
      auto val = sHost[{row, col}];
      if ((val.x == 0 && val.y == 0) || nonZeroIndexFlag[row] == 0 || nonZeroIndexFlag[col] == 0) {
        col == row ? nVis -= 1 : nVis -= 2;
      }
    }
  }
  ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "eigensolver (gpu) nVis = {}", nVis);


  std::vector<std::size_t> indices;
  indices.reserve(nBeam);
  for (std::size_t i = 0; i < nBeam; ++i) {
    if (nonZeroIndexFlag[i]) indices.push_back(i);
  }

  const std::size_t nBeamReduced = indices.size();

  ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "Eigensolver: removing {} columns / rows", nBeam - nBeamReduced);
  api::memset_async(d.data(), 0, d.size() * sizeof(T), queue.stream());
  api::memset_2d_async(vUnbeam.data(), vUnbeam.strides(1) * sizeof(api::ComplexType<T>), 0,
                       vUnbeam.shape(0), vUnbeam.shape(1), queue.stream());

  auto v = queue.create_device_array<api::ComplexType<T>, 2>({nBeamReduced, nBeamReduced});

  const char mode = vUnbeam.size() ? 'V' : 'N';
  const api::ComplexType<T> one{1, 0};
  const api::ComplexType<T> zero{0, 0};

  if(nBeamReduced == nBeam) {
    copy(queue, s,v);

    // Compute gram matrix
    auto g = queue.create_device_array<api::ComplexType<T>, 2>({nBeam, nBeam});
    gram_matrix<T>(ctx, w, xyz, wl, g);

    eigensolver::solve(queue, mode, 'L', nBeam, v.data(), v.strides(1), g.data(), g.strides(1),
                       d.data());

    if (vUnbeam.size())
      api::blas::gemm<api::ComplexType<T>>(queue.blas_handle(), api::blas::operation::None,
                                           api::blas::operation::None, one, w, v, zero, vUnbeam);
  } else {
    // Remove broken beams from w and s
    auto wReduced = queue.create_device_array<api::ComplexType<T>, 2>({nAntenna, nBeamReduced});
    auto indicesDevice = queue.create_device_array<std::size_t, 1>(indices.size());
    copy(queue, ConstHostView<std::size_t, 1>(indices.data(), indices.size(), 1), indicesDevice);

    copy_matrix_from_indices(queue.device_prop(), queue.stream(), nBeamReduced, indicesDevice.data(),
                             s.data(), s.strides(1), v.data(), v.strides(1));

    for(std::size_t i =0; i < nBeamReduced; ++i) {
      copy(queue, w.slice_view(indices[i]), wReduced.slice_view(i));
    }

    // Compute gram matrix
    auto gReduced = queue.create_device_array<api::ComplexType<T>, 2>({nBeamReduced, nBeamReduced});
    gram_matrix<T>(ctx, wReduced, xyz, wl, gReduced);

    eigensolver::solve(queue, mode, 'L', nBeamReduced, v.data(), v.strides(1), gReduced.data(),
                       gReduced.strides(1), d.data());

    if (vUnbeam.size())
      api::blas::gemm<api::ComplexType<T>>(queue.blas_handle(), api::blas::operation::None,
                                           api::blas::operation::None, one, wReduced, v, zero,
                                           vUnbeam.sub_view({0, 0}, {nAntenna, nBeamReduced}));
  }

  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvalues", d.sub_view(0, nBeamReduced));
  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", v);

  return std::make_pair(nBeamReduced, nVis);
}

template auto eigh<float>(ContextInternal& ctx, float wl,
                          ConstDeviceView<api::ComplexType<float>, 2> s,
                          ConstDeviceView<api::ComplexType<float>, 2> w,
                          ConstDeviceView<float, 2> xyz, DeviceView<float, 1> d,
                          DeviceView<api::ComplexType<float>, 2> vUnbeam) -> std::pair<std::size_t, std::size_t>;

template auto eigh<double>(ContextInternal& ctx, double wl,
                           ConstDeviceView<api::ComplexType<double>, 2> s,
                           ConstDeviceView<api::ComplexType<double>, 2> w,
                           ConstDeviceView<double, 2> xyz, DeviceView<double, 1> d,
                           DeviceView<api::ComplexType<double>, 2> vUnbeam) -> std::pair<std::size_t, std::size_t>;
}  // namespace gpu
}  // namespace bipp
