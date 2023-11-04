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
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto eigh(ContextInternal& ctx, std::size_t nEig, ConstHostView<api::ComplexType<T>, 2> aHost,
          ConstDeviceView<api::ComplexType<T>, 2> a, ConstDeviceView<api::ComplexType<T>, 2> b,
          DeviceView<T, 1> d, DeviceView<api::ComplexType<T>, 2> v) -> void {
  const auto m = a.shape()[0];
  auto& queue = ctx.gpu_queue();

  // flag working coloumns / rows
  std::vector<short> nonZeroIndexFlag(m, 0);
  for (std::size_t col = 0; col < m; ++col) {
    for (std::size_t row = col; row < m; ++row) {
      const auto val = aHost[{row, col}];
      if (val.x != 0 || val.y != 0) {
        nonZeroIndexFlag[col] |= 1;
        nonZeroIndexFlag[row] |= 1;
      }
    }
  }

  auto indicesHost = queue.create_pinned_array<std::size_t, 1>({m});

  std::size_t mReduced = 0;

  for (std::size_t i = 0; i < m; ++i) {
    if (nonZeroIndexFlag[i]) {
      indicesHost[{mReduced}] = i;
      ++mReduced;
    }
  }
  DeviceArray<std::size_t, 1> indicesDevice;

  ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "Eigensolver: removing {} coloumns / rows", m - mReduced);

  auto aBuffer = queue.create_device_array<api::ComplexType<T>, 2>({mReduced, mReduced});
  auto dBuffer = queue.create_device_array<T, 1>({mReduced});

  if (m == mReduced) {
    copy(queue, a, aBuffer);
  } else {
    indicesDevice = queue.create_device_array<std::size_t, 1>(indicesHost.shape());
    copy(queue, indicesHost, indicesDevice);
    copy_matrix_from_indices(queue.device_prop(), queue.stream(), mReduced, indicesDevice.data(),
                             a.data(), a.strides()[1], aBuffer.data(), aBuffer.strides()[1]);
  }

  int hMeig = 0;

  const auto firstEigIndexFortran = mReduced - std::min(mReduced, nEig) + 1;
  if (b.size()) {
    auto bBuffer = queue.create_device_array<api::ComplexType<T>, 2>({mReduced, mReduced});

    if (m == mReduced) {
      copy(queue, b, bBuffer);
    } else {
      copy_matrix_from_indices(queue.device_prop(), queue.stream(), mReduced, indicesDevice.data(),
                               b.data(), b.strides()[1], bBuffer.data(), bBuffer.strides()[1]);
    }

    eigensolver::solve(ctx, 'V', 'I', 'L', mReduced, aBuffer.data(), aBuffer.strides()[1],
                       bBuffer.data(), bBuffer.strides()[1], 0, 0, firstEigIndexFortran, mReduced,
                       &hMeig, dBuffer.data());
  } else {
    eigensolver::solve(ctx, 'V', 'I', 'L', mReduced, aBuffer.data(), aBuffer.strides()[1], 0, 0,
                       firstEigIndexFortran, mReduced, &hMeig, dBuffer.data());
  }

  const auto nEigOut = std::min<std::size_t>(hMeig, nEig);

  if (nEigOut < nEig) api::memset_async(d.data(), 0, d.size() * sizeof(T), queue.stream());

  if (nEigOut < nEig || m != mReduced) {
    for (std::size_t i = 0; i < v.shape()[1]; ++i)
      api::memset_async(v.slice_view(i).data(), 0, v.shape()[0] * sizeof(api::ComplexType<T>),
                        queue.stream());
  }

  // copy results to output
  copy(queue, dBuffer.sub_view({hMeig - nEigOut}, {nEigOut}), d.sub_view({0}, {nEigOut}));

  if (m == mReduced) {
    copy(queue, aBuffer.sub_view({0, hMeig - nEigOut}, {mReduced, nEigOut}),
         v.sub_view({0, 0}, {mReduced, nEigOut}));
  } else {
    copy_matrix_rows_to_indices(queue.device_prop(), queue.stream(), mReduced, nEigOut,
                                indicesDevice.data(), aBuffer.slice_view(hMeig - nEigOut).data(),
                                aBuffer.strides()[1], v.data(), v.strides()[1]);
  }

  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvalues", d);
  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", v);
}

template auto eigh<float>(ContextInternal& ctx, std::size_t nEig,
                          ConstHostView<api::ComplexType<float>, 2> aHost,
                          ConstDeviceView<api::ComplexType<float>, 2> a,
                          ConstDeviceView<api::ComplexType<float>, 2> b, DeviceView<float, 1> d,
                          DeviceView<api::ComplexType<float>, 2> v) -> void;

template auto eigh<double>(ContextInternal& ctx, std::size_t nEig,
                           ConstHostView<api::ComplexType<double>, 2> aHost,
                           ConstDeviceView<api::ComplexType<double>, 2> a,
                           ConstDeviceView<api::ComplexType<double>, 2> b, DeviceView<double, 1> d,
                           DeviceView<api::ComplexType<double>, 2> v) -> void;

}  // namespace gpu
}  // namespace bipp
