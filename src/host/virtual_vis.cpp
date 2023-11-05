#include "host/virtual_vis.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <tuple>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "host/blas_api.hpp"
#include "host/kernels/apply_filter.hpp"
#include "host/kernels/interval_indices.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace host {

template <typename T>
auto virtual_vis(ContextInternal& ctx, ConstHostView<BippFilter, 1> filter,
                 ConstHostView<T, 2> intervals, ConstHostView<T, 1> d,
                 ConstHostView<std::complex<T>, 2> v, HostView<std::complex<T>, 3> virtVis)
    -> void {
  assert(filter.size() == virtVis.shape()[2]);
  assert(intervals.shape()[1] == virtVis.shape()[1]);
  assert(v.shape()[0] * v.shape()[0] == virtVis.shape()[0]);
  assert(v.shape()[1] == d.size());

  const auto nAntenna = v.shape()[0];

  auto vScaled = HostArray<std::complex<T>, 2>(ctx.host_alloc(), v.shape());
  auto dFiltered = HostArray<T, 1>(ctx.host_alloc(), d.shape());

  for (std::size_t i = 0; i < virtVis.shape()[2]; ++i) {
    apply_filter(filter[{i}], d.size(), d.data(), dFiltered.data());

    for (std::size_t j = 0; j < virtVis.shape()[1]; ++j) {
      std::size_t start, size;
      std::tie(start, size) =
          find_interval_indices(d.size(), d.data(), intervals[{0, j}], intervals[{1, j}]);

      auto virtVisCurrent = virtVis.slice_view(i).slice_view(j);
      if (size) {
        // Multiply each col of v with the selected eigenvalue
        auto vScaledCurrent = vScaled.sub_view({0, 0}, {vScaled.shape()[0], size});
        auto vCurrent = v.sub_view({0, start}, {v.shape()[0], size});
        for (std::size_t k = 0; k < size; ++k) {
          const auto dVal = dFiltered[{start + k}];
          auto* __restrict__ vScaledPtr = &vScaledCurrent[{0, k}];
          const auto* __restrict__ vPtr = &vCurrent[{0,k}];

          ctx.logger().log(BIPP_LOG_LEVEL_DEBUG,
                           "Assigning eigenvalue {} (filtered {}) to inverval [{}, {}]",
                           d[{start + k}], dVal, intervals[{0, j}], intervals[{1, j}]);
          for (std::size_t l = 0; l < v.shape()[0]; ++l) {
            vScaledPtr[l] = vPtr[l] * dVal;
          }
        }

        // Matrix multiplication of the previously scaled v and the original v
        // with the selected eigenvalues
        // Also reshape virtualVis to nAntenna x nAntenna matrix
        assert(virtVisCurrent.size() == nAntenna * nAntenna);
        blas::gemm<std::complex<T>>(
            CblasNoTrans, CblasConjTrans, {1, 0}, vScaledCurrent, vCurrent, {0, 0},
            HostView<std::complex<T>, 2>(virtVisCurrent.data(), {nAntenna, nAntenna},
                                         {1, nAntenna}));

      } else {
        virtVisCurrent.zero();
      }
    }
  }
}

template auto virtual_vis<float>(ContextInternal& ctx, ConstHostView<BippFilter, 1> filter,
                                 ConstHostView<float, 2> intervals, ConstHostView<float, 1> d,
                                 ConstHostView<std::complex<float>, 2> v,
                                 HostView<std::complex<float>, 3> virtVis) -> void;

template auto virtual_vis<double>(ContextInternal& ctx, ConstHostView<BippFilter, 1> filter,
                                  ConstHostView<double, 2> intervals, ConstHostView<double, 1> d,
                                  ConstHostView<std::complex<double>, 2> v,
                                  HostView<std::complex<double>, 3> virtVis) -> void;

}  // namespace host
}  // namespace bipp
