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
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace host {

template <typename T>
auto virtual_vis(ContextInternal& ctx, ConstHostView<T, 2> dMasked,
                 ConstHostView<std::complex<T>, 2> vAll, HostView<std::complex<T>, 2> virtVis)
    -> void {
  assert(dMasked.shape(1) == virtVis.shape(1));
  assert(vAll.shape(0) * vAll.shape(0) == virtVis.shape(0));
  assert(vAll.shape(1) == dMasked.shape(0));

  const auto nAntenna = vAll.shape(0);

  auto vArray = HostArray<std::complex<T>, 2>(ctx.host_alloc(), vAll.shape());
  auto vScaledArray = HostArray<std::complex<T>, 2>(ctx.host_alloc(), vAll.shape());
  auto dArray = HostArray<T, 1>(ctx.host_alloc(), dMasked.shape(0));

  for (std::size_t idxImage = 0; idxImage < virtVis.shape(1); ++idxImage) {
    // only consider non-zero eigenvalues
    std::size_t nEig = 0;
    for (std::size_t idxEig = 0; idxEig < dMasked.shape(0); ++idxEig) {
      if (dMasked[{idxEig, idxImage}]) {
        copy(vAll.slice_view(idxEig), vArray.slice_view(nEig));
        dArray[nEig] = dMasked[{idxEig, idxImage}];
        ++nEig;
      }
    }

    auto v = vArray.sub_view({0, 0}, {vArray.shape(0), nEig});
    auto vScaled = vScaledArray.sub_view({0, 0}, {vArray.shape(0), nEig});
    auto d = dArray.sub_view(0, nEig);

    auto virtVisCurrent = virtVis.slice_view(idxImage);
    if (nEig) {
      for (std::size_t idxEig = 0; idxEig < nEig; ++idxEig) {
        const auto dVal = d[idxEig];
        auto* __restrict__ vScaledPtr = &vScaled[{0, idxEig}];
        const auto* __restrict__ vPtr = &v[{0, idxEig}];

        ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "Assigning eigenvalue {} (filtered {}) to level {}",
                         d[idxEig], dVal, idxImage);
        for (std::size_t l = 0; l < v.shape(0); ++l) {
          vScaledPtr[l] = vPtr[l] * dVal;
        }
      }

      // Matrix multiplication of the previously scaled vAll and the original vAll
      // with the selected eigenvalues
      // Also reshape virtualVis to nAntenna x nAntenna matrix
      assert(virtVisCurrent.size() == nAntenna * nAntenna);
      blas::gemm<std::complex<T>>(
          CblasNoTrans, CblasConjTrans, {1, 0}, vScaled, v, {0, 0},
          HostView<std::complex<T>, 2>(virtVisCurrent.data(), {nAntenna, nAntenna}, {1, nAntenna}));

    } else {
      virtVisCurrent.zero();
    }
  }
}

template auto virtual_vis<float>(ContextInternal& ctx, ConstHostView<float, 2> dMasked,
                                 ConstHostView<std::complex<float>, 2> v,
                                 HostView<std::complex<float>, 2> virtVis) -> void;

template auto virtual_vis<double>(ContextInternal& ctx, ConstHostView<double, 2> dMasked,
                                  ConstHostView<std::complex<double>, 2> v,
                                  HostView<std::complex<double>, 2> virtVis) -> void;

}  // namespace host
}  // namespace bipp
