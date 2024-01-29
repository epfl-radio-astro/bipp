#include "gpu/virtual_vis.hpp"

#include <cassert>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/kernels/scale_matrix.hpp"
#include "gpu/util/blas_api.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto virtual_vis(ContextInternal& ctx, ConstHostView<T, 2> dMasked,
                 ConstDeviceView<api::ComplexType<T>, 2> vAll,
                 DeviceView<api::ComplexType<T>, 2> virtVis) -> void {
  assert(dMasked.shape(1) == virtVis.shape(1));
  assert(vAll.shape(0) * vAll.shape(0) == virtVis.shape(0));
  assert(vAll.shape(1) == dMasked.shape(0));

  auto& queue = ctx.gpu_queue();
  const auto nAntenna = vAll.shape(0);

  const auto zero = api::ComplexType<T>{0, 0};
  const auto one = api::ComplexType<T>{1, 0};

  auto vArray = queue.create_device_array<api::ComplexType<T>, 2>(vAll.shape());
  auto vScaledArray = queue.create_device_array<api::ComplexType<T>, 2>(vAll.shape());

  auto dHostArray = queue.create_pinned_array<T, 2>(dMasked.shape());
  auto dArray = queue.create_device_array<T, 1>(dMasked.shape(0));

  for (std::size_t idxImage = 0; idxImage < virtVis.shape(1); ++idxImage) {
    auto dHost = dHostArray.slice_view(idxImage);

    // only consider non-zero eigenvalues
    std::size_t nEig = 0;
    for (std::size_t idxEig = 0; idxEig < dMasked.shape(0); ++idxEig) {
      if (dMasked[{idxEig, idxImage}]) {
        copy(queue, vAll.slice_view(idxEig), vArray.slice_view(nEig));
        dHost[nEig] = dMasked[{idxEig, idxImage}];
        ++nEig;
      }
    }

    auto virtVisCurrent = virtVis.slice_view(idxImage);

    if (!nEig) {
      api::memset_async(virtVisCurrent.data(), 0, virtVisCurrent.size_in_bytes());
    } else {
      auto v = vArray.sub_view({0, 0}, {vArray.shape(0), nEig});
      auto vScaled = vScaledArray.sub_view({0, 0}, {vArray.shape(0), nEig});
      auto d = dArray.sub_view(0, nEig);
      dHost = dHost.sub_view(0, nEig);

      copy(queue, dHost, d);

      // Multiply each col of V with the selected eigenvalue
      scale_matrix<T>(queue, nAntenna, nEig, v.data(), v.strides(1), d.data(), vScaled.data(),
                      vScaled.strides(1));

      for (std::size_t k = 0; k < nEig; ++k) {
        ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "Assigning eigenvalue {} to level {}", dHost[k],
                         idxImage);
      }

      // Matrix multiplication of the previously scaled V and the original V
      // with the selected eigenvalues
      api::blas::gemm<api::ComplexType<T>>(
          queue.blas_handle(), api::blas::operation::None, api::blas::operation::ConjugateTranspose,
          one, vScaled, v, zero,
          DeviceView<api::ComplexType<T>, 2>(virtVisCurrent.data(), {nAntenna, nAntenna},
                                             {1, nAntenna}));
    }
  }
}

template auto virtual_vis<float>(ContextInternal& ctx, ConstHostView<float, 2> dMasked,
                                 ConstDeviceView<api::ComplexType<float>, 2> vAll,
                                 DeviceView<api::ComplexType<float>, 2> virtVis) -> void;

template auto virtual_vis<double>(ContextInternal& ctx, ConstHostView<double, 2> dMasked,
                                  ConstDeviceView<api::ComplexType<double>, 2> vAll,
                                  DeviceView<api::ComplexType<double>, 2> virtVis) -> void;

}  // namespace gpu
}  // namespace bipp
