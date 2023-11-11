#include "gpu/virtual_vis.hpp"

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/kernels/apply_filter.hpp"
#include "gpu/kernels/scale_matrix.hpp"
#include "gpu/util/blas_api.hpp"
#include "gpu/util/runtime_api.hpp"
#include "host/kernels/apply_filter.hpp"
#include "host/kernels/interval_indices.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto virtual_vis(ContextInternal& ctx, ConstHostView<BippFilter, 1> filter,
                 ConstHostView<T, 2> intervals, ConstDeviceView<T, 1> d,
                 ConstDeviceView<api::ComplexType<T>, 2> v,
                 DeviceView<api::ComplexType<T>, 3> virtVis) -> void {
  assert(filter.size() == virtVis.shape(2));
  assert(intervals.shape(1) == virtVis.shape(1));
  assert(v.shape(0) * v.shape(0) == virtVis.shape(0));
  assert(v.shape(1) == d.size());

  auto& queue = ctx.gpu_queue();
  const auto nAntenna = v.shape(0);

  const auto zero = api::ComplexType<T>{0, 0};
  const auto one = api::ComplexType<T>{1, 0};

  auto vScaled = queue.create_device_array<api::ComplexType<T>, 2>(v.shape());
  auto dFiltered = queue.create_device_array<T, 1>(d.shape());
  auto dHost = queue.create_host_array<T, 1>(d.shape());
  copy(queue, d, dHost);
  queue.sync(); // make sure d is on host

  for (std::size_t i = 0; i < filter.size(); ++i) {
    apply_filter(queue, filter[{i}], d.size(), d.data(), dFiltered.data());

    for (std::size_t j = 0; j < intervals.shape(1); ++j) {
      std::size_t start, size;
      std::tie(start, size) = host::find_interval_indices(dHost.size(), dHost.data(),
                                                          intervals[{0, j}], intervals[{1, j}]);

      auto virtVisCurrent = virtVis.slice_view(i).slice_view(j);
      if (size) {
        // Multiply each col of V with the selected eigenvalue
        auto vScaledCurrent = vScaled.sub_view({0, 0}, {vScaled.shape(0), size});
        auto vCurrent = v.sub_view({0, start}, {v.shape(0), size});
        scale_matrix<T>(queue, nAntenna, size, vCurrent.data(), vCurrent.strides()[1],
                        dFiltered.data() + start, vScaledCurrent.data(),
                        vScaledCurrent.strides()[1]);

        for (std::size_t k = 0; k < size; ++k) {
          ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "Assigning eigenvalue {} to inverval [{}, {}]",
                           dHost[{start + k}], intervals[{0, j}], intervals[{1, j}]);
        }

        // Matrix multiplication of the previously scaled V and the original V
        // with the selected eigenvalues
        api::blas::gemm(queue.blas_handle(), api::blas::operation::None,
                        api::blas::operation::ConjugateTranspose, one, vScaledCurrent, vCurrent,
                        zero,
                        DeviceView<api::ComplexType<T>, 2>(virtVisCurrent.data(),
                                                           {nAntenna, nAntenna}, {1, nAntenna}));

      } else {
        api::memset_async(virtVisCurrent.data(), 0,
                          virtVisCurrent.size() * sizeof(api::ComplexType<T>), queue.stream());
      }
    }
  }
}

template auto virtual_vis<float>(ContextInternal& ctx, ConstHostView<BippFilter, 1> filter,
                                 ConstHostView<float, 2> intervals, ConstDeviceView<float, 1> d,
                                 ConstDeviceView<api::ComplexType<float>, 2> v,
                                 DeviceView<api::ComplexType<float>, 3> virtVis) -> void;

template auto virtual_vis<double>(ContextInternal& ctx, ConstHostView<BippFilter, 1> filter,
                                  ConstHostView<double, 2> intervals, ConstDeviceView<double, 1> d,
                                  ConstDeviceView<api::ComplexType<double>, 2> v,
                                  DeviceView<api::ComplexType<double>, 3> virtVis) -> void;

}  // namespace gpu
}  // namespace bipp
