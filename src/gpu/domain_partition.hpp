#pragma once

#include <cstring>
#include <memory>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"
#include "host/domain_partition.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace gpu {

class DomainPartition {
public:
  static auto none(const std::shared_ptr<ContextInternal>& ctx, std::size_t n) {
    std::vector<PartitionGroup> groupsHost(1);
    groupsHost[0] = PartitionGroup{0, n};
    return DomainPartition(ctx, std::move(groupsHost));
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
  static auto grid(const std::shared_ptr<ContextInternal>& ctx,
                   std::array<std::size_t, 3> gridDimensions,
                   std::array<ConstDeviceView<T, 1>, 3> coord) -> DomainPartition;

  inline auto groups() const -> const std::vector<PartitionGroup>& { return groupsHost_; }

  inline auto num_elements() const -> std::size_t {
    return permut_.size() ? permut_.size() : groupsHost_[0].size;
  }

  template <typename F,
            typename = std::enable_if_t<std::is_same_v<F, float> || std::is_same_v<F, double> ||
                                        std::is_same_v<F, gpu::api::ComplexFloatType> ||
                                        std::is_same_v<F, gpu::api::ComplexDoubleType>>>
  auto apply(ConstDeviceView<F,1> in, DeviceView<F, 1> out) -> void;

  template <typename F,
            typename = std::enable_if_t<std::is_same_v<F, float> || std::is_same_v<F, double> ||
                                        std::is_same_v<F, gpu::api::ComplexFloatType> ||
                                        std::is_same_v<F, gpu::api::ComplexDoubleType>>>
  auto apply(DeviceView<F,1> inOut) -> void;

  template <typename F,
            typename = std::enable_if_t<std::is_same_v<F, float> || std::is_same_v<F, double> ||
                                        std::is_same_v<F, gpu::api::ComplexFloatType> ||
                                        std::is_same_v<F, gpu::api::ComplexDoubleType>>>
  auto reverse(ConstDeviceView<F,1> in, DeviceView<F, 1> out) -> void;

  template <typename F,
            typename = std::enable_if_t<std::is_same_v<F, float> || std::is_same_v<F, double> ||
                                        std::is_same_v<F, gpu::api::ComplexFloatType> ||
                                        std::is_same_v<F, gpu::api::ComplexDoubleType>>>
  auto reverse(DeviceView<F, 1> inOut) -> void;

private:
  explicit DomainPartition(std::shared_ptr<ContextInternal> ctx, DeviceArray<std::size_t, 1> permut,
                           std::vector<PartitionGroup> groupsHost)
      : ctx_(std::move(ctx)), permut_(std::move(permut)), groupsHost_(std::move(groupsHost)) {}

  explicit DomainPartition(std::shared_ptr<ContextInternal> ctx, std::vector<PartitionGroup> groupsHost)
      : ctx_(std::move(ctx)), groupsHost_(std::move(groupsHost)) {}

  std::shared_ptr<ContextInternal> ctx_;
  DeviceArray<std::size_t, 1> permut_;
  std::vector<PartitionGroup> groupsHost_;
  DeviceArray<char,1> workBufferDevice_;
};

}  // namespace gpu
}  // namespace bipp
