#pragma once

#include <cstring>
#include <memory>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <variant>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace gpu {

class DomainPartition {
public:
  struct Group {
    std::size_t begin = 0;
    std::size_t size = 0;
  };

  static auto none(const std::shared_ptr<ContextInternal>& ctx, std::size_t n) {
    Buffer<Group> groupsHost(ctx->host_alloc(), 1);
    *groupsHost.get() = Group{0, n};
    return DomainPartition(ctx, n, std::move(groupsHost));
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
  static auto grid(const std::shared_ptr<ContextInternal>& ctx,
                   std::array<std::size_t, 3> gridDimensions, std::size_t n,
                   std::array<const T*, 3> coord) -> DomainPartition;

  inline auto groups() const -> const Buffer<Group>& { return groupsHost_; }

  inline auto num_elements() const -> std::size_t {
    return std::visit(
        [&](auto&& arg) -> std::size_t {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
            return arg.size();
          } else if constexpr (std::is_same_v<ArgType, std::size_t>) {
            return arg;
          }
          return 0;
        },
        permut_);
  }

  template <typename F,
            typename = std::enable_if_t<std::is_same_v<F, float> || std::is_same_v<F, double> ||
                                        std::is_same_v<F, gpu::api::ComplexFloatType> ||
                                        std::is_same_v<F, gpu::api::ComplexDoubleType>>>
  auto apply(const F* __restrict__ inDevice, F* __restrict__ outDevice) -> void;

  template <typename F,
            typename = std::enable_if_t<std::is_same_v<F, float> || std::is_same_v<F, double> ||
                                        std::is_same_v<F, gpu::api::ComplexFloatType> ||
                                        std::is_same_v<F, gpu::api::ComplexDoubleType>>>
  auto apply(F* inOutDevice) -> void;

  template <typename F,
            typename = std::enable_if_t<std::is_same_v<F, float> || std::is_same_v<F, double> ||
                                        std::is_same_v<F, gpu::api::ComplexFloatType> ||
                                        std::is_same_v<F, gpu::api::ComplexDoubleType>>>
  auto reverse(const F* __restrict__ inDevice, F* __restrict__ outDevice) -> void;

  template <typename F,
            typename = std::enable_if_t<std::is_same_v<F, float> || std::is_same_v<F, double> ||
                                        std::is_same_v<F, gpu::api::ComplexFloatType> ||
                                        std::is_same_v<F, gpu::api::ComplexDoubleType>>>
  auto reverse(F* inOutDevice) -> void;

private:
  explicit DomainPartition(std::shared_ptr<ContextInternal> ctx, Buffer<std::size_t> permut,
                           Buffer<Group> groupsHost)
      : ctx_(std::move(ctx)), permut_(std::move(permut)), groupsHost_(std::move(groupsHost)) {}

  explicit DomainPartition(std::shared_ptr<ContextInternal> ctx, std::size_t size,
                           Buffer<Group> groupsHost)
      : ctx_(std::move(ctx)), permut_(size), groupsHost_(std::move(groupsHost)) {}

  std::shared_ptr<ContextInternal> ctx_;
  std::variant<std::size_t, Buffer<std::size_t>> permut_;
  Buffer<Group> groupsHost_;
  Buffer<char> workBufferDevice_;
};

}  // namespace gpu
}  // namespace bipp
