#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <type_traits>
#include <variant>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace host {

class DomainPartition {
public:
  struct Group {
    std::size_t begin = 0;
    std::size_t size = 0;
  };

  static auto none(const std::shared_ptr<ContextInternal>& ctx,std::size_t n ) {
    Buffer<Group> groups(ctx->host_alloc(), 1);
    *groups.get() = Group{0, n};
    return DomainPartition(ctx, n, std::move(groups));
  }

  template <typename T, std::size_t DIM>
  static auto grid(const std::shared_ptr<ContextInternal>& ctx, std::array<std::size_t, DIM> gridDimensions,
                   std::size_t n, std::array<const T*, DIM> coord) -> DomainPartition {
    const auto gridSize = std::accumulate(gridDimensions.begin(), gridDimensions.end(),
                                          std::size_t(1), std::multiplies<std::size_t>());
    if (gridSize <= 1) return DomainPartition::none(ctx, n);

    std::array<T, DIM> minCoord, maxCoord;

    for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
      minCoord[dimIdx] = *std::min_element(coord[dimIdx], coord[dimIdx] + n);
      maxCoord[dimIdx] = *std::max_element(coord[dimIdx], coord[dimIdx] + n);
    }

    Buffer<Group> groups(ctx->host_alloc(), gridSize);
    auto* __restrict__ groupsPtr = groups.get();
    std::memset(groupsPtr, 0, groups.size_in_bytes());

    Buffer<std::size_t> permut(ctx->host_alloc(), n);
    auto* __restrict__ permutPtr = permut.get();

    std::array<T, DIM> gridSpacingInv;
    for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
      gridSpacingInv[dimIdx] = gridDimensions[dimIdx] / (maxCoord[dimIdx] - minCoord[dimIdx]);
    }

    // Compute the assigned group index in grid for each data point and store temporarily in permut
    // array. Increment groups array of following group each time, such that the size of the
    // previous group is computed.
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t groupIndex = 0;
      for (std::size_t dimIdx = DIM - 1;; --dimIdx) {
        groupIndex = groupIndex * gridDimensions[dimIdx] +
                     std::max<std::size_t>(
                         std::min(static_cast<std::size_t>(gridSpacingInv[dimIdx] *
                                                           (coord[dimIdx][i] - minCoord[dimIdx])),
                                  gridDimensions[dimIdx] - 1),
                         0);

        if (!dimIdx) break;
      }

      permutPtr[i] = groupIndex;
      ++groupsPtr[groupIndex].size;
    }

    // Compute the rolling sum, such that each group has its begin index
    for (std::size_t i = 1; i < groups.size(); ++i) {
      groupsPtr[i].begin += groupsPtr[i - 1].begin + groupsPtr[i - 1].size;
    }

    // Finally compute permutation index for each data point and restore group sizes.
    for (std::size_t i = 0; i < n; ++i) {
      permutPtr[i] = groupsPtr[permutPtr[i]].begin++;
    }

    // Restore begin index
    for (std::size_t i = 0; i < groups.size(); ++i) {
      groupsPtr[i].begin -= groupsPtr[i].size;
    }

    return DomainPartition(ctx, std::move(permut), std::move(groups));
  }

  inline auto groups() const -> const Buffer<Group>& { return groups_; }

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

  template <typename F>
  inline auto apply(const F* __restrict__ in, F* __restrict__ out) -> void {
    std::visit(
        [&](auto&& arg) {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
            const auto* __restrict__ permutPtr = arg.get();
            for (std::size_t i = 0; i < arg.size(); ++i) {
              out[permutPtr[i]] = in[i];
            }
          } else if constexpr (std::is_same_v<ArgType, std::size_t>) {
            std::memcpy(out, in, sizeof(F) * arg);
          }
        },
        permut_);
  }

  template <typename F>
  inline auto apply(F* __restrict__ inOut) -> void {
    std::visit(
        [&](auto&& arg) {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
            const auto* __restrict__ permutPtr = arg.get();
            Buffer<F> tmp(ctx_->host_alloc(), arg.size());
            F* __restrict__ tmpPtr = tmp.get();
            std::memcpy(tmpPtr, inOut, tmp.size_in_bytes());
            for (std::size_t i = 0; i < arg.size(); ++i) {
              inOut[permutPtr[i]] = tmpPtr[i];
            }
          }
        },
        permut_);
  }

  template <typename F>
  inline auto reverse(const F* __restrict__ in, F* __restrict__ out) -> void {
    std::visit(
        [&](auto&& arg) {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
            const auto* __restrict__ permutPtr = arg.get();
            for (std::size_t i = 0; i < arg.size(); ++i) {
              out[i] = in[permutPtr[i]];
            }
          } else if constexpr (std::is_same_v<ArgType, std::size_t>) {
            std::memcpy(out, in, sizeof(F) * arg);
          }
        },
        permut_);
  }

  template <typename F>
  inline auto reverse(F* __restrict__ inOut) -> void {
    std::visit(
        [&](auto&& arg) {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
            const auto* __restrict__ permutPtr = arg.get();
            Buffer<F> tmp(ctx_->host_alloc(), arg.size());
            F* __restrict__ tmpPtr = tmp.get();
            for (std::size_t i = 0; i < arg.size(); ++i) {
              tmpPtr[i] = inOut[permutPtr[i]];
            }
            std::memcpy(inOut, tmpPtr, tmp.size_in_bytes());
          }
        },
        permut_);
  }

private:
  explicit DomainPartition(std::shared_ptr<ContextInternal> ctx, Buffer<std::size_t> permut,
                           Buffer<Group> groups)
      : ctx_(std::move(ctx)), permut_(std::move(permut)), groups_(std::move(groups)) {}

  explicit DomainPartition(std::shared_ptr<ContextInternal> ctx, std::size_t size,
                           Buffer<Group> groups)
      : ctx_(std::move(ctx)), permut_(size), groups_(std::move(groups)) {}

  std::shared_ptr<ContextInternal> ctx_;
  std::variant<std::size_t, Buffer<std::size_t>> permut_;
  Buffer<Group> groups_;
};

}  // namespace host
}  // namespace bipp
