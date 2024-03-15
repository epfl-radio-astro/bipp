#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <type_traits>
#include <variant>
#include <vector>
#include <cassert>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"

namespace bipp {
struct PartitionGroup {
  std::size_t begin = 0;
  std::size_t size = 0;
};

namespace host {

class DomainPartition {
public:

  static auto none(const std::shared_ptr<ContextInternal>& ctx,std::size_t n ) {
    std::vector<PartitionGroup> groups(1);
    groups[0] = PartitionGroup{0, n};
    return DomainPartition(ctx, std::move(groups));
  }

  template <typename T, std::size_t DIM>
  static auto grid(const std::shared_ptr<ContextInternal>& ctx,
                   std::array<std::size_t, DIM> gridDimensions,
                   std::array<ConstHostView<T, 1>, DIM> coord) -> DomainPartition {
    const auto n = coord[0].size();
    for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
      assert(coord[dimIdx].size() == n);
    }

    const auto gridSize = std::accumulate(gridDimensions.begin(), gridDimensions.end(),
                                          std::size_t(1), std::multiplies<std::size_t>());
    if (gridSize <= 1) return DomainPartition::none(ctx, n);

    std::array<T, DIM> minCoord, maxCoord;

    for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
      auto minMaxIt = std::minmax_element(coord[dimIdx].data(), coord[dimIdx].data() + n);
      minCoord[dimIdx] = *minMaxIt.first;
      maxCoord[dimIdx] = *minMaxIt.second;
    }

    std::vector<PartitionGroup> groups(gridSize);

    HostArray<std::size_t, 1> permut(ctx->host_alloc(), {n});

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
                                                           (coord[dimIdx][{i}] - minCoord[dimIdx])),
                                  gridDimensions[dimIdx] - 1),
                         0);

        if (!dimIdx) break;
      }

      permut[{i}] = groupIndex;
      ++groups[groupIndex].size;
    }

    // Compute the rolling sum, such that each group has its begin index
    for (std::size_t i = 1; i < groups.size(); ++i) {
      groups[i].begin += groups[i - 1].begin + groups[i - 1].size;
    }

    // Finally compute permutation index for each data point and restore group sizes.
    for (std::size_t i = 0; i < n; ++i) {
      permut[{i}] = groups[permut[{i}]].begin++;
    }

    // Restore begin index
    for (std::size_t i = 0; i < groups.size(); ++i) {
      groups[i].begin -= groups[i].size;
    }

    return DomainPartition(ctx, std::move(permut), std::move(groups));
  }

  inline auto groups() const -> const std::vector<PartitionGroup>& { return groups_; }

  inline auto num_elements() const -> std::size_t {
    return permut_.size() ? permut_.size() : groups_[0].size;
  }

  template <typename F>
  inline auto apply(ConstHostView<F, 1> in, HostView<F, 1> out) -> void {
    assert(in.size() == out.size());
    if (permut_.size()) {
      const std::size_t* __restrict__ permutPtr = permut_.data();
      const F* __restrict__ inPtr = in.data();
      F* __restrict__ outPtr = out.data();

      assert(permut_.size() == in.size());
      assert(inPtr != outPtr);

      for (std::size_t i = 0; i < permut_.size(); ++i) {
        outPtr[permutPtr[i]] = inPtr[i];
      }

    } else {
      assert(groups_[0].size == in.size());
      copy(in, out);
    }
  }

  template <typename F>
  inline auto apply(HostView<F, 1> inOut) -> void {
    assert(permut_.size() || groups_[0].size == inOut.size());
    if (permut_.size()) {
      HostArray<F, 1> scratch(ctx_->host_alloc(), {inOut.size()});
      this->apply<F>(inOut, scratch);
      copy(scratch, inOut);
    }
  }

  template <typename F>
  inline auto reverse(ConstHostView<F, 1> in, HostView<F, 1> out) -> void {
    assert(in.size() == out.size());
    if (permut_.size()) {
      const std::size_t* __restrict__ permutPtr = permut_.data();
      const F* __restrict__ inPtr = in.data();
      F* __restrict__ outPtr = out.data();

      assert(permut_.size() == in.size());
      assert(inPtr != outPtr);

      for (std::size_t i = 0; i < permut_.size(); ++i) {
        outPtr[i] = inPtr[permutPtr[i]];
      }

    } else {
      assert(groups_[0].size == in.size());
      copy(in, out);
    }
  }

  template <typename F>
  inline auto reverse(HostView<F, 1> inOut) -> void {
    assert(permut_.size() || groups_[0].size == inOut.size());
    if (permut_.size()) {
      HostArray<F, 1> scratch(ctx_->host_alloc(), {inOut.size()});
      this->reverse<F>(inOut, scratch);
      copy(scratch, inOut);
    } else {
    }
  }

private:
  explicit DomainPartition(std::shared_ptr<ContextInternal> ctx, HostArray<std::size_t, 1> permut,
                           std::vector<PartitionGroup> groups)
      : ctx_(std::move(ctx)), permut_(std::move(permut)), groups_(std::move(groups)) {}

  explicit DomainPartition(std::shared_ptr<ContextInternal> ctx, std::vector<PartitionGroup> groups)
      : ctx_(std::move(ctx)), groups_(std::move(groups)) {}

  std::shared_ptr<ContextInternal> ctx_;
  HostArray<std::size_t, 1> permut_;
  std::vector<PartitionGroup> groups_;
};

}  // namespace host
}  // namespace bipp
