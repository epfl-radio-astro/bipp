#include <array>
#include <cassert>
#include <cstddef>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/domain_partition.hpp"
#include "gpu/util/cub_api.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace gpu {

namespace {
using GroupIndexType = unsigned;
using GroupSizeType = unsigned long long;

template <typename T, std::size_t DIM>
struct ArrayWrapper {
  ArrayWrapper(const std::array<T, DIM>& a) {
    for (std::size_t i = 0; i < DIM; ++i) data[i] = a[i];
  }

  T data[DIM];
};
}  // namespace

template <typename T, std::size_t DIM>
static __global__ void assign_group_kernel(std::size_t n,
                                           ArrayWrapper<std::size_t, DIM> gridDimensions,
                                           const T* __restrict__ minCoordsMaxGlobal,
                                           ArrayWrapper<const T*, DIM> coord,
                                           GroupIndexType* __restrict__ out) {
  T minCoords[DIM];
  T maxCoords[DIM];
  for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
    minCoords[dimIdx] = minCoordsMaxGlobal[dimIdx];
  }
  for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
    maxCoords[dimIdx] = minCoordsMaxGlobal[DIM + dimIdx];
  }

  T gridSpacingInv[DIM];
  for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
    gridSpacingInv[dimIdx] = gridDimensions.data[dimIdx] / (maxCoords[dimIdx] - minCoords[dimIdx]);
  }

  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    GroupIndexType groupIndex = 0;
    for (std::size_t dimIdx = DIM - 1;; --dimIdx) {
      const T* __restrict__ coordDimPtr = coord.data[dimIdx];
      groupIndex = groupIndex * gridDimensions.data[dimIdx] +
                   max(min(static_cast<GroupIndexType>(gridSpacingInv[dimIdx] *
                                                       (coordDimPtr[i] - minCoords[dimIdx])),
                           static_cast<GroupIndexType>(gridDimensions.data[dimIdx] - 1)),
                       GroupIndexType(0));

      if (!dimIdx) break;
    }

    out[i] = groupIndex;
  }
}

template <std::size_t BLOCK_THREADS>
static __global__ void group_count_kernel(std::size_t nGroups, std::size_t n,
                                          const GroupIndexType* __restrict__ in,
                                          GroupSizeType* groupCount) {
  __shared__ GroupIndexType inCache[BLOCK_THREADS];

  for (GroupIndexType groupStart = blockIdx.y * BLOCK_THREADS; groupStart < nGroups;
       groupStart += gridDim.y * BLOCK_THREADS) {
    const GroupIndexType myGroup = groupStart + threadIdx.x;
    GroupSizeType myCount = 0;

    for (std::size_t idxStart = blockIdx.x * BLOCK_THREADS; idxStart < n;
         idxStart += gridDim.x * BLOCK_THREADS) {
      if (idxStart + threadIdx.x < n) inCache[threadIdx.x] = in[idxStart + threadIdx.x];
      __syncthreads();
      for (std::size_t i = 0; i < min(n - idxStart, BLOCK_THREADS); ++i) {
        myCount += (myGroup == inCache[i]);
      }
    }
    if (myGroup < nGroups && myCount) atomicAdd(groupCount + myGroup, myCount);
  }
}

template <typename T>
static __global__ void assign_index_kernel(std::size_t n, T* __restrict__ out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    out[i] = i;
  }
}

template <typename T>
static __global__ void reverse_permut_kernel(std::size_t n, const std::size_t* __restrict__ permut,
                                             const T* __restrict__ in, T* __restrict__ out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    out[permut[i]] = in[i];
  }
}

template <typename T>
static __global__ void apply_permut_kernel(std::size_t n, const std::size_t* __restrict__ permut,
                                           const T* __restrict__ in, T* __restrict__ out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    out[i] = in[permut[i]];
  }
}

template <typename T, typename>
auto DomainPartition::grid(const std::shared_ptr<ContextInternal>& ctx,
                           std::array<std::size_t, 3> gridDimensions,
                           std::array<ConstDeviceView<T, 1>, 3> coord) -> DomainPartition {
  constexpr std::size_t DIM = 3;

  const auto n = coord[0].size();

  assert(coord[1].size() == n);
  assert(coord[2].size() == n);

  const auto gridSize = std::accumulate(gridDimensions.begin(), gridDimensions.end(),
                                        std::size_t(1), std::multiplies<std::size_t>());
  if (gridSize <= 1) return DomainPartition::none(ctx, n);

  auto& q = ctx->gpu_queue();

  auto permutBuffer = q.create_device_array<std::size_t, 1>(n);
  auto groupSizesBuffer = q.create_device_array<GroupSizeType, 1>(gridSize);
  auto groupBufferHost = std::vector<Group>(gridSize);

  // Create block to make sure buffers go out-of-scope before next sync call to safe memory
  {
    auto minMaxBuffer = q.create_device_array<T, 1>(2 * DIM);
    auto keyBuffer = q.create_device_array<GroupIndexType, 1>(
        {n});  // GroupIndexType should be enough to enumerate groups
    auto indicesBuffer = q.create_device_array<std::size_t, 1>(n);
    auto sortedKeyBuffer = q.create_device_array<GroupIndexType, 1>(n);

    // Compute the minimum and maximum in each dimension stored in minMax as
    // (min_x, min_y, ..., max_x, max_y, ...)
    {
      std::size_t worksize = 0;
      api::check_status(api::cub::DeviceReduce::Min<const T*, T*>(nullptr, worksize, nullptr,
                                                                  nullptr, n, q.stream()));

      auto workBuffer = q.create_device_array<char, 1>(worksize);

      for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
        api::check_status(api::cub::DeviceReduce::Min<const T*, T*>(
            workBuffer.data(), worksize, coord[dimIdx].data(), minMaxBuffer.data() + dimIdx, n,
            q.stream()));
      }

      api::check_status(api::cub::DeviceReduce::Max<const T*, T*>(nullptr, worksize, nullptr,
                                                                  nullptr, n, q.stream()));
      if (worksize > workBuffer.size()) workBuffer = q.create_device_array<char, 1>(worksize);

      for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
        api::check_status(api::cub::DeviceReduce::Max<const T*, T*>(
            workBuffer.data(), worksize, coord[dimIdx].data(), minMaxBuffer.data() + DIM + dimIdx,
            n, q.stream()));
      }
    }

    // Assign the group idx to each input element and store temporarily in the permutation array
    {
      constexpr int blockSize = 256;
      const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
      const auto grid = kernel_launch_grid(q.device_prop(), {n, 1, 1}, block);
      ArrayWrapper<const T*, 3> coordPtr({coord[0].data(), coord[1].data(), coord[2].data()});
      api::launch_kernel(assign_group_kernel<T, DIM>, grid, block, 0, q.stream(), n, gridDimensions,
                         minMaxBuffer.data(), coordPtr, keyBuffer.data());
    }

    // Write indices before sorting
    {
      constexpr int blockSize = 256;
      const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
      const auto grid = kernel_launch_grid(q.device_prop(), {n, 1, 1}, block);
      api::launch_kernel(assign_index_kernel<std::size_t>, grid, block, 0, q.stream(), n,
                         indicesBuffer.data());
    }

    // Compute permutation through sorting indices and group keys
    {
      std::size_t workSize = 0;
      api::check_status(api::cub::DeviceRadixSort::SortPairs(
          nullptr, workSize, keyBuffer.data(), sortedKeyBuffer.data(), indicesBuffer.data(),
          permutBuffer.data(), n, 0, sizeof(GroupIndexType) * 8, q.stream()));

      auto workBuffer = q.create_device_array<char, 1>(workSize);
      api::check_status(api::cub::DeviceRadixSort::SortPairs(
          workBuffer.data(), workSize, keyBuffer.data(), sortedKeyBuffer.data(),
          indicesBuffer.data(), permutBuffer.data(), n, 0, sizeof(GroupIndexType) * 8, q.stream()));
    }

    // Compute the number of elements in each group
    {
      api::memset_async(groupSizesBuffer.data(), 0, groupSizesBuffer.size_in_bytes(), q.stream());

      constexpr int blockSize = 512;
      const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
      const auto grid =
          kernel_launch_grid(q.device_prop(), {n, gridSize, 1}, {block.x, block.x, 1});
      api::launch_kernel(group_count_kernel<blockSize>, grid, block, 0, q.stream(), gridSize, n,
                         sortedKeyBuffer.data(), groupSizesBuffer.data());
    }
  }

  // Compute group begin and size
  {
    auto groupSizesHostBuffer = q.create_pinned_array<GroupSizeType, 1>(groupSizesBuffer.size());

    copy(q, groupSizesBuffer, groupSizesHostBuffer);

    // make sure copy operations are done
    q.sync();

    auto* __restrict__ groupsPtr = groupBufferHost.data();
    auto* __restrict__ groupSizesPtr = groupSizesHostBuffer.data();

    // number of groups is always >= 1
    groupsPtr[0].begin = 0;
    groupsPtr[0].size = groupSizesPtr[0];
    // Compute group begin index
    for (std::size_t i = 1; i < groupBufferHost.size(); ++i) {
      groupsPtr[i].begin = groupsPtr[i - 1].size + groupsPtr[i - 1].begin;
      groupsPtr[i].size = groupSizesPtr[i];
    }
  }

  return DomainPartition(ctx, std::move(permutBuffer), std::move(groupBufferHost));
}

template <typename F, typename>
auto DomainPartition::apply(ConstDeviceView<F, 1> in, DeviceView<F, 1> out) -> void {
  if (permut_.size()) {
    assert(permut_.size() == in.size());
    assert(permut_.size() == out.size());
    auto& q = ctx_->gpu_queue();
    constexpr int blockSize = 256;
    const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
    const auto grid = kernel_launch_grid(q.device_prop(), {permut_.size(), 1, 1}, block);
    api::launch_kernel(apply_permut_kernel<F>, grid, block, 0, q.stream(), permut_.size(),
                       permut_.data(), in.data(), out.data());

  } else {
    assert(groupsHost_[0].size == in.size());
    assert(groupsHost_[0].size == out.size());
    copy(ctx_->gpu_queue(), in, out);
  }
}

template <typename F, typename>
auto DomainPartition::apply(DeviceView<F, 1> inOut) -> void {
  if (permut_.size()) {
    assert(permut_.size() == inOut.size());
    auto& q = ctx_->gpu_queue();
    constexpr int blockSize = 256;
    const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
    const auto grid = kernel_launch_grid(q.device_prop(), {permut_.size(), 1, 1}, block);

    auto tmpBuffer = q.create_device_array<F, 1>(inOut.shape());
    api::launch_kernel(apply_permut_kernel<F>, grid, block, 0, q.stream(), permut_.size(),
                       permut_.data(), inOut.data(), tmpBuffer.data());
    copy(q, tmpBuffer, inOut);

  } else {
    assert(groupsHost_[0].size == inOut.size());
  }
}

template <typename F, typename>
auto DomainPartition::reverse(ConstDeviceView<F, 1> in, DeviceView<F, 1> out) -> void {
  if (permut_.size()) {
    assert(permut_.size() == in.size());
    assert(permut_.size() == out.size());
    auto& q = ctx_->gpu_queue();
    constexpr int blockSize = 256;
    const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
    const auto grid = kernel_launch_grid(q.device_prop(), {permut_.size(), 1, 1}, block);
    api::launch_kernel(reverse_permut_kernel<F>, grid, block, 0, q.stream(), permut_.size(),
                       permut_.data(), in.data(), out.data());

  } else {
    assert(groupsHost_[0].size == in.size());
    assert(groupsHost_[0].size == out.size());
    copy(ctx_->gpu_queue(), in, out);
  }
}

template <typename F, typename>
auto DomainPartition::reverse(DeviceView<F, 1> inOut) -> void {
  if (permut_.size()) {
    assert(permut_.size() == inOut.size());
    auto& q = ctx_->gpu_queue();
    constexpr int blockSize = 256;
    const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
    const auto grid = kernel_launch_grid(q.device_prop(), {permut_.size(), 1, 1}, block);

    auto tmpBuffer = q.create_device_array<F, 1>(inOut.shape());
    api::launch_kernel(reverse_permut_kernel<F>, grid, block, 0, q.stream(), permut_.size(),
                       permut_.data(), inOut.data(), tmpBuffer.data());
    copy(q, tmpBuffer, inOut);

  } else {
    assert(groupsHost_[0].size == inOut.size());
  }
}

#define BIPP_INSTANTIATE_DP_GPU(TYPE)                                                      \
  template auto DomainPartition::grid<TYPE>(const std::shared_ptr<ContextInternal>& ctx,   \
                                            std::array<std::size_t, 3> gridDimensions,     \
                                            std::array<ConstDeviceView<TYPE, 1>, 3> coord) \
      ->DomainPartition;

BIPP_INSTANTIATE_DP_GPU(float)
BIPP_INSTANTIATE_DP_GPU(double)

#define BIPP_INSTANTIATE_DP_APPLY_REVERSE_GPU(TYPE)                                                \
  template auto DomainPartition::apply<TYPE>(ConstDeviceView<TYPE, 1> in, DeviceView<TYPE, 1> out) \
      ->void;                                                                                      \
  template auto DomainPartition::apply<TYPE>(DeviceView<TYPE, 1> inOut)->void;                     \
  template auto DomainPartition::reverse<TYPE>(ConstDeviceView<TYPE, 1> in,                        \
                                               DeviceView<TYPE, 1> out)                            \
      ->void;                                                                                      \
  template auto DomainPartition::reverse<TYPE>(DeviceView<TYPE, 1> inOut)->void;

BIPP_INSTANTIATE_DP_APPLY_REVERSE_GPU(float)
BIPP_INSTANTIATE_DP_APPLY_REVERSE_GPU(double)
BIPP_INSTANTIATE_DP_APPLY_REVERSE_GPU(api::ComplexFloatType)
BIPP_INSTANTIATE_DP_APPLY_REVERSE_GPU(api::ComplexDoubleType)

}  // namespace gpu
}  // namespace bipp
