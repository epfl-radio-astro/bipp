#include <array>
#include <cstddef>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/domain_partition.hpp"
#include "gpu/util/cub_api.hpp"
#include "gpu/util/kernel_launch_grid.hpp"
#include "gpu/util/runtime.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/buffer.hpp"

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
                                           ArrayWrapper<const T*, DIM> coord, GroupIndexType* __restrict__ out) {
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
                           std::array<std::size_t, 3> gridDimensions, std::size_t n,
                           std::array<const T*, 3> coord) -> DomainPartition {
  constexpr std::size_t DIM = 3;

  const auto gridSize = std::accumulate(gridDimensions.begin(), gridDimensions.end(),
                                        std::size_t(1), std::multiplies<std::size_t>());
  if (gridSize <= 1) return DomainPartition::none(ctx, n);

  auto& q = ctx->gpu_queue();

  auto permutBuffer = q.create_device_buffer<std::size_t>(n);
  auto groupSizesBuffer = q.create_device_buffer<GroupSizeType>(gridSize);
  auto groupBufferHost = q.create_pinned_buffer<Group>(gridSize);

  // Create block to make sure buffers go out-of-scope before next sync call to safe memory
  {
    auto minMaxBuffer = q.create_device_buffer<T>(2 * DIM);
    auto keyBuffer = q.create_device_buffer<GroupIndexType>(
        n);  // GroupIndexType should be enough to enumerate groups
    auto indicesBuffer = q.create_device_buffer<std::size_t>(n);
    auto sortedKeyBuffer = q.create_device_buffer<GroupIndexType>(n);

    // Compute the minimum and maximum in each dimension stored in minMax as
    // (min_x, min_y, ..., max_x, max_y, ...)
    {
      std::size_t worksize = 0;
      api::check_status(api::cub::DeviceReduce::Min<const T*, T*>(nullptr, worksize, nullptr,
                                                                  nullptr, n, q.stream()));

      auto workBuffer = q.create_device_buffer<char>(worksize);

      for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
        api::check_status(api::cub::DeviceReduce::Min<const T*, T*>(
            workBuffer.get(), worksize, coord[dimIdx], minMaxBuffer.get() + dimIdx, n, q.stream()));
      }

      api::check_status(api::cub::DeviceReduce::Max<const T*, T*>(nullptr, worksize, nullptr,
                                                                  nullptr, n, q.stream()));
      if (worksize > workBuffer.size()) workBuffer = q.create_device_buffer<char>(worksize);

      for (std::size_t dimIdx = 0; dimIdx < DIM; ++dimIdx) {
        api::check_status(api::cub::DeviceReduce::Max<const T*, T*>(
            workBuffer.get(), worksize, coord[dimIdx], minMaxBuffer.get() + DIM + dimIdx, n,
            q.stream()));
      }
    }

    // Assign the group idx to each input element and store temporarily in the permutation array
    {
      constexpr int blockSize = 256;
      const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
      const auto grid = kernel_launch_grid(q.device_prop(), {n, 1, 1}, block);
      api::launch_kernel(assign_group_kernel<T, DIM>, grid, block, 0, q.stream(), n, gridDimensions,
                         minMaxBuffer.get(), coord, keyBuffer.get());
    }

    // Write indices before sorting
    {
      constexpr int blockSize = 256;
      const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
      const auto grid = kernel_launch_grid(q.device_prop(), {n, 1, 1}, block);
      api::launch_kernel(assign_index_kernel<std::size_t>, grid, block, 0, q.stream(), n,
                         indicesBuffer.get());
    }

    // Compute permutation through sorting indices and group keys
    {
      std::size_t workSize = 0;
      api::check_status(api::cub::DeviceRadixSort::SortPairs(
          nullptr, workSize, keyBuffer.get(), sortedKeyBuffer.get(), indicesBuffer.get(),
          permutBuffer.get(), n, 0, sizeof(GroupIndexType) * 8, q.stream()));

      auto workBuffer = q.create_device_buffer<char>(workSize);
      api::check_status(api::cub::DeviceRadixSort::SortPairs(
          workBuffer.get(), workSize, keyBuffer.get(), sortedKeyBuffer.get(), indicesBuffer.get(),
          permutBuffer.get(), n, 0, sizeof(GroupIndexType) * 8, q.stream()));
    }

    // Compute the number of elements in each group
    {
      api::memset_async(groupSizesBuffer.get(), 0, groupSizesBuffer.size_in_bytes(), q.stream());

      constexpr int blockSize = 512;
      const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
      const auto grid =
          kernel_launch_grid(q.device_prop(), {n, gridSize, 1}, {block.x, block.x, 1});
      api::launch_kernel(group_count_kernel<blockSize>, grid, block, 0, q.stream(), gridSize, n,
                         sortedKeyBuffer.get(), groupSizesBuffer.get());
    }
  }

  // Compute group begin and size
  {
    auto groupSizesHostBuffer = q.create_pinned_buffer<GroupSizeType>(groupSizesBuffer.size());
    api::memcpy_async(groupSizesHostBuffer.get(), groupSizesBuffer.get(),
                      groupSizesBuffer.size_in_bytes(), api::flag::MemcpyDeviceToHost, q.stream());

    // make sure copy operations are done
    q.sync();

    auto* __restrict__ groupsPtr = groupBufferHost.get();
    auto* __restrict__ groupSizesPtr = groupSizesHostBuffer.get();

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
auto DomainPartition::apply(const F* __restrict__ inDevice, F* __restrict__ outDevice) -> void {
  std::visit(
      [&](auto&& arg) {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
          auto& q = ctx_->gpu_queue();
          constexpr int blockSize = 256;
          const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
          const auto grid = kernel_launch_grid(q.device_prop(), {arg.size(), 1, 1}, block);
          api::launch_kernel(apply_permut_kernel<F>, grid, block, 0, q.stream(), arg.size(),
                             arg.get(), inDevice, outDevice);

        } else if constexpr (std::is_same_v<ArgType, std::size_t>) {
          gpu::api::memcpy_async(outDevice, inDevice, sizeof(F) * arg,
                                 gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());
        }
      },
      permut_);
}

template <typename F, typename>
auto DomainPartition::apply(F* inOutDevice) -> void {
  std::visit(
      [&](auto&& arg) {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
          auto& q = ctx_->gpu_queue();
          if (workBufferDevice_.size_in_bytes() < sizeof(F) * arg.size())
            workBufferDevice_ = q.create_device_buffer<char>(sizeof(F) * arg.size());

          auto workPtr = reinterpret_cast<F*>(workBufferDevice_.get());

          gpu::api::memcpy_async(workPtr, inOutDevice, sizeof(F) * arg.size(),
                                 gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());

          constexpr int blockSize = 256;
          const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
          const auto grid = kernel_launch_grid(q.device_prop(), {arg.size(), 1, 1}, block);
          api::launch_kernel(apply_permut_kernel<F>, grid, block, 0, q.stream(), arg.size(),
                             arg.get(), workPtr, inOutDevice);
        }
      },
      permut_);
}

template <typename F, typename>
auto DomainPartition::reverse(const F* __restrict__ inDevice, F* __restrict__ outDevice) -> void {
  std::visit(
      [&](auto&& arg) {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
          auto& q = ctx_->gpu_queue();
          constexpr int blockSize = 256;
          const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
          const auto grid = kernel_launch_grid(q.device_prop(), {arg.size(), 1, 1}, block);
          api::launch_kernel(reverse_permut_kernel<F>, grid, block, 0, q.stream(), arg.size(),
                             arg.get(), inDevice, outDevice);

        } else if constexpr (std::is_same_v<ArgType, std::size_t>) {
          gpu::api::memcpy_async(outDevice, inDevice, sizeof(F) * arg,
                                 gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());
        }
      },
      permut_);
}

template <typename F, typename>
auto DomainPartition::reverse(F* inOutDevice) -> void {
  std::visit(
      [&](auto&& arg) {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Buffer<std::size_t>>) {
          auto& q = ctx_->gpu_queue();

          if (workBufferDevice_.size_in_bytes() < sizeof(F) * arg.size())
            workBufferDevice_ = q.create_device_buffer<char>(sizeof(F) * arg.size());

          auto workPtr = reinterpret_cast<F*>(workBufferDevice_.get());

          gpu::api::memcpy_async(workPtr, inOutDevice, sizeof(F) * arg.size(),
                                 gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());

          constexpr int blockSize = 256;
          const dim3 block(std::min<int>(blockSize, q.device_prop().maxThreadsDim[0]), 1, 1);
          const auto grid = kernel_launch_grid(q.device_prop(), {arg.size(), 1, 1}, block);
          api::launch_kernel(reverse_permut_kernel<F>, grid, block, 0, q.stream(), arg.size(),
                             arg.get(), workPtr, inOutDevice);
        }
      },
      permut_);
}

template auto DomainPartition::grid<float>(const std::shared_ptr<ContextInternal>& ctx,
                                           std::array<std::size_t, 3> gridDimensions, std::size_t n,
                                           std::array<const float*, 3> coord) -> DomainPartition;

template auto DomainPartition::grid<double>(const std::shared_ptr<ContextInternal>& ctx,
                                            std::array<std::size_t, 3> gridDimensions,
                                            std::size_t n, std::array<const double*, 3> coord)
    -> DomainPartition;

template auto DomainPartition::apply<float>(const float* __restrict__ inDevice,
                                            float* __restrict__ outDevice) -> void;

template auto DomainPartition::apply<double>(const double* __restrict__ inDevice,
                                             double* __restrict__ outDevice) -> void;

template auto DomainPartition::apply<float>(float* inOutDevice) -> void;

template auto DomainPartition::apply<double>(double* inOutDevice) -> void;

template auto DomainPartition::apply<api::ComplexFloatType>(api::ComplexFloatType* inOutDevice)
    -> void;

template auto DomainPartition::apply<api::ComplexDoubleType>(api::ComplexDoubleType* inOutDevice)
    -> void;

template auto DomainPartition::apply<api::ComplexFloatType>(
    const api::ComplexFloatType* __restrict__ inDevice,
    api::ComplexFloatType* __restrict__ outDevice) -> void;

template auto DomainPartition::apply<api::ComplexDoubleType>(
    const api::ComplexDoubleType* __restrict__ inDevice,
    api::ComplexDoubleType* __restrict__ outDevice) -> void;

template auto DomainPartition::reverse<float>(const float* __restrict__ inDevice,
                                              float* __restrict__ outDevice) -> void;

template auto DomainPartition::reverse<double>(const double* __restrict__ inDevice,
                                               double* __restrict__ outDevice) -> void;

template auto DomainPartition::reverse<float>(float* inOutDevice) -> void;

template auto DomainPartition::reverse<double>(double* inOutDevice) -> void;

template auto DomainPartition::reverse<api::ComplexFloatType>(api::ComplexFloatType* inOutDevice)
    -> void;

template auto DomainPartition::reverse<api::ComplexDoubleType>(api::ComplexDoubleType* inOutDevice)
    -> void;

template auto DomainPartition::reverse<api::ComplexFloatType>(
    const api::ComplexFloatType* __restrict__ inDevice,
    api::ComplexFloatType* __restrict__ outDevice) -> void;

template auto DomainPartition::reverse<api::ComplexDoubleType>(
    const api::ComplexDoubleType* __restrict__ inDevice,
    api::ComplexDoubleType* __restrict__ outDevice) -> void;

}  // namespace gpu
}  // namespace bipp
