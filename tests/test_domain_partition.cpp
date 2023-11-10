#include <array>
#include <memory>
#include <numeric>
#include <random>
#include <tuple>
#include <variant>
#include <vector>

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gtest/gtest.h"
#include "host/domain_partition.hpp"
#include "memory/view.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/domain_partition.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

template <typename T>
class DomainPartitionTest : public ::testing::TestWithParam<std::tuple<BippProcessingUnit>> {
public:
  using ValueType = T;

  DomainPartitionTest() : ctx_(new bipp::ContextInternal(std::get<0>(GetParam()))) {}

  auto test_grid(std::array<std::size_t, 3> gridDimensions, std::array<std::vector<T>, 3> domain) {
    ASSERT_EQ(domain[0].size(), domain[1].size());
    ASSERT_EQ(domain[0].size(), domain[2].size());

    bipp::HostView<T, 1> domainX(domain[0].data(), {domain[0].size()}, {1});
    bipp::HostView<T, 1> domainY(domain[1].data(), {domain[1].size()}, {1});
    bipp::HostView<T, 1> domainZ(domain[2].data(), {domain[2].size()}, {1});

    const auto gridSize = std::accumulate(gridDimensions.begin(), gridDimensions.end(),
                                          std::size_t(1), std::multiplies<std::size_t>());

    std::array<T, 3> minCoord, maxCoord, gridSpacing;

    for (std::size_t dimIdx = 0; dimIdx < minCoord.size(); ++dimIdx) {
      minCoord[dimIdx] = *std::min_element(domain[dimIdx].begin(), domain[dimIdx].end());
      maxCoord[dimIdx] = *std::max_element(domain[dimIdx].begin(), domain[dimIdx].end());
      gridSpacing[dimIdx] = (maxCoord[dimIdx] - minCoord[dimIdx]) / gridDimensions[dimIdx];
    }

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    std::variant<bipp::host::DomainPartition, bipp::gpu::DomainPartition> partition =
        bipp::host::DomainPartition::none(ctx_, domain[0].size());
#else
    std::variant<bipp::host::DomainPartition> partition =
        bipp::host::DomainPartition::none(ctx_, domain[0].size());
#endif

    if (ctx_->processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto bufferX = ctx_->gpu_queue().create_device_buffer<T>(domain[0].size());
      auto bufferY = ctx_->gpu_queue().create_device_buffer<T>(domain[0].size());
      auto bufferZ = ctx_->gpu_queue().create_device_buffer<T>(domain[0].size());

      bipp::gpu::api::memcpy_async(bufferX.get(), domain[0].data(), bufferX.size_in_bytes(),
                                   bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());
      bipp::gpu::api::memcpy_async(bufferY.get(), domain[1].data(), bufferY.size_in_bytes(),
                                   bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());
      bipp::gpu::api::memcpy_async(bufferZ.get(), domain[2].data(), bufferZ.size_in_bytes(),
                                   bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());

      partition = bipp::gpu::DomainPartition::grid<T>(
          ctx_, gridDimensions, domain[0].size(), {bufferX.get(), bufferY.get(), bufferZ.get()});

#else
      ASSERT_TRUE(false);
#endif
    } else {
      partition = bipp::host::DomainPartition::grid<T, 3>(ctx_, gridDimensions,
                                                          {domainX, domainY, domainZ});
    }

    std::visit(
        [&](auto&& arg) -> void {
          using variantType = std::decay_t<decltype(arg)>;
          // Make sure groups cover all indices
          std::vector<bool> inputCover(domain[0].size());
          for (const auto& [begin, size] : arg.groups()) {
            ASSERT_LE(begin + size, domain[0].size());
            for (std::size_t i = begin; i < begin + size; ++i) {
              inputCover[i] = true;
            }
          }
          for (std::size_t i = 0; i < domain[0].size(); ++i) {
            ASSERT_TRUE(inputCover[i]);
          }

          for (std::size_t dimIdx = 0; dimIdx < minCoord.size(); ++dimIdx) {
            bipp::HostArray<T, 1> dataInPlace(ctx_->host_alloc(), {domain[dimIdx].size()});
            bipp::copy(bipp::HostView<T, 1>(domain[dimIdx].data(), {domain[dimIdx].size()}, {1}),
                       dataInPlace);
            auto dataOutOfPlace = bipp::HostArray<T, 1>(ctx_->host_alloc(), {dataInPlace.size()});

            // apply in place and out of place
            if constexpr (std::is_same_v<variantType, bipp::host::DomainPartition>) {
              arg.apply(dataInPlace, dataOutOfPlace);
              arg.apply(dataInPlace);
            } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
              auto dataInPlaceDevice =
                  ctx_->gpu_queue().create_device_buffer<T>(dataInPlace.size());
              auto dataOutOfPlaceDevice =
                  ctx_->gpu_queue().create_device_buffer<T>(dataOutOfPlace.size());

              bipp::gpu::api::memcpy_async(
                  dataInPlaceDevice.get(), dataInPlace.data(), dataInPlaceDevice.size_in_bytes(),
                  bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());

              arg.apply(dataInPlaceDevice.get(), dataOutOfPlaceDevice.get());
              arg.apply(dataInPlaceDevice.get());

              bipp::gpu::api::memcpy_async(
                  dataInPlace.data(), dataInPlaceDevice.get(), dataInPlaceDevice.size_in_bytes(),
                  bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());
              bipp::gpu::api::memcpy_async(dataOutOfPlace.data(), dataOutOfPlaceDevice.get(),
                                           dataOutOfPlaceDevice.size_in_bytes(),
                                           bipp::gpu::api::flag::MemcpyDefault,
                                           ctx_->gpu_queue().stream());

              ctx_->gpu_queue().sync();
#endif
            }

            // check data
            for (std::size_t i = 0; i < dataInPlace.size(); ++i) {
              ASSERT_EQ(dataInPlace[{i}], dataOutOfPlace[{i}]);
            }

            for (const auto& [begin, size] : arg.groups()) {
              if (size) {
                auto minGroup = *std::min_element(dataInPlace.data() + begin,
                                                  dataInPlace.data() + begin + size);
                auto maxGroup = *std::max_element(dataInPlace.data() + begin,
                                                  dataInPlace.data() + begin + size);

                ASSERT_LE(maxGroup - minGroup, gridSpacing[dimIdx]);
              }
            }

            // reverse in place and out of place
            if constexpr (std::is_same_v<variantType, bipp::host::DomainPartition>) {
              arg.reverse(dataInPlace, dataOutOfPlace);
              arg.reverse(dataInPlace);
            } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
              auto dataInPlaceDevice =
                  ctx_->gpu_queue().create_device_buffer<T>(dataInPlace.size());
              auto dataOutOfPlaceDevice =
                  ctx_->gpu_queue().create_device_buffer<T>(dataOutOfPlace.size());

              bipp::gpu::api::memcpy_async(
                  dataInPlaceDevice.get(), dataInPlace.data(), dataInPlaceDevice.size_in_bytes(),
                  bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());

              arg.reverse(dataInPlaceDevice.get(), dataOutOfPlaceDevice.get());
              arg.reverse(dataInPlaceDevice.get());

              bipp::gpu::api::memcpy_async(
                  dataInPlace.data(), dataInPlaceDevice.get(), dataInPlaceDevice.size_in_bytes(),
                  bipp::gpu::api::flag::MemcpyDefault, ctx_->gpu_queue().stream());
              bipp::gpu::api::memcpy_async(dataOutOfPlace.data(), dataOutOfPlaceDevice.get(),
                                           dataOutOfPlaceDevice.size_in_bytes(),
                                           bipp::gpu::api::flag::MemcpyDefault,
                                           ctx_->gpu_queue().stream());

              ctx_->gpu_queue().sync();
#endif
            }

            // check reversed data
            for (std::size_t i = 0; i < dataInPlace.size(); ++i) {
              ASSERT_EQ(dataInPlace[{i}], dataOutOfPlace[{i}]);
              ASSERT_EQ(dataInPlace[{i}], domain[dimIdx][i]);
            }
          }
        },
        partition);
  }

  std::shared_ptr<bipp::ContextInternal> ctx_;
};

using DomainPartitionSingle = DomainPartitionTest<float>;
using DomainPartitionDouble = DomainPartitionTest<double>;

template <typename T>
static auto test_grid_random(std::size_t n, std::array<std::size_t, 3> gridDimensions,
                             DomainPartitionTest<T>& t) -> void {
  std::minstd_rand randGen(42);
  std::uniform_real_distribution<T> distriX(-5.0, 10.0);
  std::uniform_real_distribution<T> distriY(2.0, 20.0);
  std::uniform_real_distribution<T> distriZ(100.0, 5000.0);

  std::vector<T> x(n);
  std::vector<T> y(n);
  std::vector<T> z(n);

  for (auto& val : x) val = distriX(randGen);
  for (auto& val : y) val = distriY(randGen);
  for (auto& val : z) val = distriZ(randGen);

  t.test_grid(gridDimensions, {x, y, z});
}

TEST_P(DomainPartitionSingle, grid_n1) { test_grid_random(1, {2, 3, 4}, *this); }

TEST_P(DomainPartitionDouble, grid_n1) { test_grid_random(1, {2, 3, 4}, *this); }

TEST_P(DomainPartitionSingle, grid_n100) { test_grid_random(100, {2, 3, 4}, *this); }

TEST_P(DomainPartitionDouble, grid_n100) { test_grid_random(100, {2, 3, 4}, *this); }

TEST_P(DomainPartitionSingle, grid_n4000) { test_grid_random(4000, {2, 3, 4}, *this); }

TEST_P(DomainPartitionDouble, grid_n4000) { test_grid_random(4000, {2, 3, 4}, *this); }

static auto param_type_names(const ::testing::TestParamInfo<std::tuple<BippProcessingUnit>>& info)
    -> std::string {
  std::stringstream stream;

  if (std::get<0>(info.param) == BIPP_PU_CPU)
    stream << "CPU";
  else
    stream << "GPU";

  return stream.str();
}

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#define TEST_PROCESSING_UNITS BIPP_PU_CPU, BIPP_PU_GPU
#else
#define TEST_PROCESSING_UNITS BIPP_PU_CPU
#endif

INSTANTIATE_TEST_SUITE_P(DomainPartition, DomainPartitionSingle,
                         ::testing::Combine(::testing::Values(TEST_PROCESSING_UNITS)),
                         param_type_names);

INSTANTIATE_TEST_SUITE_P(Lofar, DomainPartitionDouble,
                         ::testing::Combine(::testing::Values(TEST_PROCESSING_UNITS)),
                         param_type_names);
