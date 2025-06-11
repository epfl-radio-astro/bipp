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

    auto partition = bipp::host::DomainPartition::grid<T, 3>(ctx_->host_alloc(), gridDimensions,
                                                             {domainX, domainY, domainZ});

    // Make sure groups cover all indices
    std::vector<bool> inputCover(domain[0].size());
    for (const auto& [begin, size] : partition.groups()) {
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
      partition.template apply<T>(dataInPlace, dataOutOfPlace);
      partition.template apply<T>(dataInPlace);

      // check data
      for (std::size_t i = 0; i < dataInPlace.size(); ++i) {
        ASSERT_EQ(dataInPlace[{i}], dataOutOfPlace[{i}]);
      }

      for (const auto& [begin, size] : partition.groups()) {
        if (size) {
          auto minGroup =
              *std::min_element(dataInPlace.data() + begin, dataInPlace.data() + begin + size);
          auto maxGroup =
              *std::max_element(dataInPlace.data() + begin, dataInPlace.data() + begin + size);

          ASSERT_LE(maxGroup - minGroup, gridSpacing[dimIdx]);
        }
      }

      // reverse in place and out of place
      partition.template reverse<T>(dataInPlace, dataOutOfPlace);
      partition.template reverse<T>(dataInPlace);

      // check reversed data
      for (std::size_t i = 0; i < dataInPlace.size(); ++i) {
        ASSERT_EQ(dataInPlace[{i}], dataOutOfPlace[{i}]);
        ASSERT_EQ(dataInPlace[{i}], domain[dimIdx][i]);
      }
    }
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

INSTANTIATE_TEST_SUITE_P(DomainPartition, DomainPartitionDouble,
                         ::testing::Combine(::testing::Values(TEST_PROCESSING_UNITS)),
                         param_type_names);
