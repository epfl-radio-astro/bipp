#pragma once

#include <cassert>
#include <complex>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "bipp/config.h"
#include "memory/view.hpp"

namespace bipp {

namespace host {

template<typename T>
struct UVWGroup {
    std::array<T, 3> lowerBoundInclusive = {0};
    std::array<T, 3> upperBoundExclusive = {0};
};

// create partition groups, such that the inclusive range between lowerBound and upperBound is
// partitioned into a grid
template <typename T>
inline auto create_uvw_partitions(std::array<std::size_t, 3> grid, std::array<T, 3> lowerBound,
                                  std::array<T, 3> upperBound) -> std::vector<UVWGroup<T>> {
    std::array<T, 3> spacing = {(upperBound[0] - lowerBound[0]) / T(grid[0]),
                                    (upperBound[1] - lowerBound[1]) / T(grid[1]),
                                    (upperBound[2] - lowerBound[2]) / T(grid[2])};

    constexpr T roundupFactor = T(1) + std::numeric_limits<T>::epsilon();


    std::vector<UVWGroup<T>> partitions;
    partitions.reserve(
        std::reduce(grid.begin(), grid.end(), std::size_t(1), std::multiplies<std::size_t>()));

    for (std::size_t i = 0; i < grid[0]; ++i) {
      for (std::size_t j = 0; j < grid[1]; ++j) {
        for (std::size_t k = 0; k < grid[2]; ++k) {
          std::array<T, 3> lowerBoundInclusive = {lowerBound[0] + i * spacing[0],
                                                  lowerBound[1] + j * spacing[1],
                                                  lowerBound[2] + k * spacing[2]};
          std::array<T, 3> upperBoundExclusive = {lowerBound[0] + (i + 1) * spacing[0],
                                                  lowerBound[1] + (j + 1) * spacing[1],
                                                  lowerBound[2] + (k + 1) * spacing[2]};

          // make sure the maximum is still within bounds by scaling the upper bound with floating
          // point epsilon
          if (i == grid[0] - 1) upperBoundExclusive[0] = upperBound[0] * roundupFactor;
          if (j == grid[1] - 1) upperBoundExclusive[1] = upperBound[1] * roundupFactor;
          if (k == grid[2] - 1) upperBoundExclusive[2] = upperBound[2] * roundupFactor;

          partitions.emplace_back(UVWGroup<T>{lowerBoundInclusive, upperBoundExclusive});
        }
      }
    }

    return partitions;
}

// write all elements within group to the front and return a subview equal to the group size
// overwrites the input data
template <typename T>
inline auto apply_uvw_partition(const UVWGroup<T>& group, HostView<T, 2> uvw,
                                HostView<std::complex<T>, 1> virtualVis)
    -> std::pair<HostView<T, 2>, HostView<std::complex<T>, 1>> {

    const auto n = uvw.shape(0);

    std::size_t count = 0;

    T* __restrict__ u = uvw.slice_view(0).data();
    T* __restrict__ v = uvw.slice_view(1).data();
    T* __restrict__ w = uvw.slice_view(2).data();
    std::complex<T>* __restrict__ virtualVisPtr = virtualVis.data();

    for (std::size_t i = 0; i < n; ++i) {
      assert(i < uvw.shape(0));
      assert(i < virtualVis.shape(0));
      if (u[i] >= group.lowerBoundInclusive[0] && u[i] < group.upperBoundExclusive[0] &&
          v[i] >= group.lowerBoundInclusive[1] && v[i] < group.upperBoundExclusive[1] &&
          w[i] >= group.lowerBoundInclusive[2] && w[i] < group.upperBoundExclusive[2]) {
        u[count] = u[i];
        v[count] = v[i];
        w[count] = w[i];
        virtualVisPtr[count] = virtualVisPtr[i];
        ++count;
      }
    }

    return {uvw.sub_view({0, 0}, {count, uvw.shape(1)}), virtualVis.sub_view(0, count)};
}

}  // namespace host
}  // namespace bipp
