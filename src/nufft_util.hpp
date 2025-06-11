#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "bipp/config.h"
//---
#include "memory/view.hpp"

namespace bipp {
template <typename T, typename MemFunc>
auto optimal_parition_size(ConstHostView<T, 2> uvw, std::array<T, 3> xyzMin,
                           std::array<T, 3> xyzMax, unsigned long long maxMem, MemFunc&& memFunc)
    -> std::array<std::size_t, 3> {
  auto uView = uvw.slice_view(0);
  auto vView = uvw.slice_view(1);
  auto wView = uvw.slice_view(2);
  auto uMinMax = std::minmax_element(uView.data(), uView.data() + uView.size());
  auto vMinMax = std::minmax_element(vView.data(), vView.data() + vView.size());
  auto wMinMax = std::minmax_element(wView.data(), wView.data() + wView.size());
  const std::array<T, 3> uvwMin = {*uMinMax.first, *vMinMax.first, *wMinMax.first};
  const std::array<T, 3> uvwMax = {*uMinMax.second, *vMinMax.second, *wMinMax.second};

  const std::array<T, 3> xyzExtent = {xyzMax[0] - xyzMin[0], xyzMax[1] - xyzMin[1],
                                      xyzMax[2] - xyzMin[2]};

  std::array<T, 3> uvwExtent = {uvwMax[0] - uvwMin[0], uvwMax[1] - uvwMin[1],
                                uvwMax[2] - uvwMin[2]};

  std::array<std::size_t, 3> grid = {1, 1, 1};

  while (memFunc(uvwMin,
                 {uvwMin[0] + uvwExtent[0], uvwMin[1] + uvwExtent[1], uvwMin[2] + uvwExtent[2]},
                 xyzMin, xyzMax) > maxMem) {
    std::size_t maxIdx = 0;
    if (uvwExtent[1] * xyzExtent[1] >= uvwExtent[0] * xyzExtent[0] &&
        uvwExtent[1] * xyzExtent[1] >= uvwExtent[2] * xyzExtent[2]) {
      maxIdx = 1;
    } else if (uvwExtent[2] * xyzExtent[2] >= uvwExtent[0] * xyzExtent[0] &&
               uvwExtent[2] * xyzExtent[2] >= uvwExtent[1] * xyzExtent[1]) {
      maxIdx = 2;
    }

    grid[maxIdx] += 1;
    uvwExtent[maxIdx] = (uvwMax[maxIdx] - uvwMin[maxIdx]) / grid[maxIdx];
  }

  return grid;
}
}  // namespace bipp
