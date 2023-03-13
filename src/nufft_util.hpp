#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "bipp/config.h"

namespace bipp {
inline auto optimal_nufft_input_partition(std::array<double, 3> inputExtent,
                                          std::array<double, 3> outputExtent,
                                          std::size_t maxFFTGridSize)
    -> std::array<std::size_t, 3> {
  constexpr double upsampfactor = 2.0;  // upsampling factor option used for finufft
  constexpr double pi = 3.14159265358979323846;

  std::size_t fftGridSize = 1;
  for (std::size_t i = 0; i < inputExtent.size(); ++i) {
    fftGridSize *=
        static_cast<std::size_t>(2.0 * upsampfactor * inputExtent[i] * outputExtent[i] / pi);
  }

  const auto partitionSizeTarget = (fftGridSize + maxFFTGridSize - 1) / maxFFTGridSize;

  std::array<std::size_t, 3> gridSize = {1, 1, 1};
  std::array<double, 3> gridSpacing = inputExtent;
  while (gridSize[0] * gridSize[1] * gridSize[2] < partitionSizeTarget) {
    std::array<double, 3> sortedSpacing = gridSpacing;
    std::sort(sortedSpacing.begin(), sortedSpacing.end());

    const auto maxIndex =
        std::max_element(gridSpacing.begin(), gridSpacing.end()) - gridSpacing.begin();

    // Increase grid size at dimension with largest gridSpacing by multiplying with factor between
    // largest and second largest grid spacing
    gridSize[maxIndex] *= std::min(std::max<std::size_t>(2, sortedSpacing[2] / sortedSpacing[1]),
                                   partitionSizeTarget);
    gridSpacing[maxIndex] = inputExtent[maxIndex] / gridSize[maxIndex];
  }

  return gridSize;
}
}  // namespace bipp
