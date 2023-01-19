#pragma once

#include <algorithm>
#include <array>

#include "bipp/config.h"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
inline auto kernel_launch_grid(const api::DevicePropType& prop,
                               const std::array<std::size_t, 3>& size, const dim3& blockSize)
    -> dim3 {
  const std::array<std::size_t, 3> blockSizeT = {static_cast<std::size_t>(blockSize.x),
                                                 static_cast<std::size_t>(blockSize.y),
                                                 static_cast<std::size_t>(blockSize.z)};
  return dim3(static_cast<int>(std::min<std::size_t>((size[0] + blockSizeT[0] - 1) / blockSizeT[0],
                                                     prop.maxGridSize[0])),
              static_cast<int>(std::min<std::size_t>((size[1] + blockSizeT[1] - 1) / blockSizeT[1],
                                                     prop.maxGridSize[1])),
              static_cast<int>(std::min<std::size_t>((size[2] + blockSizeT[2] - 1) / blockSizeT[2],
                                                     prop.maxGridSize[2])));
}
}  // namespace gpu
}  // namespace bipp
