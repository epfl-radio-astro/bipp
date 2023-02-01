#pragma once

#include "bipp/config.h"
#include "gpu/util/runtime_api.hpp"

#ifdef BIPP_ROCM
#include <hipcub/hipcub.hpp>
#endif
#ifdef BIPP_CUDA
#include <cub/cub.cuh>
#endif

namespace bipp {
namespace gpu {
namespace api {
#ifdef BIPP_CUDA
namespace cub = ::cub;
#endif
#ifdef BIPP_ROCM
namespace cub = ::hipcub;
#endif
}  // namespace api
}  // namespace gpu
}  // namespace bipp
