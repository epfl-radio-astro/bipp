#pragma once

#include <complex>
#include <cstddef>
#include <memory>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "memory/view.hpp"


namespace bipp {
template <typename T>
class NUFFTInterface {
public:
  virtual auto add_input(ConstHostView<T, 2> uvw, ConstHostView<std::complex<T>, 1> values)
      -> void = 0;

  virtual auto set_output_points(ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                                 ConstHostView<T, 1> pixelZ) -> void = 0;

  virtual auto transform_and_add(HostView<float, 1> out) -> void = 0;

  virtual ~NUFFTInterface() = default;
};
}  // namespace bipp
