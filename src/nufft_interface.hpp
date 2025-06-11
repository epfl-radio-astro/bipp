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
  virtual auto add(ConstHostView<T, 2> uvw, ConstHostView<std::complex<T>, 2> values) -> void = 0;

  virtual auto get_image(std::size_t imgIdx, HostView<float, 1> image) -> void = 0;

  virtual ~NUFFTInterface() = default;
};
}  // namespace bipp
