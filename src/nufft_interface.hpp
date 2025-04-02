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
  virtual auto transform_and_add(ConstHostView<std::complex<T>, 1> values, HostView<float, 1> out)
      -> void = 0;

  virtual ~NUFFTInterface() = default;
};
}  // namespace bipp
