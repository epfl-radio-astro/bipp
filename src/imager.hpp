#pragma once

#include <complex>
#include <cstddef>
#include <memory>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "context_internal.hpp"
#include "memory/view.hpp"
#include "synthesis_interface.hpp"

namespace bipp {

template <typename T>
class Imager {
public:
  static auto standard_synthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nLevel,
                                 ConstView<BippFilter, 1> filter, ConstView<T, 1> pixelX,
                                 ConstView<T, 1> pixelY, ConstView<T, 1> pixelZ) -> Imager<T>;

  auto collect(T wl, const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc,
               ConstView<std::complex<T>, 2> s, ConstView<std::complex<T>, 2> w,
               ConstView<T, 2> xyz, ConstView<T, 2> uvw) -> void;

  auto get(BippFilter f, T* out, std::size_t ld) -> void;

private:
  explicit Imager(std::unique_ptr<SynthesisInterface<T>> syn) : synthesis_(std::move(syn)) {}

  std::unique_ptr<SynthesisInterface<T>> synthesis_;
};

}  // namespace bipp
