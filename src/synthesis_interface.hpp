#pragma once

#include <complex>
#include <cstddef>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "memory/view.hpp"
#include "context_internal.hpp"

namespace bipp {

enum class SynthesisType { Standard, NUFFT };

template <typename T>
struct ProcessData {
  ProcessData(T wl, ConstHostView<std::complex<T>, 2> v, ConstHostView<T, 2> dMasked,
              ConstHostView<T, 2> xyzUvw)
      : wl(wl), v(v), dMasked(dMasked), xyzUvw(xyzUvw) {}

  T wl;
  ConstHostView<std::complex<T>, 2> v;
  ConstHostView<T, 2> dMasked;
  ConstHostView<T, 2> xyzUvw;
};


template <typename T>
class SynthesisInterface {
public:
  virtual auto collect(T wl, ConstView<std::complex<T>, 2> vView, ConstHostView<T, 2> dMasked,
                       ConstView<T, 2> xyzUvwView) -> void = 0;

  virtual auto process(const std::vector<ProcessData<T>>& data) -> void {}

  virtual auto get(BippFilter f, View<T, 2> out) -> void = 0;

  virtual auto type() const -> SynthesisType = 0;

  virtual auto filter(std::size_t idx) const -> BippFilter = 0;

  virtual auto context() -> ContextInternal& = 0;

  virtual auto gpu_enabled() const -> bool = 0;

  virtual auto image() -> View<T, 3> = 0;

  virtual ~SynthesisInterface() = default;
};

}  // namespace bipp
