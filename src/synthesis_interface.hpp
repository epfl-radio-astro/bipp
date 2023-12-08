#pragma once

#include <complex>
#include <cstddef>
#include <memory>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "collector_interface.hpp"
#include "context_internal.hpp"
#include "memory/view.hpp"

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
  virtual auto process(CollectorInterface<T>& collector) -> void {}

  virtual auto get(BippFilter f, View<T, 2> out) -> void = 0;

  virtual auto type() const -> SynthesisType = 0;

  virtual auto filter(std::size_t idx) const -> BippFilter = 0;

  virtual auto context() -> const std::shared_ptr<ContextInternal>& = 0;

  virtual auto image() -> View<T, 3> = 0;

  virtual ~SynthesisInterface() = default;
};

}  // namespace bipp
