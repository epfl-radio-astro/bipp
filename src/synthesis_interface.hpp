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

  virtual auto get(View<T, 2> out) -> void = 0;

  virtual auto type() const -> SynthesisType = 0;

  virtual auto context() -> const std::shared_ptr<ContextInternal>& = 0;

  virtual auto image() -> View<T, 2> = 0;

  virtual auto normalize_by_nvis() const -> bool = 0;

  virtual ~SynthesisInterface() = default;
};

}  // namespace bipp
