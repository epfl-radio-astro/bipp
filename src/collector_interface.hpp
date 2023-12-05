#pragma once

#include <complex>
#include <cstddef>
#include <type_traits>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {

template <typename T>
class CollectorInterface {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

  struct Data {
    Data(T wl, ConstHostView<std::complex<T>, 2> v, ConstHostView<T, 2> dMasked,
         ConstHostView<T, 2> xyzUvw)
        : wl(wl), v(v), dMasked(dMasked), xyzUvw(xyzUvw) {}

    T wl;
    ConstView<std::complex<T>, 2> v;
    ConstHostView<T, 2> dMasked;
    ConstView<T, 2> xyzUvw;
  };

  virtual auto collect(T wl, ConstView<std::complex<T>, 2> v, ConstHostView<T, 2> dMasked,
                       ConstView<T, 2> xyzUvw) -> void = 0;

  virtual auto serialize() const -> HostArray<char, 1> = 0;

  virtual auto deserialize(ConstHostView<char, 1> serialData) -> void = 0;

  virtual auto get_data(std::size_t idx) const -> typename CollectorInterface<T>::Data = 0;

  virtual ~CollectorInterface() = default;
};

}  // namespace bipp
