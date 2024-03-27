#pragma once

#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

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
    Data(T wl, const std::size_t nVis, ConstView<std::complex<T>, 2> v, ConstHostView<T, 2> dMasked, ConstView<T, 2> xyzUvw)
         : wl(wl), nVis(nVis), v(v), dMasked(dMasked), xyzUvw(xyzUvw) {}

    T wl;
    std::size_t nVis;
    ConstView<std::complex<T>, 2> v;
    ConstHostView<T, 2> dMasked;
    ConstView<T, 2> xyzUvw;
  };

  virtual auto collect(T wl, const std::size_t nVis, ConstView<std::complex<T>, 2> v, ConstHostView<T, 2> dMasked,
                       ConstView<T, 2> xyzUvw) -> void = 0;

  virtual auto serialize() const -> HostArray<char, 1> = 0;

  virtual auto deserialize(ConstHostView<char, 1> serialData) -> void = 0;

  virtual auto get_data() const -> std::vector<typename CollectorInterface<T>::Data> = 0;

  virtual auto size() const -> std::size_t = 0;

  virtual auto get_nvis() const -> std::size_t = 0;

  virtual auto clear() -> void = 0;

  virtual ~CollectorInterface() = default;
};

}  // namespace bipp
