#pragma once

#include <complex>
#include <cstddef>
#include <cstring>

#include "bipp/bipp.hpp"
#include "bipp/config.h"

namespace bipp {
namespace host {

template <typename T>
auto apply_filter(BippFilter f, std::size_t nEig, const T* D, T* DFiltered) -> void {
  switch (f) {
    case BIPP_FILTER_STD: {
      for (std::size_t i = 0; i < nEig; ++i) {
        DFiltered[i] = 1;
      }
      break;
    }
    case BIPP_FILTER_SQRT: {
      for (std::size_t i = 0; i < nEig; ++i) {
        DFiltered[i] = std::sqrt(D[i]);
      }
      break;
    }
    case BIPP_FILTER_INV: {
      for (std::size_t i = 0; i < nEig; ++i) {
        const auto value = D[i];
        if (value)
          DFiltered[i] = 1 / value;
        else
          DFiltered[i] = 0;
      }
      break;
    }
    case BIPP_FILTER_INV_SQ: {
      for (std::size_t i = 0; i < nEig; ++i) {
        const auto value = D[i];
        if (value)
          DFiltered[i] = T(1) / (value * value);
        else
          DFiltered[i] = 0;
      }
      break;
    }
    case BIPP_FILTER_LSQ: {
      std::memcpy(DFiltered, D, nEig * sizeof(T));
      break;
    }
  }
}

}  // namespace host
}  // namespace bipp
