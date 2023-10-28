#pragma once

#include <complex>
#include <cstddef>
#include <cstring>

#include "bipp/bipp.hpp"
#include "bipp/config.h"

namespace bipp {
namespace host {

// Compute the location of the interval [a, b] within the ascending or descending array D.
// Returns the first index and size. Assuming n is small -> linear search should
// suffice
template <typename T>
auto find_interval_indices(std::size_t n, const T* D, T a, T b)
    -> std::tuple<std::size_t, std::size_t> {
  if (!n) return {0, 0};
  std::size_t l = n;
  std::size_t r = 0;

  T prev = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto value = D[i];
    if (i > 0 && value > prev) break;
    if (value <= b && value >= a) {
      if (i < l) l = i;
      if (i > r) r = i;
    }
    prev = D[i];
  }

  return {l, l <= r ? r - l + 1 : 0};
}

}  // namespace host
}  // namespace bipp
