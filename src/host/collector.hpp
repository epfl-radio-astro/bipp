#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "collector_interface.hpp"
#include "context_internal.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace host {

template <typename T>
class Collector : public CollectorInterface<T> {
public:
  Collector(std::shared_ptr<ContextInternal> ctx, std::size_t numReserveSteps);

  auto collect(T wl, ConstView<std::complex<T>, 2> v, ConstHostView<T, 2> dMasked,
               ConstView<T, 2> xyzUvw) -> void override;

  auto serialize() const -> HostArray<char, 1> override;

  auto deserialize(ConstHostView<char, 1> serialData) -> void override;

  auto get_data(std::size_t idx) const -> typename CollectorInterface<T>::Data override;

private:
  std::shared_ptr<ContextInternal> ctx_;
  std::size_t numReserveSteps_;
  std::vector<T> wlData_;
  std::vector<HostArray<std::complex<T>, 2>> vData_;
  std::vector<HostArray<T, 2>> dMaskedData_;
  std::vector<HostArray<T, 2>> xyzUvwData_;
};

}  // namespace host
}  // namespace bipp
