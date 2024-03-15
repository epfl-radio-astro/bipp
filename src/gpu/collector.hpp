#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "bipp/config.h"


#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "bipp/enums.h"
#include "collector_interface.hpp"
#include "context_internal.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace gpu {

template <typename T>
class Collector : public CollectorInterface<T> {
public:
  explicit Collector(std::shared_ptr<ContextInternal> ctx);

  auto collect(T wl, std::size_t nVis, ConstView<std::complex<T>, 2> v, ConstHostView<T, 2> dMasked,
               ConstView<T, 2> xyzUvw) -> void override;

  auto serialize() const -> HostArray<char, 1> override;

  auto deserialize(ConstHostView<char, 1> serialData) -> void override;

  auto get_data() const -> std::vector<typename CollectorInterface<T>::Data> override;

  auto size() const -> std::size_t override { return wlData_.size(); }

  auto get_nvis() const -> std::size_t override {
      return std::accumulate(nVisData_.begin(), nVisData_.end(), 0);
  }

  auto clear() -> void override;

private:
  std::shared_ptr<ContextInternal> ctx_;
  std::size_t numReserveSteps_;
  std::vector<T> wlData_;
  std::vector<std::size_t> nVisData_;
  std::vector<DeviceArray<std::complex<T>, 2>> vData_;
  std::vector<HostArray<T, 2>> dMaskedData_;
  std::vector<HostArray<T, 2>> xyzUvwData_;
};

}  // namespace gpu
}  // namespace bipp
#endif
