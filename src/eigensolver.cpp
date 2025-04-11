#include "host/eigensolver.hpp"

#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

#include "bipp/bipp.hpp"
#include "bipp/config.h"
#include "bipp/context.hpp"
#include "context_internal.hpp"
#include "memory/view.hpp"

namespace bipp {

template <typename T, typename>
BIPP_EXPORT auto eigh(T wl, std::size_t nAntenna, std::size_t nBeam, const std::complex<T>* s,
                      std::size_t lds, const std::complex<T>* w, std::size_t ldw, const T* xyz,
                      std::size_t ldxyz, T* d, std::complex<T>* v, std::size_t ldv)
    -> std::pair<std::size_t, T> {
  return host::eigh<T>(wl, ConstHostView<std::complex<T>, 2>(s, {nBeam, nBeam}, {1, lds}),
                       ConstHostView<std::complex<T>, 2>(w, {nAntenna, nBeam}, {1, ldw}),
                       ConstHostView<T, 2>(xyz, {nAntenna, 3}, {1, ldxyz}),
                       HostView<T, 1>(d, nBeam, 1),
                       HostView<std::complex<T>, 2>(v, {nAntenna, nBeam}, {1, ldv}));
}

template auto eigh<float, void>(float wl, std::size_t nAntenna, std::size_t nBeam,
                                const std::complex<float>* s, std::size_t lds,
                                const std::complex<float>* w, std::size_t ldw, const float* xyz,
                                std::size_t ldxyz, float* d, std::complex<float>* v,
                                std::size_t ldv) -> std::pair<std::size_t, float>;

template auto eigh<double, void>(double wl, std::size_t nAntenna, std::size_t nBeam,
                                 const std::complex<double>* s, std::size_t lds,
                                 const std::complex<double>* w, std::size_t ldw, const double* xyz,
                                 std::size_t ldxyz, double* d, std::complex<double>* v,
                                 std::size_t ldv) -> std::pair<std::size_t, double>;

}  // namespace bipp
