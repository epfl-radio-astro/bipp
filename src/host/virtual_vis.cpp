#include "host/virtual_vis.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <tuple>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "host/blas_api.hpp"
#include "host/kernels/apply_filter.hpp"
#include "host/kernels/interval_indices.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace host {

template <typename T>
auto virtual_vis(ContextInternal& ctx, std::size_t nFilter, const BippFilter* filter,
                 std::size_t nIntervals, const T* intervals, std::size_t ldIntervals,
                 std::size_t nEig, const T* D, std::size_t nAntenna, const std::complex<T>* V,
                 std::size_t ldv, std::size_t nBeam, const std::complex<T>* W, std::size_t ldw,
                 std::complex<T>* virtVis, std::size_t ldVirtVis1, std::size_t ldVirtVis2,
                 std::size_t ldVirtVis3, const std::size_t nz_vis) -> void {

  Buffer<std::complex<T>> VUnbeamBuffer;
  if (W) {
    VUnbeamBuffer = Buffer<std::complex<T>>(ctx.host_alloc(), nAntenna * nEig);
    blas::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nAntenna, nEig, nBeam, {1, 0}, W, ldw, V,
               ldv, {0, 0}, VUnbeamBuffer.get(), nAntenna);
    V = VUnbeamBuffer.get();
    ldv = nAntenna;
  }
  // V is always of shape (nAntenna, nEig) from here on

  auto VMulDBuffer = Buffer<std::complex<T>>(ctx.host_alloc(), nEig * nAntenna);

  auto DFilteredBuffer = Buffer<T>(ctx.host_alloc(), nEig);
  for (std::size_t i = 0; i < nFilter; ++i) {
    apply_filter(filter[i], nEig, D, DFilteredBuffer.get());
    const T* DFiltered = DFilteredBuffer.get();

    for (std::size_t j = 0; j < nIntervals; ++j) {
      std::size_t start, size;
      std::tie(start, size) = find_interval_indices(nEig, D, intervals[j * ldIntervals],
                                                    intervals[j * ldIntervals + 1]);

      auto virtVisCurrent = virtVis + i * ldVirtVis1 + j * ldVirtVis2;
      if (size) {
        // Multiply each col of V with the selected eigenvalue
        for (std::size_t k = 0; k < size; ++k) {
          auto VMulD = VMulDBuffer.get() + k * nAntenna;
          const auto VSelect = V + (start + k) * ldv;
          const auto DVal = nz_vis > 0 ? DFiltered[start + k] / nz_vis : DFiltered[start + k];

          ctx.logger().log(
              BIPP_LOG_LEVEL_DEBUG, "Assigning eigenvalue {} (filtered {}) to inverval [{}, {}]",
              D[start + k], DVal, intervals[j * ldIntervals], intervals[j * ldIntervals + 1]);
          // Handle sensitivity case where nz_vis is zero
          for (std::size_t l = 0; l < nAntenna; ++l) {
            VMulD[l] = VSelect[l] * DVal;
          }
        }

        // Matrix multiplication of the previously scaled V and the original V
        // with the selected eigenvalues
        blas::gemm(CblasColMajor, CblasNoTrans, CblasConjTrans, nAntenna, nAntenna, size, {1, 0},
                   VMulDBuffer.get(), nAntenna, V + start * ldv, ldv, {0, 0}, virtVisCurrent,
                   ldVirtVis3);

      } else {
        for (std::size_t k = 0; k < nAntenna; ++k) {
          std::memset(static_cast<void*>(virtVisCurrent + k * ldVirtVis3), 0,
                      nAntenna * sizeof(std::complex<T>));
        }
      }
    }
  }
}

template auto virtual_vis<float>(ContextInternal& ctx, std::size_t nFilter,
                                 const BippFilter* filter, std::size_t nIntervals,
                                 const float* intervals, std::size_t ldIntervals, std::size_t nEig,
                                 const float* D, std::size_t nAntenna, const std::complex<float>* V,
                                 std::size_t ldv, std::size_t nBeam, const std::complex<float>* W,
                                 std::size_t ldw, std::complex<float>* virtVis,
                                 std::size_t ldVirtVis1, std::size_t ldVirtVis2,
                                 std::size_t ldVirtVis3, const std::size_t nz_vis) -> void;

template auto virtual_vis<double>(ContextInternal& ctx, std::size_t nFilter,
                                  const BippFilter* filter, std::size_t nIntervals,
                                  const double* intervals, std::size_t ldIntervals,
                                  std::size_t nEig, const double* D, std::size_t nAntenna,
                                  const std::complex<double>* V, std::size_t ldv, std::size_t nBeam,
                                  const std::complex<double>* W, std::size_t ldw,
                                  std::complex<double>* virtVis, std::size_t ldVirtVis1,
                                  std::size_t ldVirtVis2, std::size_t ldVirtVis3,
                                  const std::size_t nz_vis) -> void;

}  // namespace host
}  // namespace bipp
