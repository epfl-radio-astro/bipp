#pragma once

#include <bipp/config.h>

#include <bipp/context.hpp>
#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

template <typename T>
class BIPP_EXPORT NufftSynthesis {
public:
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
  using valueType = T;

  /**
   * Create a nufft synthesis plan.
   *
   * @param[in] ctx Context handle.
   * @param[in] tol Target precision tolorance.
   * @param[in] nAntenna Number of antenna.
   * @param[in] nBeam Number of beam.
   * @param[in] nIntervals Number of intervals.
   * @param[in] nFilter Number of filter.
   * @param[in] filter Array of filters of size nFilter.
   * @param[in] nPixel Number of image pixels.
   * @param[in] lmnX Array of image x coordinates of size nPixel.
   * @param[in] lmnY Array of image y coordinates of size nPixel.
   * @param[in] lmnZ Array of image z coordinates of size nPixel.
   */
  NufftSynthesis(Context& ctx, T tol, std::size_t nAntenna, std::size_t nBeam,
                 std::size_t nIntervals, std::size_t nFilter, const BippFilter* filter,
                 std::size_t nPixel, const T* lmnX, const T* lmnY, const T* lmnZ);

  /**
   * Collect radio data.
   *
   * @param[in] nEig Number of eigenvalues.
   * @param[in] wl The wavelength.
   * @param[in] intervals 2D array of intervals of size (2, nIntervals).
   * @param[in] ldIntervals Leading dimension of intervals.
   * @param[in] s Optional complex 2D sensitivity array of size (nBeam, nBeam). May be null.
   * @param[in] lds Leading dimension of s.
   * @param[in] w 2D complex beamforming array of size (nAntenna, nBeam).
   * @param[in] ldw Leading dimension of w.
   * @param[in] xyz 2D antenna position array of size (nAntenna, 3).
   * @param[in] ldxyz Leading dimension of xyz.
   * @param[in] uvw UVW coordinates expressed in the local UVW frame of size (nAntenna * nAntenna,
   * 3).
   * @param[in] lduvw Leading dimension of uvw.
   */
  auto collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
               const std::complex<T>* s, std::size_t lds, const std::complex<T>* w, std::size_t ldw,
               const T* xyz, std::size_t ldxyz, const T* uvw, std::size_t lduvw) -> void;

  /**
   * Get image.
   *
   * @param[in] f Filter to get image for.
   * @param[out] img 2D image array of size (nPixel, nIntervals).
   * @param[in] ld Leading dimension of img.
   */
  auto get(BippFilter f, T* img, std::size_t ld) -> void;

private:
  /*! \cond PRIVATE */
  std::unique_ptr<void, std::function<void(void*)>> plan_;
  /*! \endcond */
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */