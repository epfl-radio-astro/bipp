#pragma once

#include <bipp/config.h>

#include <bipp/enums.h>
#include <bipp/context.hpp>
#include <bipp/communicator.hpp>
#include <bipp/exceptions.hpp>
#include <bipp/nufft_synthesis.hpp>
#include <bipp/standard_synthesis.hpp>

#ifdef BIPP_MPI
#include <bipp/communicator.hpp>
#endif

#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>



/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

/**
 * Data processor for the gram matrix in single precision.
 *
 * @param[in] ctx Context handle.
 * @param[in] nAntenna Number of antenna.
 * @param[in] nBeam Number of beams.
 * @param[in] w Beamforming matrix.
 * @param[in] ldw Leading dimension of W.
 * @param[in] xyz Three dimensional antenna coordinates, where each coloumn
 * represents one dimension.
 * @param[in] ldxyz Leading dimension of xyz.
 * @param[in] wl Wavelength for which to compute the gram matrix.
 * @param[out] g Gram matrix.
 * @param[in] ldg Leading of G.
 */
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, double> || std::is_same_v<T, float>>>
BIPP_EXPORT auto gram_matrix(Context& ctx, std::size_t nAntenna, std::size_t nBeam,
                             const std::complex<T>* w, std::size_t ldw, const T* xyz,
                             std::size_t ldxyz, T wl, std::complex<T>* g, std::size_t ldg) -> void;

/**
 * Compute eigenvalues.
 *
 * @param[in] ctx Context handle.
 * @param[in] wl Wavelength.
 * @param[in] nAntenna Number of antenna.
 * @param[in] nBeam Number of beam.
 * @param[in] s Optional 2D sensitivity array of size (nBeam, nBeam). May be null.
 * @param[in] lds Leading dimension of s.
 * @param[in] w 2D beamforming array of size (nAntenna, nBeam).
 * @param[in] ldw Leading dimension of w.
 * @param[in] xyz 2D antenna position array of size (nAntenna, 3).
 * @param[in] ldxyz Leading dimension of xyz.
 * @param[out] d Eigenvalues. Size nBeam. Zero padded if number of computed eigenvalues < nBeam.
 * @return A pair consisting of the number of computed eigenvalues and the number of (non-zero) processed visibilities.
 */
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, double> || std::is_same_v<T, float>>>
BIPP_EXPORT auto eigh(Context& ctx, T wl, std::size_t nAntenna, std::size_t nBeam,
                      const std::complex<T>* s, std::size_t lds, const std::complex<T>* w,
                      std::size_t ldw, const T* xyz, std::size_t ldxyz, T* d) -> std::pair<std::size_t, std::size_t>;

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
