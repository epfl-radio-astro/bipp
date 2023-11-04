#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>

#include <bipp/context.hpp>
#include <bipp/exceptions.hpp>
#include <bipp/nufft_synthesis.hpp>
#include <bipp/standard_synthesis.hpp>
#include <complex>
#include <cstddef>
#include <type_traits>

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
 * @param[out] ldg Leading of G.
 */
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, double> || std::is_same_v<T, float>>>
BIPP_EXPORT auto gram_matrix(Context& ctx, std::size_t nAntenna, std::size_t nBeam,
                             const std::complex<T>* w, std::size_t ldw, const T* xyz,
                             std::size_t ldxyz, T wl, std::complex<T>* g, std::size_t ldg) -> void;

/**
 * Compute the positive eigenvalues and eigenvectors of a hermitian matrix in
 * single precision. Optionally solves a general eigenvalue problem.
 *
 * @param[in] ctx Context handle.
 * @param[in] nAntenna Order of matrix A.
 * @param[in] nEig Maximum number of eigenvalues to compute.
 * @param[in] a Hermitian matrix A. Only the lower triangle is read.
 * @param[in] lda Leading dimension of A.
 * @param[in] b Matrix B. Optional. When not null, a general eigenvalue problem
 * is solved.
 * @param[in] ldb Leading dimension of B.
 * @param[out] nEigOut Number of positive eigenvalues found.
 * @param[out] d Eigenvalues.
 * @param[out] v Eigenvectors stored as Matrix coloumns.
 * @param[out] ldv Leading of V.
 */
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, double> || std::is_same_v<T, float>>>
BIPP_EXPORT auto eigh(Context& ctx, std::size_t nAntenna, std::size_t nEig,
                      const std::complex<T>* a, std::size_t lda, const std::complex<T>* b,
                      std::size_t ldb, std::size_t* nEigOut, T* d, std::complex<T>* v,
                      std::size_t ldv) -> void;

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
