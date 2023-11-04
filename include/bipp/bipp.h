#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>
#include <bipp/errors.h>
#include <stddef.h>

typedef void* BippContext;
typedef void* BippNufftSynthesis;
typedef void* BippNufftSynthesisF;
typedef void* BippStandardSynthesis;
typedef void* BippStandardSynthesisF;
typedef void* BippNufftSynthesisOptions;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a context.
 *
 * @param[in] pu Processing unit to use. If BIPP_PU_AUTO, GPU will be used
 * if possible, CPU otherwise.
 * @param[out] ctx Context handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_ctx_create(BippProcessingUnit pu, BippContext* ctx);

/**
 * Destroy a context.
 *
 * @param[in] ctx Context handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_ctx_destroy(BippContext* ctx);

/**
 * Create Nufft Synthesis options.
 *
 * @param[out] opt Options handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_ns_options_create(BippNufftSynthesisOptions* opt);

/**
 * Destroy a Nufft Synthesis handle.
 *
 * @param[in] opt Options handle
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_ns_options_destroy(BippNufftSynthesisOptions* opt);

/**
 * Set Nufft tolerance parameter.
 *
 * @param[in] opt Options handle.
 * @param[in] tol Tolerance used for computing the Nufft.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_ns_options_set_tolerance(BippNufftSynthesisOptions opt, float tol);

/**
 * Set number of collected data packages to be processed together. Larger size will increase memory
 * usage but improve performance.
 *
 * @param[in] opt Options handle.
 * @param[in] size Group size.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_ns_options_set_collect_group_size(BippNufftSynthesisOptions opt,
                                                             size_t size);

/**
 * Set Nufft Synthesis image partition method to "auto".
 *
 * @param[in] opt Options handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_ns_options_set_local_image_partition_auto(BippNufftSynthesisOptions opt);

/**
 * Set Nufft Synthesis image partition method to "none".
 *
 * @param[in] opt Options handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_ns_options_set_local_image_partition_none(BippNufftSynthesisOptions opt);

/**
 * Set Nufft Synthesis image partition method to "grid".
 *
 * @param[in] opt Options handle.
 * @param[in] dimX Grid dimension in x.
 * @param[in] dimY Grid dimension in y.
 * @param[in] dimZ Grid dimension in z.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_ns_options_set_local_image_partition_grid(BippNufftSynthesisOptions opt,
                                                                     size_t dimX, size_t dimY,
                                                                     size_t dimZ);

/**
 * Set Nufft Synthesis uvw partition method to "auto".
 *
 * @param[in] opt Options handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_ns_options_set_local_uvw_partition_auto(BippNufftSynthesisOptions opt);

/**
 * Set Nufft Synthesis uvw partition method to "none".
 *
 * @param[in] opt Options handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_ns_options_set_local_uvw_partition_none(BippNufftSynthesisOptions opt);

/**
 * Set Nufft Synthesis uvw partition method to "grid".
 *
 * @param[in] opt Options handle.
 * @param[in] dimX Grid dimension in x.
 * @param[in] dimY Grid dimension in y.
 * @param[in] dimZ Grid dimension in z.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_ns_options_set_local_uvw_partition_grid(BippNufftSynthesisOptions opt,
                                                                   size_t dimX, size_t dimY,
                                                                   size_t dimZ);

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
 * @param[out] plan The output handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_nufft_synthesis_create_f(BippContext ctx, BippNufftSynthesisOptions opt,
                                                    size_t nAntenna, size_t nBeam,
                                                    size_t nIntervals, size_t nFilter,
                                                    const BippFilter* filter, size_t nPixel,
                                                    const float* lmnX, const float* lmnY,
                                                    const float* lmnZ, BippNufftSynthesisF* plan);

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
 * @param[out] plan The output handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_nufft_synthesis_create(BippContext ctx, BippNufftSynthesisOptions opt,
                                                  size_t nAntenna, size_t nBeam, size_t nIntervals,
                                                  size_t nFilter, const BippFilter* filter,
                                                  size_t nPixel, const double* lmnX,
                                                  const double* lmnY, const double* lmnZ,
                                                  BippNufftSynthesis* plan);

/**
 * Destroy a nufft synthesis plan.
 *
 * @param[in] plan Plan handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_nufft_synthesis_destroy_f(BippNufftSynthesisF* plan);

/**
 * Destroy a nufft synthesis plan.
 *
 * @param[in] plan Plan handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_nufft_synthesis_destroy(BippNufftSynthesis* plan);

/**
 * Collect radio data.
 *
 * @param[in] plan Plan handle.
 * @param[in] nEig Number of eigenvalues.
 * @param[in] wl The wavelength.
 * @param[in] intervals 2D array of intervals of size (2, nIntervals).
 * @param[in] ldIntervals Leading dimension of intervals.
 * @param[in] s Optional complex 2D sensitivity array of size (nBeam, nBeam). May be null.
 * @param[in] lds Leading dimension of s.
 * @param[in] w 2D beamforming array of size (nAntenna, nBeam).
 * @param[in] ldw Leading dimension of w.
 * @param[in] xyz 2D complex antenna position array of size (nAntenna, 3).
 * @param[in] ldxyz Leading dimension of xyz.
 * @param[in] uvw UVW coordinates expressed in the local UVW frame of size (nAntenna * nAntenna, 3).
 * @param[in] lduvw Leading dimension of uvw.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_nufft_synthesis_collect_f(BippNufftSynthesisF plan, size_t nEig,
                                                     float wl, const float* intervals,
                                                     size_t ldIntervals, const void* s, size_t lds,
                                                     const void* w, size_t ldw, const float* xyz,
                                                     size_t ldxyz, const float* uvw, size_t lduvw);

/**
 * Collect radio data.
 *
 * @param[in] plan Plan handle.
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
 * @param[in] uvw UVW coordinates expressed in the local UVW frame of size (nAntenna * nAntenna, 3).
 * @param[in] lduvw Leading dimension of uvw.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_nufft_synthesis_collect(BippNufftSynthesis plan, size_t nEig, double wl,
                                                   const double* intervals, size_t ldIntervals,
                                                   const void* s, size_t lds, const void* w,
                                                   size_t ldw, const double* xyz, size_t ldxyz,
                                                   const double* uvw, size_t lduvw);

/**
 * Get image.
 *
 * @param[in] plan Plan handle.
 * @param[in] f Filter to get image for.
 * @param[out] img 2D image array of size (nPixel, nIntervals).
 * @param[in] ld Leading dimension of img.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_nufft_synthesis_get_f(BippNufftSynthesisF plan, BippFilter f, float* img,
                                                 size_t ld);

/**
 * Get image.
 *
 * @param[in] plan Plan handle.
 * @param[in] f Filter to get image for.
 * @param[out] img 2D image array of size (nPixel, nIntervals).
 * @param[in] ld Leading dimension of img.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_nufft_synthesis_get(BippNufftSynthesisF plan, BippFilter f, double* img,
                                               size_t ld);

/**
 * Create a standard synthesis plan.
 *
 * @param[in] ctx Context handle.
 * @param[in] nAntenna Number of antenna.
 * @param[in] nBeam Number of beam.
 * @param[in] nIntervals Number of intervals.
 * @param[in] nFilter Number of filter.
 * @param[in] filter Array of filters of size nFilter.
 * @param[in] nPixel Number of image pixels.
 * @param[in] lmnX Array of image x coordinates of size nPixel.
 * @param[in] lmnY Array of image y coordinates of size nPixel.
 * @param[in] lmnZ Array of image z coordinates of size nPixel.
 * @param[out] plan The output handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_standard_synthesis_create_f(BippContext ctx, size_t nAntenna,
                                                       size_t nBeam, size_t nIntervals,
                                                       size_t nFilter, const BippFilter* filter,
                                                       size_t nPixel, const float* lmnX,
                                                       const float* lmnY, const float* lmnZ,
                                                       BippStandardSynthesisF* plan);

/**
 * Create a standard synthesis plan.
 *
 * @param[in] ctx Context handle.
 * @param[in] nAntenna Number of antenna.
 * @param[in] nBeam Number of beam.
 * @param[in] nIntervals Number of intervals.
 * @param[in] nFilter Number of filter.
 * @param[in] filter Array of filters of size nFilter.
 * @param[in] nPixel Number of image pixels.
 * @param[in] lmnX Array of image x coordinates of size nPixel.
 * @param[in] lmnY Array of image y coordinates of size nPixel.
 * @param[in] lmnZ Array of image z coordinates of size nPixel.
 * @param[out] plan The output handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_standard_synthesis_create(BippContext ctx, size_t nAntenna, size_t nBeam,
                                                     size_t nIntervals, size_t nFilter,
                                                     const BippFilter* filter, size_t nPixel,
                                                     const double* lmnX, const double* lmnY,
                                                     const double* lmnZ,
                                                     BippStandardSynthesis* plan);

/**
 * Destroy a standard synthesis plan.
 *
 * @param[in] plan Plan handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_standard_synthesis_destroy_f(BippStandardSynthesisF* plan);

/**
 * Destroy a standard synthesis plan.
 *
 * @param[in] plan Plan handle.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_standard_synthesis_destroy(BippStandardSynthesis* plan);

/**
 * Collect radio data.
 *
 * @param[in] plan Plan handle.
 * @param[in] nEig Number of eigenvalues.
 * @param[in] wl The wavelength.
 * @param[in] intervals 2D array of intervals of size (2, nIntervals).
 * @param[in] ldIntervals Leading dimension of intervals.
 * @param[in] s Optional 2D sensitivity array of size (nBeam, nBeam). May be null.
 * @param[in] lds Leading dimension of s.
 * @param[in] w 2D beamforming array of size (nAntenna, nBeam).
 * @param[in] ldw Leading dimension of w.
 * @param[in] xyz 2D antenna position array of size (nAntenna, 3).
 * @param[in] ldxyz Leading dimension of xyz.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_standard_synthesis_collect_f(BippStandardSynthesisF plan, size_t nEig,
                                                        float wl, const float* intervals,
                                                        size_t ldIntervals, const void* s,
                                                        size_t lds, const void* w, size_t ldw,
                                                        const float* xyz, size_t ldxyz);

/**
 * Collect radio data.
 *
 * @param[in] plan Plan handle.
 * @param[in] nEig Number of eigenvalues.
 * @param[in] wl The wavelength.
 * @param[in] intervals 2D array of intervals of size (2, nIntervals).
 * @param[in] ldIntervals Leading dimension of intervals.
 * @param[in] s Optional 2D sensitivity array of size (nBeam, nBeam). May be null.
 * @param[in] lds Leading dimension of s.
 * @param[in] w 2D beamforming array of size (nAntenna, nBeam).
 * @param[in] ldw Leading dimension of w.
 * @param[in] xyz 2D antenna position array of size (nAntenna, 3).
 * @param[in] ldxyz Leading dimension of xyz.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_standard_synthesis_collect(BippStandardSynthesis plan, size_t nEig,
                                                      double wl, const double* intervals,
                                                      size_t ldIntervals, const void* s, size_t lds,
                                                      const void* w, size_t ldw, const double* xyz,
                                                      size_t ldxyz);

/**
 * Get image.
 *
 * @param[in] plan Plan handle.
 * @param[in] f Filter to get image for.
 * @param[out] img 2D image array of size (nPixel, nIntervals).
 * @param[in] ld Leading dimension of img.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_standard_synthesis_get_f(BippStandardSynthesisF plan, BippFilter f,
                                                    float* img, size_t ld);

/**
 * Get image.
 *
 * @param[in] plan Plan handle.
 * @param[in] f Filter to get image for.
 * @param[out] img 2D image array of size (nPixel, nIntervals).
 * @param[in] ld Leading dimension of img.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_standard_synthesis_get(BippStandardSynthesisF plan, BippFilter f,
                                                  double* img, size_t ld);

/**
 * Compute the positive eigenvalues and eigenvectors of a hermitian matrix in
 * single precision. Optionally solves a general eigenvalue problem.
 *
 * @param[in] ctx Context handle.
 * @param[in] m Order of matrix A.
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
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_eigh_f(BippContext ctx, size_t m, size_t nEig, const void* a, size_t lda,
                                  const void* b, size_t ldb, size_t* nEigOut, float* d, void* v,
                                  size_t ldv);

/**
 * Compute the positive eigenvalues and eigenvectors of a hermitian matrix in
 * double precision. Optionally solves a general eigenvalue problem.
 *
 * @param[in] ctx Context handle.
 * @param[in] m Order of matrix A.
 * @param[in] nEig Maximum number of eigenvalues to compute.
 * @param[in] a Matrix A.
 * @param[in] lda Leading dimension of A.
 * @param[in] b Matrix B. Optional. When not null, a general eigenvalue problem
 * is solved.
 * @param[in] ldb Leading dimension of B.
 * @param[out] nEigOut Number of positive eigenvalues found.
 * @param[out] d Eigenvalues.
 * @param[out] v Eigenvectors stored as Matrix coloumns.
 * @param[out] ldv Leading of V.
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_eigh(BippContext ctx, size_t m, size_t nEig, const void* a, size_t lda,
                                const void* b, size_t ldb, size_t* nEigOut, double* d, void* v,
                                size_t ldv);

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
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_gram_matrix_f(BippContext ctx, size_t nAntenna, size_t nBeam,
                                         const void* w, size_t ldw, const float* xyz, size_t ldxyz,
                                         float wl, void* g, size_t ldg);

/**
 * Data processor for the gram matrix in double precision.
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
 * @return Error code or BIPP_SUCCESS.
 */
BIPP_EXPORT BippError bipp_gram_matrix(BippContext ctx, size_t nAntenna, size_t nBeam,
                                       const void* w, size_t ldw, const double* xyz, size_t ldxyz,
                                       double wl, void* g, size_t ldg);

#ifdef __cplusplus
}
#endif
