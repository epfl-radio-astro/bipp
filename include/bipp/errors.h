#pragma once

#include <bipp/config.h>

enum BippError {
  /**
   * Success. No error.
   */
  BIPP_SUCCESS,
  /**
   * Unknown error.
   */
  BIPP_UNKNOWN_ERROR,
  /**
   * Internal error.
   */
  BIPP_INTERNAL_ERROR,
  /**
   * Invalid parameter error.
   */
  BIPP_INVALID_PARAMETER_ERROR,
  /**
   * Invalid pointer error.
   */
  BIPP_INVALID_POINTER_ERROR,
  /**
   * Invalid handle error.
   */
  BIPP_INVALID_HANDLE_ERROR,
  /**
   * Eigensolver error.
   */
  BIPP_EIGENSOLVER_ERROR,
  /**
   * fiNUFFT error.
   */
  BIPP_FINUFFT_ERROR,
  /**
   * Not Implemented error.
   */
  BIPP_NOT_IMPLEMENTED_ERROR,
  /**
   * GPU error.
   */
  BIPP_GPU_ERROR,
  /**
   * GPU support error.
   */
  BIPP_GPU_SUPPORT_ERROR,
  /**
   * GPU blas error.
   */
  BIPP_GPU_BLAS_ERROR,
  /**
   * Invalid allocator function error.
   */
  BIPP_INVALID_ALLOCATOR_FUNCTION,
  /**
   * MPI Error
   */
  BIPP_MPI_ERROR,
  /**
   * HDF5 error.
   */
  BIPP_HDF5_ERROR,
  /**
   * File error.
   */
  BIPP_FILE_ERROR
};

#ifndef __cplusplus
/*! \cond PRIVATE */
// C only
typedef enum BippError BippError;
/*! \endcond */
#endif  // cpp
