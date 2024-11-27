#pragma once

#include <bipp/config.h>
#include <bipp/errors.h>

#include <stdexcept>
#include <cstddef>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

/**
 * A generic error. Base type for all other exceptions.
 */
class BIPP_EXPORT GenericError : public std::exception {
public:
  const char* what() const noexcept override { return "BIPP: Generic error"; }

  virtual BippError error_code() const noexcept { return BippError::BIPP_UNKNOWN_ERROR; }
};

class BIPP_EXPORT InternalError : public GenericError {
public:
  InternalError() : msg_("BIPP: Internal error") {}

  // Only to be used with string literals
  template <std::size_t N>
  InternalError(const char (&msg)[N]) : msg_(msg) {}

  const char* what() const noexcept override { return msg_; }

  BippError error_code() const noexcept override { return BippError::BIPP_INTERNAL_ERROR; }

private:
  const char* msg_;
};

class BIPP_EXPORT InvalidParameterError : public GenericError {
public:
  const char* what() const noexcept override { return "BIPP: Invalid parameter error"; }

  BippError error_code() const noexcept override { return BippError::BIPP_INVALID_PARAMETER_ERROR; }
};

class BIPP_EXPORT InvalidPointerError : public GenericError {
public:
  const char* what() const noexcept override { return "BIPP: Invalid pointer error"; }

  BippError error_code() const noexcept override { return BippError::BIPP_INVALID_POINTER_ERROR; }
};

class BIPP_EXPORT InvalidAllocatorFunctionError : public GenericError {
public:
  const char* what() const noexcept override { return "BIPP: Invalid allocator function error"; }

  BippError error_code() const noexcept override {
    return BippError::BIPP_INVALID_ALLOCATOR_FUNCTION;
  }
};

class BIPP_EXPORT EigensolverError : public GenericError {
public:
  const char* what() const noexcept override { return "BIPP: Eigensolver error"; }

  BippError error_code() const noexcept override { return BippError::BIPP_EIGENSOLVER_ERROR; }
};

class BIPP_EXPORT FiNUFFTError : public GenericError {
public:
  const char* what() const noexcept override { return "BIPP: fiNUFFT error"; }

  BippError error_code() const noexcept override { return BippError::BIPP_FINUFFT_ERROR; }
};

class BIPP_EXPORT HDF5Error : public GenericError {
public:
  const char* what() const noexcept override { return "BIPP: HDF5Error error"; }

  BippError error_code() const noexcept override { return BippError::BIPP_HDF5_ERROR; }
};

class BIPP_EXPORT NotImplementedError : public GenericError {
public:
  const char* what() const noexcept override { return "BIPP: Not implemented"; }

  BippError error_code() const noexcept override { return BippError::BIPP_NOT_IMPLEMENTED_ERROR; }
};

class BIPP_EXPORT GPUError : public GenericError {
public:
  GPUError() : msg_("") {}

  GPUError(const char* msg) : msg_(msg) {}

  const char* what() const noexcept override { return msg_; }

  BippError error_code() const noexcept override { return BippError::BIPP_GPU_ERROR; }

private:
  const char* msg_;
};

class BIPP_EXPORT GPUSupportError : public GPUError {
public:
  GPUSupportError() : GPUError("BIPP: Not compiled with GPU support") {}

  BippError error_code() const noexcept override { return BippError::BIPP_GPU_SUPPORT_ERROR; }
};

class BIPP_EXPORT GPUBlasError : public GPUError {
public:
  GPUBlasError() : GPUError("BIPP: GPU BLAS error") {}

  GPUBlasError(const char* msg) : GPUError(msg) {}

  BippError error_code() const noexcept override { return BippError::BIPP_GPU_BLAS_ERROR; }
};

class BIPP_EXPORT MPIError : public GPUError {
public:
  MPIError() : GPUError("BIPP: MPI Error") {}

  BippError error_code() const noexcept override { return BippError::BIPP_MPI_ERROR; }
};

class BIPP_EXPORT FileError : public GenericError {
public:
  FileError() : msg_("") {}

  FileError(const char* msg) : msg_(msg) {}

  const char* what() const noexcept override { return msg_; }

  BippError error_code() const noexcept override { return BippError::BIPP_GPU_ERROR; }

private:
  const char* msg_;
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
