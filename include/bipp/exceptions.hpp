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
  GenericError() : msg_("BIPP: Generic error") {}

  // Only to be used with string literals
  template <std::size_t N>
  GenericError(const char (&msg)[N]) : msg_(msg) {}

  const char* what() const noexcept override { return msg_; }

  virtual BippError error_code() const noexcept { return BippError::BIPP_GENERIC_ERROR; }

private:
  const char* msg_;
};

class BIPP_EXPORT InternalError : public GenericError {
public:
  InternalError() : GenericError("BIPP: Internal error") {}

  template <std::size_t N>
  InternalError(const char (&msg)[N]) : GenericError(msg) {}

  BippError error_code() const noexcept override { return BippError::BIPP_INTERNAL_ERROR; }

private:
  const char* msg_;
};

class BIPP_EXPORT InvalidParameterError : public GenericError {
public:
  InvalidParameterError() : GenericError("BIPP: InvalidParameterError error") {}

  template <std::size_t N>
  InvalidParameterError(const char (&msg)[N]) : GenericError(msg) {}

  BippError error_code() const noexcept override { return BippError::BIPP_INVALID_PARAMETER_ERROR; }
};

class BIPP_EXPORT InvalidPointerError : public GenericError {
public:
  InvalidPointerError() : GenericError("BIPP: InvalidPointerError error") {}

  BippError error_code() const noexcept override { return BippError::BIPP_INVALID_POINTER_ERROR; }
};

class BIPP_EXPORT InvalidAllocatorFunctionError : public GenericError {
public:
  InvalidAllocatorFunctionError() : GenericError("BIPP: InvalidAllocatorFunctionError error") {}

  BippError error_code() const noexcept override {
    return BippError::BIPP_INVALID_ALLOCATOR_FUNCTION;
  }
};

class BIPP_EXPORT EigensolverError : public GenericError {
public:
  EigensolverError() : GenericError("BIPP: EigensolverError error") {}

  BippError error_code() const noexcept override { return BippError::BIPP_EIGENSOLVER_ERROR; }
};

class BIPP_EXPORT FiNUFFTError : public GenericError {
public:
  FiNUFFTError() : GenericError("BIPP: FiNUFFTError error") {}

  BippError error_code() const noexcept override { return BippError::BIPP_FINUFFT_ERROR; }
};

class BIPP_EXPORT HDF5Error : public GenericError {
public:
  HDF5Error() : GenericError("BIPP: HDF5Error error") {}

  template <std::size_t N>
  HDF5Error(const char (&msg)[N]) : GenericError(msg) {}

  BippError error_code() const noexcept override { return BippError::BIPP_HDF5_ERROR; }
};

class BIPP_EXPORT NotImplementedError : public GenericError {
public:
  NotImplementedError() : GenericError("BIPP: NotImplementedError error") {}

  BippError error_code() const noexcept override { return BippError::BIPP_NOT_IMPLEMENTED_ERROR; }
};

class BIPP_EXPORT GPUError : public GenericError {
public:
  GPUError() : msg_("BIPP: GPU Error") {}

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
  FileError() : GenericError("BIPP: FileError") {}

  template <std::size_t N>
  FileError(const char (&msg)[N]) : GenericError(msg) {}

  BippError error_code() const noexcept override { return BippError::BIPP_FILE_ERROR; }
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
