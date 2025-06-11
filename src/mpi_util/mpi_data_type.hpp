#pragma once

#include <cassert>
#include <complex>
#include <cstring>
#include <memory>
#include <array>

#include "bipp/config.h"
#include "memory/view.hpp"

#ifdef BIPP_MPI

#include <mpi.h>

#include "mpi_util/mpi_check_status.hpp"

namespace bipp {


template <typename T>
struct MPIType;

template <>
struct MPIType<char> {
  inline static auto get() -> MPI_Datatype { return MPI_CHAR; }
};

template <>
struct MPIType<signed short int> {
  inline static auto get() -> MPI_Datatype { return MPI_SHORT; }
};

template <>
struct MPIType<signed int> {
  inline static auto get() -> MPI_Datatype { return MPI_INT; }
};

template <>
struct MPIType<signed long int> {
  inline static auto get() -> MPI_Datatype { return MPI_LONG; }
};

template <>
struct MPIType<signed long long int> {
  inline static auto get() -> MPI_Datatype { return MPI_LONG_LONG; }
};

template <>
struct MPIType<signed char> {
  inline static auto get() -> MPI_Datatype { return MPI_SIGNED_CHAR; }
};

template <>
struct MPIType<unsigned char> {
  inline static auto get() -> MPI_Datatype { return MPI_UNSIGNED_CHAR; }
};

template <>
struct MPIType<unsigned short int> {
  inline static auto get() -> MPI_Datatype { return MPI_UNSIGNED_SHORT; }
};

template <>
struct MPIType<unsigned int> {
  inline static auto get() -> MPI_Datatype { return MPI_UNSIGNED; }
};

template <>
struct MPIType<unsigned long int> {
  inline static auto get() -> MPI_Datatype { return MPI_UNSIGNED_LONG; }
};

template <>
struct MPIType<unsigned long long int> {
  inline static auto get() -> MPI_Datatype { return MPI_UNSIGNED_LONG_LONG; }
};

template <>
struct MPIType<float> {
  inline static auto get() -> MPI_Datatype { return MPI_FLOAT; }
};

template <>
struct MPIType<double> {
  inline static auto get() -> MPI_Datatype { return MPI_DOUBLE; }
};

template <>
struct MPIType<long double> {
  inline static auto get() -> MPI_Datatype { return MPI_LONG_DOUBLE; }
};

template <>
struct MPIType<std::complex<float>> {
  inline static auto get() -> MPI_Datatype { return MPI_C_COMPLEX; }
};

template <>
struct MPIType<std::complex<double>> {
  inline static auto get() -> MPI_Datatype { return MPI_C_DOUBLE_COMPLEX; }
};

// Storage for MPI datatypes
class MPIDatatypeHandle {
public:
  MPIDatatypeHandle() = default;

  // Create custom datatype with ownership
  // Does not call MPI_Type_commit!
  // Can take predifined MPI types such as MPI_DOUBLE, on which MPI_Type_free() will not be called
  // NOTE: Freeing a MPI_Datatype on which this type depends on does not affect this type (see "The
  // MPI core")
  MPIDatatypeHandle(const MPI_Datatype& mpiType) {
    assert(mpiType != MPI_DATATYPE_NULL);
    int numIntegers, numAddresses, numDatatypes, combiner;
    mpi_check_status(
        MPI_Type_get_envelope(mpiType, &numIntegers, &numAddresses, &numDatatypes, &combiner));
    if (combiner != MPI_COMBINER_NAMED && combiner != MPI_COMBINER_DUP) {
      // take ownership and call MPI_Type_free upon release
      type_ = std::shared_ptr<MPI_Datatype>(new MPI_Datatype(mpiType), [](MPI_Datatype* ptr) {
        assert(*ptr != MPI_DATATYPE_NULL);
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
          MPI_Type_free(ptr);
        }
        delete ptr;
      });
    } else {
      // only copy type descriptor, will not call MPI_Type_free()
      type_ = std::make_shared<MPI_Datatype>(mpiType);
    }
  }

  inline auto get() const -> const MPI_Datatype& {
    assert(type_);
    assert(*type_ != MPI_DATATYPE_NULL);
    return *type_;
  }

  inline auto empty() const noexcept -> bool { return type_ == nullptr; }

  template <typename T, std::size_t DIM>
  inline static auto create(const ConstView<T, DIM>& v) -> MPIDatatypeHandle {
    MPI_Datatype newType;

    if constexpr (DIM == 1) {
      mpi_check_status(MPI_Type_contiguous(v.size(), MPIType<T>::get(), &newType));
      mpi_check_status(MPI_Type_commit(&newType));
      return MPIDatatypeHandle(newType);
    }

    std::array<int, DIM> arrayOfSizes, arrayOfSubsizes, arrayOfStarts;

    for(std::size_t i = 0; i < DIM; ++i) {
      arrayOfStarts[i] = 0;
      arrayOfSubsizes[i] = v.shape(i);
      if (i < DIM - 1)
        arrayOfSizes[i] = v.strides(i + 1) / v.strides(i);
      else
        arrayOfSizes[i] = v.shape(i);
    }

    mpi_check_status(MPI_Type_create_subarray(DIM, arrayOfSizes.data(), arrayOfSubsizes.data(),
                                              arrayOfStarts.data(), MPI_ORDER_FORTRAN,
                                              MPIType<T>::get(), &newType));
    mpi_check_status(MPI_Type_commit(&newType));
    return MPIDatatypeHandle(newType);
  }

private:
  std::shared_ptr<MPI_Datatype> type_ = nullptr;
};

}  // namespace bipp

#endif
