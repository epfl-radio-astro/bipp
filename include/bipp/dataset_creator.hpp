#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>

#include <string>
#include <complex>
#include <cstddef>
#include <memory>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

class BIPP_EXPORT DatasetCreator {
public:
  DatasetCreator(const std::string& fileName, const std::string& description, std::size_t nAntenna,
                 std::size_t nBeam);

  void process_and_write(float wl, const std::complex<float>* s, std::size_t lds,
                         const std::complex<float>* w, std::size_t ldw, const float* xyz,
                         std::size_t ldxyz, const float* uvw, std::size_t lduvw);

  void process_and_write(double wl, const std::complex<double>* s, std::size_t lds,
                         const std::complex<double>* w, std::size_t ldw, const double* xyz,
                         std::size_t ldxyz, const double* uvw, std::size_t lduvw);

  void write(float wl, std::size_t nVis, const std::complex<float>* v, std::size_t ldv,
             const float* d, const float* xyz, std::size_t ldxyz, const float* uvw,
             std::size_t lduvw);


  std::size_t num_antenna() const;

  std::size_t num_beam() const;

private:
  class DatasetCreatorImpl;
  struct DatasetCreatorImplDeleter {
    void operator()(DatasetCreatorImpl* p);
  };

  std::unique_ptr<DatasetCreatorImpl, DatasetCreatorImplDeleter> impl_;
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
