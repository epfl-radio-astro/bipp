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

class DatasetCreator {
public:
  DatasetCreator(const std::string& fileName, const std::string& description, std::size_t nAntenna,
                 std::size_t nBeam);

  void process_and_write(std::size_t nAntenna, std::size_t nBeam, float wl,
                         const std::complex<float>* s, std::size_t lds,
                         const std::complex<float>* w, std::size_t ldw, const float* xyz,
                         std::size_t ldxyz, const float* uvw, std::size_t lduvw);

  void process_and_write(std::size_t nAntenna, std::size_t nBeam, double wl,
                         const std::complex<double>* s, std::size_t lds,
                         const std::complex<double>* w, std::size_t ldw, const double* xyz,
                         std::size_t ldxyz, const double* uvw, std::size_t lduvw);

  void write(float wl, std::size_t nVis, const std::complex<float>* v, std::size_t ldv,
             const float* d, const float* xyz, std::size_t ldxyz, const float* uvw,
             std::size_t lduvw);

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
