#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>

#include <bipp/dataset.hpp>
#include <complex>
#include <cstddef>
#include <memory>
#include <string>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */


class BIPP_EXPORT DatasetFile : public Dataset {
public:
  static DatasetFile open(const std::string& fileName);

  static DatasetFile create(const std::string& fileName, const std::string& description,
                        std::size_t nAntenna, std::size_t nBeam);

  const std::string& description() const override;

  std::size_t num_samples() const override;

  std::size_t num_antenna() const override;

  std::size_t num_beam() const override;

  void eig_vec(std::size_t index, std::complex<float>* v, std::size_t ldv) override;

  void eig_val(std::size_t index, float* d) override;

  void uvw(std::size_t index, float* uvw, std::size_t lduvw) override;

  void uvw_min_max(std::size_t index, float* uvwMin, float* uvwMax) override;

  void xyz(std::size_t index, float* xyz, std::size_t ldxyz) override;

  float wl(std::size_t index) override;

  std::size_t n_vis(std::size_t index) override;

  // void process_and_write(float wl, const std::complex<float>* s, std::size_t lds,
  //                        const std::complex<float>* w, std::size_t ldw, const float* xyz,
  //                        std::size_t ldxyz, const float* uvw, std::size_t lduvw);

  // void process_and_write(double wl, const std::complex<double>* s, std::size_t lds,
  //                        const std::complex<double>* w, std::size_t ldw, const double* xyz,
  //                        std::size_t ldxyz, const double* uvw, std::size_t lduvw);

  void write(float wl, std::size_t nVis, const std::complex<float>* v, std::size_t ldv,
             const float* d, const float* xyz, std::size_t ldxyz, const float* uvw,
             std::size_t lduvw);

  void close();

  bool is_open() const noexcept;

private:
  class DatasetFileImpl;
  struct DatasetFileImplDeleter {
    void operator()(DatasetFileImpl* p);
  };

  DatasetFile(DatasetFileImpl*);

  std::unique_ptr<DatasetFileImpl, DatasetFileImplDeleter> impl_;
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
