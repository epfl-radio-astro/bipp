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
                            std::size_t nAntenna, std::size_t nBeam, float raDeg, float decDeg);

  const std::string& description() const override;

  std::size_t num_samples() const override;

  std::size_t num_antenna() const override;

  std::size_t num_beam() const override;

  void eig_vec(std::size_t index, std::complex<float>* v, std::size_t ldv) override;

  void eig_val(std::size_t index, float* d) override;

  void uvw(std::size_t index, float* uvw, std::size_t lduvw) override;

  float wl(std::size_t index) override;

  float time_stamp(std::size_t index) override;

  float scale(std::size_t index) override;

  float ra_deg() const override;

  float dec_deg() const override;

  void write(float timeStamp, float wl, float scale, const std::complex<float>* v, std::size_t ldv,
             const float* d, const float* uvw, std::size_t lduvw);

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
