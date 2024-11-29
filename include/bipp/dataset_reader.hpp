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

class BIPP_EXPORT DatasetReader {
public:
  explicit DatasetReader(const std::string& fileName);

  const std::string& description() const;

  std::size_t num_samples() const;

  std::size_t num_antenna() const;

  std::size_t num_beam() const;

  void read_eig_vec(std::size_t index, std::complex<float>* v, std::size_t ldv);

  void read_eig_val(std::size_t index, float* d);

  void read_uvw(std::size_t index, float* uvw, std::size_t lduvw);

  void read_xyz(std::size_t index, float* xyz, std::size_t ldxyz);

  float read_wl(std::size_t index);

  std::size_t read_n_vis(std::size_t index);

  void close();

  bool is_open() const noexcept;

private:
  class DatasetReaderImpl;
  struct DatasetReaderImplDeleter {
    void operator()(DatasetReaderImpl* p);
  };

  std::unique_ptr<DatasetReaderImpl, DatasetReaderImplDeleter> impl_;
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
