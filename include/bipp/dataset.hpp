#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>

#include <string>
#include <complex>
#include <cstddef>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

class BIPP_EXPORT Dataset {
public:
  virtual const std::string& description() const = 0;

  virtual std::size_t num_samples() const = 0;

  virtual std::size_t num_antenna() const = 0;

  virtual std::size_t num_beam() const = 0;

  virtual void eig_vec(std::size_t index, std::complex<float>* v, std::size_t ldv) = 0;

  virtual void eig_val(std::size_t index, float* d) = 0;

  virtual void uvw(std::size_t index, float* uvw, std::size_t lduvw) = 0;

  virtual void uvw_min_max(std::size_t index, float* uvwMin, float* uvwMax) = 0;

  virtual void xyz(std::size_t index, float* xyz, std::size_t ldxyz) = 0;

  virtual float wl(std::size_t index) = 0;

  virtual float scale(std::size_t index) = 0;

  virtual ~Dataset() = default;
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
