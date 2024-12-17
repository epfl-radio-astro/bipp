#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>

#include <string>
#include <cstddef>
#include <vector>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

class BIPP_EXPORT Image {
public:
  virtual std::vector<std::string> tags() const = 0;

  virtual std::size_t num_tags() const = 0;

  virtual void get(const std::string& tag, float* image) = 0;

  virtual void set(const std::string& tag, const float* image) = 0;

  virtual std::size_t num_pixel() const = 0;

  virtual void pixel_lmn(float* lmn, std::size_t ldlmn) = 0;

  virtual ~Image() = default;
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
