#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>

#include <string>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <variant>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

class BIPP_EXPORT ImageData {
public:
  virtual std::vector<std::string> tags() const = 0;

  virtual std::size_t num_tags() const = 0;

  virtual void get(const std::string& tag, float* image) = 0;

  virtual void set(const std::string& tag, const float* image) = 0;

  virtual std::size_t width() const = 0;

  virtual std::size_t height() const = 0;

  virtual float fov_deg() const = 0;

  virtual float ra_deg() const = 0;

  virtual float dec_deg() const = 0;

  virtual ~ImageData() = default;
};


/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
