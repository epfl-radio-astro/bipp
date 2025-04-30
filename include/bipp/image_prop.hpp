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

class BIPP_EXPORT ImageProp {
public:
  using MetaType = std::variant<std::size_t, float, std::vector<float>>;

  virtual void pixel_lmn(float* lmn, std::size_t ldlmn) = 0;

  virtual std::unordered_map<std::string, MetaType> meta_data() const = 0;

  virtual void set_meta(const std::string& name, const MetaType& data) = 0;

  virtual std::size_t num_pixel() const = 0;

  virtual ~ImageProp() = default;
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
