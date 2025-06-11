#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>

#include <bipp/image_prop.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

class BIPP_EXPORT ImagePropFile : public ImageProp {
public:
  using MetaType = std::variant<std::size_t, float, std::vector<float>>;

  static ImagePropFile create(const std::string& fileName, std::size_t height, std::size_t width, float fovDeg,
                    const float* lmn, std::size_t ldlmn);

  static ImagePropFile open(const std::string& fileName);

  void close();

  bool is_open() const noexcept;

  void pixel_lmn(float* lmn, std::size_t ldlmn) override;

  std::unordered_map<std::string, MetaType> meta_data() const override;

  void set_meta(const std::string& name, const MetaType& data) override;

  std::size_t width() const override;

  std::size_t height() const override;

  float fov_deg() const override;

private:
  class ImagePropFileImpl;
  struct ImageFileImplDeleter {
    void operator()(ImagePropFileImpl* p);
  };

  ImagePropFile(ImagePropFileImpl*);

  std::unique_ptr<ImagePropFileImpl, ImageFileImplDeleter> impl_;
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
