#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>

#include <bipp/image.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

class BIPP_EXPORT ImageFile : public Image {
public:
  using MetaType = std::variant<std::size_t, float, std::vector<float>>;

  static ImageFile create(const std::string& fileName, std::size_t numPixel, const float* lmn,
                          std::size_t ldlmn);

  static ImageFile open(const std::string& fileName);

  std::unordered_map<std::string, MetaType> meta_data() const;

  void set_meta(const std::string& name, const MetaType& data);

  void close();

  bool is_open() const noexcept;

  std::vector<std::string> tags() const override;

  std::size_t num_tags() const override;

  void get(const std::string& tag, float* image) override;

  void set(const std::string& tag, const float* image) override;

  std::size_t num_pixel() const override;

  void pixel_lmn(float* lmn, std::size_t ldlmn) override;

private:
  class ImageFileImpl;
  struct ImageFileImplDeleter {
    void operator()(ImageFileImpl* p);
  };

  ImageFile(ImageFileImpl*);

  std::unique_ptr<ImageFileImpl, ImageFileImplDeleter> impl_;
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
