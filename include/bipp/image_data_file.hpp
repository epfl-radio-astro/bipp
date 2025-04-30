#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>

#include <bipp/image_data.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

class BIPP_EXPORT ImageDataFile : public ImageData {
public:
  static ImageDataFile create(const std::string& fileName, std::size_t numPixel);

  static ImageDataFile open(const std::string& fileName);

  void close();

  bool is_open() const noexcept;

  std::vector<std::string> tags() const override;

  std::size_t num_tags() const override;

  void get(const std::string& tag, float* image) override;

  void set(const std::string& tag, const float* image) override;

  std::size_t num_pixel() const override;

private:
  class ImageDataFileImpl;
  struct ImageFileImplDeleter {
    void operator()(ImageDataFileImpl* p);
  };

  ImageDataFile(ImageDataFileImpl*);

  std::unique_ptr<ImageDataFileImpl, ImageFileImplDeleter> impl_;
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
