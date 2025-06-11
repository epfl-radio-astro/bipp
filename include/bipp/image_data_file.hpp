#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>

#include <bipp/image_data.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

class BIPP_EXPORT ImageDataFile : public ImageData {
public:
  static ImageDataFile create(const std::string& fileName, std::size_t height, std::size_t width,
                              float fovDeg, float raDeg, float decDeg);

  static ImageDataFile open(const std::string& fileName);

  void close();

  bool is_open() const noexcept;

  std::vector<std::string> tags() const override;

  std::size_t num_tags() const override;

  void get(const std::string& tag, float* image) override;

  void set(const std::string& tag, const float* image) override;

  std::size_t width() const override;

  std::size_t height() const override;

  float fov_deg() const override;

  float ra_deg() const override;

  float dec_deg() const override;

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
