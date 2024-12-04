#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>

#include <string>
#include <cstddef>
#include <memory>
#include <vector>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

class BIPP_EXPORT ImageReader {
public:
  explicit ImageReader(const std::string& fileName);

  const std::vector<std::string>& tags() const;

  std::size_t num_tags() const;

  void read(const std::string& tag, float* image);

  const std::string& dataset_file_name() const;

  const std::string& dataset_description() const;

  std::size_t num_pixel() const;

  void close();

  bool is_open() const noexcept;

private:
  class ImageReaderImpl;
  struct ImageReaderImplDeleter {
    void operator()(ImageReaderImpl* p);
  };

  std::unique_ptr<ImageReaderImpl, ImageReaderImplDeleter> impl_;
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
