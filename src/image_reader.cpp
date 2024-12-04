#include "bipp/image_reader.hpp"

#include <string>
#include <vector>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "context_internal.hpp"
#include "host/eigensolver.hpp"
#include "io/image_file_reader.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"

namespace bipp {
class ImageReader::ImageReaderImpl {
public:
  explicit ImageReaderImpl(const std::string& fileName) : reader_(fileName){}

  auto tags() const -> const std::vector<std::string>& { return reader_.tags(); };

  inline auto num_tags() const -> std::size_t { return reader_.num_tags(); }

  auto read(const std::string& tag, float* image) -> void {
    reader_.read(tag, HostView<float, 1>(image, reader_.num_pixel(), 1));
  }

  auto dataset_file_name() const -> const std::string& {
    return reader_.dataset_file_name();
  }

  auto dataset_description() const -> const std::string& {
    return reader_.dataset_description();
  }

  auto num_pixel() const -> std::size_t {
    return reader_.num_pixel();
  }

private:
  ImageFileReader reader_;
};

void ImageReader::ImageReaderImplDeleter::operator()(ImageReaderImpl* p) { delete p; }

ImageReader::ImageReader(const std::string& fileName)
    : impl_(new ImageReaderImpl(fileName)) {}

const std::vector<std::string>& ImageReader::tags() const {
  if (impl_)
    return impl_->tags();
  else
    throw GenericError("ImageReader: use after close");
}

std::size_t ImageReader::num_tags() const {
  if (impl_)
    return impl_->num_tags();
  else
    throw GenericError("ImageReader: use after close");
}

void ImageReader::read(const std::string& tag, float* image) {
  if (impl_)
    impl_->read(tag, image);
  else
    throw GenericError("ImageReader: use after close");
}

const std::string& ImageReader::dataset_file_name() const {
  if (impl_)
    return impl_->dataset_file_name();
  else
    throw GenericError("ImageReader: use after close");
}

const std::string& ImageReader::dataset_description() const {
  if (impl_)
    return impl_->dataset_description();
  else
    throw GenericError("ImageReader: use after close");
}

std::size_t ImageReader::num_pixel() const {
  if (impl_)
    return impl_->num_pixel();
  else
    throw GenericError("ImageReader: use after close");
}

void ImageReader::close() { return impl_.reset(); }

bool ImageReader::is_open() const noexcept { return bool(impl_); }

}  // namespace bipp
