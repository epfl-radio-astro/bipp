#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "bipp/config.h"
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {

class ImageFileReader {
public:
  explicit ImageFileReader(const std::string& fileName);

  auto tags() const -> const std::vector<std::string>&;

  inline auto num_tags() const -> std::size_t { return tags().size(); }

  auto read(const std::string& tag, HostView<float, 1> image) -> void;

  auto dataset_file_name() const -> const std::string&;

  auto dataset_description() const -> const std::string&;

  auto num_pixel() const -> std::size_t;

private:
  class ImageFileReaderImpl;
  struct ImageFileReaderImplDeleter {
    auto operator()(ImageFileReaderImpl* p) -> void;
  };

  std::unique_ptr<ImageFileReaderImpl, ImageFileReaderImplDeleter> impl_;
};

}  // namespace bipp
