#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <string>

#include "bipp/config.h"
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {

class ImageFileWriter {
public:
  using ValueType = float;

  ImageFileWriter(const std::string& fileName, const std::string& datasetFileName,
                  const std::string& datasetDescription);

  auto write(const std::string& tag, ConstHostView<float, 1> image) -> void;

private:
  class ImageFileWriterImpl;
  struct ImageFileWriterImplDeleter {
    auto operator()(ImageFileWriterImpl* p) -> void;
  };

  std::unique_ptr<ImageFileWriterImpl, ImageFileWriterImplDeleter> impl_;
};

}  // namespace bipp
