#pragma once

#include <string>
#include <complex>
#include <memory>

#include "bipp/config.h"

#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {

class DatasetWriter {
public:
  using ValueType = float;

  DatasetWriter(const std::string& fileName, const std::string& datasetName, std::size_t nAntenna,
                std::size_t nBeam);

  auto write(ValueType wl, std::size_t nVis, ConstHostView<std::complex<ValueType>, 2> v,
             ConstHostView<ValueType, 1> d, ConstHostView<ValueType, 2> uvw) -> void;

private:
  class DatasetWriterImpl;
  struct DatasetWriterImplDeleter {
    auto operator()(DatasetWriterImpl* p) -> void;
  };

  std::unique_ptr<DatasetWriterImpl, DatasetWriterImplDeleter> impl_;
};

}  // namespace bipp
