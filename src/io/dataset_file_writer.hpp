#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <string>

#include "bipp/config.h"
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {

class DatasetFileWriter {
public:
  using ValueType = float;

  DatasetFileWriter(const std::string& fileName, const std::string& description,
                    std::size_t nAntenna, std::size_t nBeam);

  auto write(ValueType wl, std::size_t nVis, ConstHostView<std::complex<ValueType>, 2> v,
             ConstHostView<ValueType, 1> d, ConstHostView<ValueType, 2> uvw,
             ConstHostView<ValueType, 2> xyz) -> void;

private:
  class DatasetFileWriterImpl;
  struct DatasetFileWriterImplDeleter {
    auto operator()(DatasetFileWriterImpl* p) -> void;
  };

  std::unique_ptr<DatasetFileWriterImpl, DatasetFileWriterImplDeleter> impl_;
};

}  // namespace bipp
