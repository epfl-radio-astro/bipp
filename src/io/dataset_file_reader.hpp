#pragma once

#include <string>
#include <complex>
#include <memory>
#include <cstddef>

#include "bipp/config.h"

#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {

class DatasetFileReader {
public:
  using ValueType = float;

  explicit DatasetFileReader(const std::string& fileName);

  auto description() const noexcept -> const std::string&;

  auto num_samples() const noexcept -> std::size_t;

  auto num_antenna() const noexcept -> std::size_t;

  auto num_beam() const noexcept -> std::size_t;

  auto read_eig_vec(std::size_t index, HostView<std::complex<ValueType>, 2> v) -> void;

  auto read_eig_val(std::size_t index, HostView<ValueType, 1> d) -> void;

  auto read_uvw(std::size_t index, HostView<ValueType, 2> uvw) -> void;

  auto read_xyz(std::size_t index, HostView<ValueType, 2> xyz) -> void;

  auto read_wl(std::size_t index) -> ValueType;

  auto read_n_vis(std::size_t index) -> std::size_t;

private:
  class DatasetFileReaderImpl;
  struct DatasetFileReaderImplDeleter {
    auto operator()(DatasetFileReaderImpl* p) -> void;
  };

  std::unique_ptr<DatasetFileReaderImpl, DatasetFileReaderImplDeleter> impl_;
};

}  // namespace bipp
