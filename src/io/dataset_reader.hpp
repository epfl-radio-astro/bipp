#pragma once

#include <string>
#include <complex>
#include <memory>
#include <cstddef>

#include "bipp/config.h"

#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {

class DatasetReader {
public:
  using ValueType = float;

  explicit DatasetReader(const std::string& fileName);

  auto name() const noexcept -> const std::string&;

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
  class DatasetReaderImpl;
  struct DatasetReaderImplDeleter {
    auto operator()(DatasetReaderImpl* p) -> void;
  };

  std::unique_ptr<DatasetReaderImpl, DatasetReaderImplDeleter> impl_;
};

}  // namespace bipp
