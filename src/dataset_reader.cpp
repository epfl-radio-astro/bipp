#include "bipp/dataset_reader.hpp"

#include <vector>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "context_internal.hpp"
#include "host/eigensolver.hpp"
#include "io/dataset_file_reader.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"
#include "memory/copy.hpp"

namespace bipp {
class DatasetReader::DatasetReaderImpl {
public:
  explicit DatasetReaderImpl(const std::string& fileName)
      : reader_(fileName) {}

  const std::string& description() const noexcept { return reader_.description(); }

  auto num_samples() const noexcept -> std::size_t { return reader_.num_samples(); }

  std::size_t num_antenna() const noexcept { return reader_.num_antenna(); }

  std::size_t num_beam() const noexcept { return reader_.num_beam(); }

  void read_eig_vec(std::size_t index, std::complex<float>* v, std::size_t ldv) {
    auto vView = HostView<std::complex<float>, 2>(v, {num_antenna(), num_beam()}, {1, ldv});
    if (vView.is_contiguous()) {
      reader_.read_eig_vec(index, vView);
    } else {
      std::vector<std::complex<float>> buffer(vView.size());
      auto bufferView =
          HostView<std::complex<float>, 2>(buffer.data(), vView.shape(), {1, vView.shape(0)});
      reader_.read_eig_vec(index, bufferView);
      copy(bufferView, vView);
    }
  }

  void read_eig_val(std::size_t index, float* d) {
    reader_.read_eig_val(index, HostView<float, 1>(d, num_beam(), 1));
  }

  void read_uvw(std::size_t index, float* uvw, std::size_t lduvw) {
    auto uvwView = HostView<float, 2>(uvw, {num_antenna() * num_antenna(), 3}, {1, lduvw});
    if (uvwView.is_contiguous()) {
      reader_.read_uvw(index, uvwView);
    } else {
      std::vector<float> buffer(uvwView.size());
      auto bufferView = HostView<float, 2>(buffer.data(), uvwView.shape(), {1, uvwView.shape(0)});
      reader_.read_uvw(index, bufferView);
      copy(bufferView, uvwView);
    }
  }

  void read_xyz(std::size_t index, float* xyz, std::size_t ldxyz) {
    auto xyzView = HostView<float, 2>(xyz, {num_antenna(), 3}, {1, ldxyz});
    if (xyzView.is_contiguous()) {
      reader_.read_xyz(index, xyzView);
    } else {
      std::vector<float> buffer(xyzView.size());
      auto bufferView = HostView<float, 2>(buffer.data(), xyzView.shape(), {1, xyzView.shape(0)});
      reader_.read_xyz(index, bufferView);
      copy(bufferView, xyzView);
    }
  }

  float read_wl(std::size_t index) { return reader_.read_wl(index); }

  std::size_t read_n_vis(std::size_t index) { return reader_.read_n_vis(index); }

private:
  DatasetFileReader reader_;
};

void DatasetReader::DatasetReaderImplDeleter::operator()(DatasetReaderImpl* p) { delete p; }

DatasetReader::DatasetReader(const std::string& fileName)
    : impl_(new DatasetReaderImpl(fileName)) {}

const std::string& DatasetReader::description() const {
  if (impl_)
    return impl_->description();
  else
    throw GenericError("DatasetReader: read after close");
}

std::size_t DatasetReader::num_samples() const {
  if (impl_)
    return impl_->num_samples();
  else
    throw GenericError("DatasetReader: read after close");
}

std::size_t DatasetReader::num_antenna() const {
  if (impl_)
    return impl_->num_antenna();
  else
    throw GenericError("DatasetReader: read after close");
}

std::size_t DatasetReader::num_beam() const {
  if (impl_)
    return impl_->num_beam();
  else
    throw GenericError("DatasetReader: read after close");
}

void DatasetReader::read_eig_vec(std::size_t index, std::complex<float>* v, std::size_t ldv) {
  if (impl_)
    impl_->read_eig_vec(index, v, ldv);
  else
    throw GenericError("DatasetReader: read after close");
}

void DatasetReader::read_eig_val(std::size_t index, float* d) {
  if (impl_)
    impl_->read_eig_val(index, d);
  else
    throw GenericError("DatasetReader: read after close");
}

void DatasetReader::read_uvw(std::size_t index, float* uvw, std::size_t lduvw) {
  if (impl_)
    impl_->read_uvw(index, uvw, lduvw);
  else
    throw GenericError("DatasetReader: read after close");
}

void DatasetReader::read_xyz(std::size_t index, float* xyz, std::size_t ldxyz) {
  if (impl_)
    impl_->read_xyz(index, xyz, ldxyz);
  else
    throw GenericError("DatasetReader: read after close");
}

float DatasetReader::read_wl(std::size_t index) {
  if (impl_)
    return impl_->read_wl(index);
  else
    throw GenericError("DatasetReader: read after close");
}

std::size_t DatasetReader::read_n_vis(std::size_t index) {
  if (impl_)
    return impl_->read_n_vis(index);
  else
    throw GenericError("DatasetReader: read after close");
}

void DatasetReader::close() { return impl_.reset(); }

bool DatasetReader::is_open() const noexcept { return bool(impl_); }

}  // namespace bipp
