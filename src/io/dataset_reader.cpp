#include "io/dataset_reader.hpp"

#include <hdf5.h>

#include <memory>
#include <string>
#include <type_traits>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "io/h5_util.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"
#include "io/dataset_spec.hpp"

namespace bipp {

class DatasetReader::DatasetReaderImpl {
public:
  explicit DatasetReaderImpl(const std::string& fileName) {
    h5File_ = h5::check(H5Fopen(fileName.data(), H5F_ACC_RDONLY, H5P_DEFAULT));

    // check format version
    if(h5::read_size_attr(h5File_.id(), "formatVersionMajor" ) != datasetFormatVersionMajor) {
      throw FileError("Dataset format major version mismatch.");
    }

    if(h5::read_size_attr(h5File_.id(), "formatVersionMinor" ) > datasetFormatVersionMinor) {
      throw FileError("Dataset format minor version mismatch.");
    }

    // attributes
    nBeam_ = h5::read_size_attr(h5File_.id(), "nBeam");
    nAntenna_ = h5::read_size_attr(h5File_.id(), "nAntenna");
    name_ = h5::read_string_attr(h5File_.id(), "name");

    // create array types
    {
      std::array<hsize_t, 1> dims = {nBeam_};
      h5EigValType_ = h5::check(H5Tarray_create(h5::get_type_id<ValueType>(), dims.size(), dims.data()));
    }
    {
      std::array<hsize_t, 2> dims = {nBeam_, 2 * nAntenna_};
      h5EigVecType_ = h5::check(H5Tarray_create(h5::get_type_id<ValueType>(), dims.size(), dims.data()));
    }
    {
      std::array<hsize_t, 2> dims = {3, nAntenna_ * nAntenna_};
      h5UVWType_ = h5::check(H5Tarray_create(h5::get_type_id<ValueType>(), dims.size(), dims.data()));
    }
    {
      std::array<hsize_t, 2> dims = {3, nAntenna_};
      h5XYZType_ = h5::check(H5Tarray_create(h5::get_type_id<ValueType>(), dims.size(), dims.data()));
    }

    h5EigVal_ = h5::check(H5Dopen(h5File_.id(), "eigVal", H5P_DEFAULT));
    h5EigVec_ = h5::check(H5Dopen(h5File_.id(), "eigVec", H5P_DEFAULT));
    h5UVW_ = h5::check(H5Dopen(h5File_.id(), "uvw", H5P_DEFAULT));
    h5XYZ_ = h5::check(H5Dopen(h5File_.id(), "xyz", H5P_DEFAULT));
    h5Wl_ = h5::check(H5Dopen(h5File_.id(), "wl", H5P_DEFAULT));
    h5NVis_ = h5::check(H5Dopen(h5File_.id(), "nvis", H5P_DEFAULT));
    h5MinU_ = h5::check(H5Dopen(h5File_.id(), "uMin", H5P_DEFAULT));
    h5MinV_ = h5::check(H5Dopen(h5File_.id(), "vMin", H5P_DEFAULT));
    h5MinW_ = h5::check(H5Dopen(h5File_.id(), "wMin", H5P_DEFAULT));
    h5MaxU_ = h5::check(H5Dopen(h5File_.id(), "uMax", H5P_DEFAULT));
    h5MaxV_ = h5::check(H5Dopen(h5File_.id(), "vMax", H5P_DEFAULT));
    h5MaxW_ = h5::check(H5Dopen(h5File_.id(), "wMax", H5P_DEFAULT));

    // all samples have the same size. Select one to set number of samples.
    h5::DataSpace dspace = h5::check(H5Dget_space(h5NVis_.id()));
    hsize_t dim = 0;
    hsize_t maxDim = 0;
    h5::check(H5Sget_simple_extent_dims(dspace.id(), &dim, &maxDim));

    nSamples_ = dim;


    //TODO: set cache sizes?
  }

  auto name() const noexcept -> const std::string& {
    return name_;
  }

  auto num_samples() const noexcept -> std::size_t {
    return nSamples_;
  }

  auto num_antenna() const noexcept -> std::size_t {
    return nAntenna_;
  }

  auto num_beam() const noexcept -> std::size_t {
    return nBeam_;
  }

  auto read_eig_vec(std::size_t index, HostView<std::complex<ValueType>, 2> v) -> void {
    if(!v.is_contiguous()){
      throw InternalError("DatasetReader: eigenvector view must be contiguous.");
    }
    h5::read_single_element(index, h5EigVec_.id(), h5EigVecType_.id(), v.data());
  };

  auto read_eig_val(std::size_t index, HostView<ValueType, 1> d) -> void {
    h5::read_single_element(index, h5EigVal_.id(), h5EigValType_.id(), d.data());
  }

  auto read_uvw(std::size_t index, HostView<ValueType, 2> uvw) -> void {
    if(!uvw.is_contiguous()){
      throw InternalError("DatasetReader: uvw view must be contiguous.");
    }
    h5::read_single_element(index, h5UVW_.id(), h5UVWType_.id(), uvw.data());
  }

  auto read_xyz(std::size_t index, HostView<ValueType, 2> xyz) -> void {
    if(!xyz.is_contiguous()){
      throw InternalError("DatasetReader: xyz view must be contiguous.");
    }
    h5::read_single_element(index, h5XYZ_.id(), h5XYZType_.id(), xyz.data());
  }

  auto read_wl(std::size_t index) -> ValueType {
    ValueType value = 0;
    h5::read_single_element(index, h5Wl_.id(), h5::get_type_id<ValueType>(), &value);
    return value;
  }

  auto read_n_vis(std::size_t index) -> std::size_t {
    unsigned int value = 0;
    h5::read_single_element(index, h5Wl_.id(), h5::get_type_id<decltype(value)>(), &value);
    return value;
  }

private:
  hsize_t nAntenna_ = 0;
  hsize_t nBeam_ = 0;
  hsize_t nSamples_ = 0;
  std::string name_;

  // file
  h5::File h5File_ = H5I_INVALID_HID;

  // array types
  h5::DataType h5EigValType_ = H5I_INVALID_HID;
  h5::DataType h5EigVecType_ = H5I_INVALID_HID;
  h5::DataType h5UVWType_ = H5I_INVALID_HID;
  h5::DataType h5XYZType_ = H5I_INVALID_HID;

  // datasets
  h5::DataSet h5EigVal_;
  h5::DataSet h5EigVec_;
  h5::DataSet h5UVW_;
  h5::DataSet h5XYZ_;
  h5::DataSet h5Wl_;
  h5::DataSet h5NVis_;
  h5::DataSet h5MinU_;
  h5::DataSet h5MinV_;
  h5::DataSet h5MinW_;
  h5::DataSet h5MaxU_;
  h5::DataSet h5MaxV_;
  h5::DataSet h5MaxW_;
};

auto DatasetReader::DatasetReaderImplDeleter::operator()(DatasetReaderImpl* p) -> void {
  if (p) delete p;
}

DatasetReader::DatasetReader(const std::string& fileName)
    : impl_(new DatasetReaderImpl(fileName)) {}

auto DatasetReader::name() const noexcept -> const std::string& { return impl_->name(); }

auto DatasetReader::num_samples() const noexcept -> std::size_t { return impl_->num_samples(); }

auto DatasetReader::num_antenna() const noexcept -> std::size_t { return impl_->num_antenna(); }

auto DatasetReader::num_beam() const noexcept -> std::size_t { return impl_->num_beam(); }

auto DatasetReader::read_eig_vec(std::size_t index,
                                 HostView<std::complex<ValueType>, 2> v) -> void {
  impl_->read_eig_vec(index, v);
};

auto DatasetReader::read_eig_val(std::size_t index, HostView<ValueType, 1> d) -> void {
  impl_->read_eig_val(index, d);
}

auto DatasetReader::read_uvw(std::size_t index, HostView<ValueType, 2> uvw) -> void {
  impl_->read_uvw(index, uvw);
}

auto DatasetReader::read_xyz(std::size_t index, HostView<ValueType, 2> xyz) -> void {
  impl_->read_xyz(index, xyz);
}

auto DatasetReader::read_wl(std::size_t index) -> ValueType { return impl_->read_wl(index); }

auto DatasetReader::read_n_vis(std::size_t index) -> std::size_t {
  return impl_->read_n_vis(index);
}

}  // namespace bipp
