#include "bipp/dataset_file.hpp"

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "context_internal.hpp"
#include "io/dataset_spec.hpp"
#include "io/h5_util.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"

namespace bipp {

class DatasetFile::DatasetFileImpl {
public:
  // create new dataset
  DatasetFileImpl(const std::string& fileName, const std::string& description, std::size_t nAntenna,
                  std::size_t nBeam)
      : ctx_(BIPP_PU_CPU), nAntenna_(nAntenna), nBeam_(nBeam), description_(description) {
    h5File_ = h5::check(H5Fcreate(fileName.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));

    // attributes
    h5::create_size_attr(h5File_.id(), "formatVersionMajor", datasetFormatVersionMajor);
    h5::create_size_attr(h5File_.id(), "formatVersionMinor", datasetFormatVersionMinor);
    h5::create_size_attr(h5File_.id(), "nBeam", nBeam);
    h5::create_size_attr(h5File_.id(), "nAntenna", nAntenna);
    h5::create_string_attr(h5File_.id(), "description",
                           description.size() ? description : " ");

    // create array types
    {
      std::array<hsize_t, 1> dims = {nBeam};
      h5EigValType_ =
          h5::check(H5Tarray_create(h5::get_type_id<float>(), dims.size(), dims.data()));
    }
    {
      std::array<hsize_t, 2> dims = {nBeam, 2 * nAntenna};
      h5EigVecType_ =
          h5::check(H5Tarray_create(h5::get_type_id<float>(), dims.size(), dims.data()));
    }
    {
      std::array<hsize_t, 2> dims = {3, nAntenna * nAntenna};
      h5UVWType_ =
          h5::check(H5Tarray_create(h5::get_type_id<float>(), dims.size(), dims.data()));
    }

    {
      std::array<hsize_t, 2> dims = {3, nAntenna};
      h5XYZType_ =
          h5::check(H5Tarray_create(h5::get_type_id<float>(), dims.size(), dims.data()));
    }

    // TODO: set optimal chunk size individually
    constexpr hsize_t CHUNK_SIZE = 5;

    // create data spaces
    h5Wl_ = h5::create_one_dim_space(h5File_.id(), "wl", h5::get_type_id<float>(), CHUNK_SIZE);
    h5Scale_ = h5::create_one_dim_space(h5File_.id(), "scale", h5::get_type_id<float>(),
                                       CHUNK_SIZE);
    h5MinU_ =
        h5::create_one_dim_space(h5File_.id(), "uMin", h5::get_type_id<float>(), CHUNK_SIZE);
    h5MinV_ =
        h5::create_one_dim_space(h5File_.id(), "vMin", h5::get_type_id<float>(), CHUNK_SIZE);
    h5MinW_ =
        h5::create_one_dim_space(h5File_.id(), "wMin", h5::get_type_id<float>(), CHUNK_SIZE);
    h5MaxU_ =
        h5::create_one_dim_space(h5File_.id(), "uMax", h5::get_type_id<float>(), CHUNK_SIZE);
    h5MaxV_ =
        h5::create_one_dim_space(h5File_.id(), "vMax", h5::get_type_id<float>(), CHUNK_SIZE);
    h5MaxW_ =
        h5::create_one_dim_space(h5File_.id(), "wMax", h5::get_type_id<float>(), CHUNK_SIZE);

    h5EigVal_ = h5::create_one_dim_space(h5File_.id(), "eigVal", h5EigValType_.id(), CHUNK_SIZE);
    h5EigVec_ = h5::create_one_dim_space(h5File_.id(), "eigVec", h5EigVecType_.id(), CHUNK_SIZE);
    h5UVW_ = h5::create_one_dim_space(h5File_.id(), "uvw", h5UVWType_.id(), CHUNK_SIZE);
    h5XYZ_ = h5::create_one_dim_space(h5File_.id(), "xyz", h5XYZType_.id(), CHUNK_SIZE);
  }

  // open existing dataset
  explicit DatasetFileImpl(const std::string& fileName) : ctx_(BIPP_PU_CPU) {
    h5File_ = h5::check(H5Fopen(fileName.data(), H5F_ACC_RDWR, H5P_DEFAULT));

    // check format version
    if (h5::read_size_attr(h5File_.id(), "formatVersionMajor") != datasetFormatVersionMajor) {
      throw FileError("Dataset format major version mismatch.");
    }

    if (h5::read_size_attr(h5File_.id(), "formatVersionMinor") > datasetFormatVersionMinor) {
      throw FileError("Dataset format minor version mismatch.");
    }

    // attributes
    nBeam_ = h5::read_size_attr(h5File_.id(), "nBeam");
    nAntenna_ = h5::read_size_attr(h5File_.id(), "nAntenna");
    description_ = h5::read_string_attr(h5File_.id(), "description");

    // create array types
    {
      std::array<hsize_t, 1> dims = {nBeam_};
      h5EigValType_ =
          h5::check(H5Tarray_create(h5::get_type_id<float>(), dims.size(), dims.data()));
    }
    {
      std::array<hsize_t, 2> dims = {nBeam_, 2 * nAntenna_};
      h5EigVecType_ =
          h5::check(H5Tarray_create(h5::get_type_id<float>(), dims.size(), dims.data()));
    }
    {
      std::array<hsize_t, 2> dims = {3, nAntenna_ * nAntenna_};
      h5UVWType_ = h5::check(H5Tarray_create(h5::get_type_id<float>(), dims.size(), dims.data()));
    }
    {
      std::array<hsize_t, 2> dims = {3, nAntenna_};
      h5XYZType_ = h5::check(H5Tarray_create(h5::get_type_id<float>(), dims.size(), dims.data()));
    }

    h5EigVal_ = h5::check(H5Dopen(h5File_.id(), "eigVal", H5P_DEFAULT));
    h5EigVec_ = h5::check(H5Dopen(h5File_.id(), "eigVec", H5P_DEFAULT));
    h5UVW_ = h5::check(H5Dopen(h5File_.id(), "uvw", H5P_DEFAULT));
    h5XYZ_ = h5::check(H5Dopen(h5File_.id(), "xyz", H5P_DEFAULT));
    h5Wl_ = h5::check(H5Dopen(h5File_.id(), "wl", H5P_DEFAULT));
    h5Scale_ = h5::check(H5Dopen(h5File_.id(), "scale", H5P_DEFAULT));
    h5MinU_ = h5::check(H5Dopen(h5File_.id(), "uMin", H5P_DEFAULT));
    h5MinV_ = h5::check(H5Dopen(h5File_.id(), "vMin", H5P_DEFAULT));
    h5MinW_ = h5::check(H5Dopen(h5File_.id(), "wMin", H5P_DEFAULT));
    h5MaxU_ = h5::check(H5Dopen(h5File_.id(), "uMax", H5P_DEFAULT));
    h5MaxV_ = h5::check(H5Dopen(h5File_.id(), "vMax", H5P_DEFAULT));
    h5MaxW_ = h5::check(H5Dopen(h5File_.id(), "wMax", H5P_DEFAULT));

    // all samples have the same size. Select one to set number of samples.
    h5::DataSpace dspace = h5::check(H5Dget_space(h5Scale_.id()));
    hsize_t dim = 0;
    hsize_t maxDim = 0;
    h5::check(H5Sget_simple_extent_dims(dspace.id(), &dim, &maxDim));

    nSamples_ = dim;

    // TODO: set cache sizes?
  }

  auto description() const noexcept -> const std::string& { return description_; }

  auto num_samples() const noexcept -> std::size_t { return nSamples_; }

  auto num_antenna() const noexcept -> std::size_t { return nAntenna_; }

  auto num_beam() const noexcept -> std::size_t { return nBeam_; }

  auto eig_vec(std::size_t index, std::complex<float>* v, std::size_t ldv) -> void {
    if(ldv == nAntenna_){
      h5::read_single_element(index, h5EigVec_.id(), h5EigVecType_.id(), v);
    } else {
      HostArray<std::complex<float>, 2> buffer(ctx_.host_alloc(), {nAntenna_, nBeam_});
      h5::read_single_element(index, h5EigVec_.id(), h5EigVecType_.id(), buffer.data());
      copy(buffer, HostView<std::complex<float>, 2>(v, buffer.shape(), {1, ldv}));
    }
  };

  auto eig_val(std::size_t index, float* d) -> void {
    h5::read_single_element(index, h5EigVal_.id(), h5EigValType_.id(), d);
  }

  auto uvw(std::size_t index, float* uvw, std::size_t lduvw) -> void {
    if (lduvw == nAntenna_ * nAntenna_) {
      h5::read_single_element(index, h5UVW_.id(), h5UVWType_.id(), uvw);
    } else {
      HostArray<float, 2> buffer(ctx_.host_alloc(), {nAntenna_ * nAntenna_, 3});
      h5::read_single_element(index, h5UVW_.id(), h5UVWType_.id(), buffer.data());
      copy(buffer, HostView<float, 2>(uvw, buffer.shape(), {1, lduvw}));
    }
  }

  auto uvw_min_max(std::size_t index, float* uvwMin, float* uvwMax) -> void {
    h5::read_single_element(index, h5MinU_.id(), h5::get_type_id<float>(), uvwMin);
    h5::read_single_element(index, h5MinV_.id(), h5::get_type_id<float>(), uvwMin + 1);
    h5::read_single_element(index, h5MinW_.id(), h5::get_type_id<float>(), uvwMin + 2);

    h5::read_single_element(index, h5MaxU_.id(), h5::get_type_id<float>(), uvwMax);
    h5::read_single_element(index, h5MaxV_.id(), h5::get_type_id<float>(), uvwMax + 1);
    h5::read_single_element(index, h5MaxW_.id(), h5::get_type_id<float>(), uvwMax + 2);
  }

  auto xyz(std::size_t index, float* xyz, std::size_t ldxyz) -> void {
    if (ldxyz == nAntenna_ * nAntenna_) {
      h5::read_single_element(index, h5XYZ_.id(), h5XYZType_.id(), xyz);
    } else {
      HostArray<float, 2> buffer(ctx_.host_alloc(), {nAntenna_ * nAntenna_, 3});
      h5::read_single_element(index, h5XYZ_.id(), h5XYZType_.id(), buffer.data());
      copy(buffer, HostView<float, 2>(xyz, buffer.shape(), {1, ldxyz}));
    }
  }


  auto wl(std::size_t index) -> float {
    float value = 0;
    h5::read_single_element(index, h5Wl_.id(), h5::get_type_id<float>(), &value);
    return value;
  }

  auto scale(std::size_t index) -> float {
    float value = 0;
    h5::read_single_element(index, h5Scale_.id(), h5::get_type_id<decltype(value)>(), &value);
    return value;
  }

  auto write(float wl, float scale, const std::complex<float>* v, std::size_t ldv, const float* d,
             const float* xyz, std::size_t ldxyz, const float* uvw, std::size_t lduvw) -> void {
    ConstHostView<std::complex<float>, 2> vView(v, {nAntenna_, nBeam_}, {1, ldv});
    ConstHostView<float, 2> uvwView (uvw, {nAntenna_ * nAntenna_, 3}, {1, lduvw});
    ConstHostView<float, 2> xyzView(xyz, {nAntenna_, 3}, {1, ldxyz});

    HostArray<std::complex<float>, 2> vArray;
    HostArray<float, 2> uvwArray;
    HostArray<float, 2> xyzArray;


    if(!vView.is_contiguous()) {
      vArray = HostArray<std::complex<float>,2>(ctx_.host_alloc(), vView.shape());
      copy(vView, vArray);
      vView = vArray;
    }

    if(!uvwView.is_contiguous()) {
      uvwArray = HostArray<float, 2>(ctx_.host_alloc(), uvwView.shape());
      copy(uvwView, uvwArray);
      uvwView = uvwArray;
    }

    if(!xyzView.is_contiguous()) {
      xyzArray = HostArray<float, 2>(ctx_.host_alloc(), xyzView.shape());
      copy(xyzView, xyzArray);
      xyzView = xyzArray;
    }

    // extend datasets
    const auto index = nSamples_;
    const auto newSize = index + 1;

    h5::check(H5Dextend(h5EigVal_.id(), &newSize));
    h5::check(H5Dextend(h5EigVec_.id(), &newSize));
    h5::check(H5Dextend(h5UVW_.id(), &newSize));
    h5::check(H5Dextend(h5XYZ_.id(), &newSize));
    h5::check(H5Dextend(h5Wl_.id(), &newSize));
    h5::check(H5Dextend(h5Scale_.id(), &newSize));
    h5::check(H5Dextend(h5MinU_.id(), &newSize));
    h5::check(H5Dextend(h5MinV_.id(), &newSize));
    h5::check(H5Dextend(h5MinW_.id(), &newSize));
    h5::check(H5Dextend(h5MaxU_.id(), &newSize));
    h5::check(H5Dextend(h5MaxV_.id(), &newSize));
    h5::check(H5Dextend(h5MaxW_.id(), &newSize));

    // increase countr once datasets have been extended
    ++nSamples_;

    // write
    h5::write_single_element(index, h5EigVal_.id(), h5EigValType_.id(), d);
    h5::write_single_element(index, h5EigVec_.id(), h5EigVecType_.id(), v);
    h5::write_single_element(index, h5UVW_.id(), h5UVWType_.id(), uvwView.data());
    h5::write_single_element(index, h5XYZ_.id(), h5XYZType_.id(), xyzView.data());

    h5::write_single_element(index, h5Wl_.id(), h5::get_type_id<float>(), &wl);
    h5::write_single_element(index, h5Scale_.id(), h5::get_type_id<decltype(scale)>(),
                             &scale);

    const auto uMM = std::minmax_element(&uvwView[{0, 0}], &uvwView[{0, 0}] + uvwView.shape(0));
    const auto vMM = std::minmax_element(&uvwView[{0, 1}], &uvwView[{0, 1}] + uvwView.shape(0));
    const auto wMM = std::minmax_element(&uvwView[{0, 2}], &uvwView[{0, 2}] + uvwView.shape(0));

    h5::write_single_element(index, h5MinU_.id(), h5::get_type_id<float>(), uMM.first);
    h5::write_single_element(index, h5MaxU_.id(), h5::get_type_id<float>(), uMM.second);

    h5::write_single_element(index, h5MinV_.id(), h5::get_type_id<float>(), vMM.first);
    h5::write_single_element(index, h5MaxV_.id(), h5::get_type_id<float>(), vMM.second);

    h5::write_single_element(index, h5MinW_.id(), h5::get_type_id<float>(), wMM.first);
    h5::write_single_element(index, h5MaxW_.id(), h5::get_type_id<float>(), wMM.second);
  }

private:
  ContextInternal ctx_;

  hsize_t nAntenna_ = 0;
  hsize_t nBeam_ = 0;
  hsize_t nSamples_ = 0;
  std::string description_;

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
  h5::DataSet h5Scale_;
  h5::DataSet h5MinU_;
  h5::DataSet h5MinV_;
  h5::DataSet h5MinW_;
  h5::DataSet h5MaxU_;
  h5::DataSet h5MaxV_;
  h5::DataSet h5MaxW_;
};

void DatasetFile::DatasetFileImplDeleter::operator()(DatasetFileImpl* p) { delete p; }

DatasetFile::DatasetFile(DatasetFileImpl* impl) : impl_(impl) {}

DatasetFile DatasetFile::open(const std::string& fileName) {
  return DatasetFile(new DatasetFileImpl(fileName));
}

DatasetFile DatasetFile::create(const std::string& fileName, const std::string& description,
                            std::size_t nAntenna, std::size_t nBeam) {
  return DatasetFile(new DatasetFileImpl(fileName, description, nAntenna, nBeam));
}

const std::string& DatasetFile::description() const {
  if (impl_)
    return impl_->description();
  else
    throw GenericError("DatasetFile: access after close");
}

std::size_t DatasetFile::num_samples() const {
  if (impl_)
    return impl_->num_samples();
  else
    throw GenericError("DatasetFile: access after close");
}

std::size_t DatasetFile::num_antenna() const {
  if (impl_)
    return impl_->num_antenna();
  else
    throw GenericError("DatasetFile: access after close");
}

std::size_t DatasetFile::num_beam() const {
  if (impl_)
    return impl_->num_beam();
  else
    throw GenericError("DatasetFile: access after close");
}

void DatasetFile::eig_vec(std::size_t index, std::complex<float>* v, std::size_t ldv) {
  if (impl_)
    impl_->eig_vec(index, v, ldv);
  else
    throw GenericError("DatasetFile: access after close");
}

void DatasetFile::eig_val(std::size_t index, float* d) {
  if (impl_)
    impl_->eig_val(index, d);
  else
    throw GenericError("DatasetFile: access after close");
}

void DatasetFile::uvw(std::size_t index, float* uvw, std::size_t lduvw) {
  if (impl_)
    impl_->uvw(index, uvw, lduvw);
  else
    throw GenericError("DatasetFile: access after close");
}

void DatasetFile::uvw_min_max(std::size_t index, float* uvwMin, float* uvwMax) {
  if (impl_)
    impl_->uvw_min_max(index, uvwMin, uvwMax);
  else
    throw GenericError("DatasetFile: access after close");
}

void DatasetFile::xyz(std::size_t index, float* xyz, std::size_t ldxyz) {
  if (impl_)
    impl_->xyz(index, xyz, ldxyz);
  else
    throw GenericError("DatasetFile: access after close");
}

float DatasetFile::wl(std::size_t index) {
  if (impl_)
    return impl_->wl(index);
  else
    throw GenericError("DatasetFile: access after close");
}

float DatasetFile::scale(std::size_t index) {
  if (impl_)
    return impl_->scale(index);
  else
    throw GenericError("DatasetFile: access after close");
}

void DatasetFile::write(float wl, float scale, const std::complex<float>* v,
                        std::size_t ldv, const float* d, const float* xyz, std::size_t ldxyz,
                        const float* uvw, std::size_t lduvw) {
  if (impl_)
    impl_->write(wl, scale, v, ldv, d, xyz, ldxyz, uvw, lduvw);
  else
    throw GenericError("DatasetFile: write after close");
}

void DatasetFile::close() { return impl_.reset(); }

bool DatasetFile::is_open() const noexcept { return bool(impl_); }

}  // namespace bipp
