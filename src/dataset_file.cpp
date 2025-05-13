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
                  std::size_t nBeam, float raDeg, float decDeg)
      : fileName_(fileName),
        ctx_(BIPP_PU_CPU),
        nAntenna_(nAntenna),
        nBeam_(nBeam),
        description_(description),
        raDeg_(raDeg),
        decDeg_(decDeg) {
    h5File_ = h5::check(H5Fcreate(fileName.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));

    // attributes
    h5::create_size_attr(h5File_.id(), "formatVersionMajor", datasetFormatVersionMajor);
    h5::create_size_attr(h5File_.id(), "formatVersionMinor", datasetFormatVersionMinor);
    h5::create_size_attr(h5File_.id(), "nBeam", nBeam);
    h5::create_size_attr(h5File_.id(), "nAntenna", nAntenna);
    h5::create_string_attr(h5File_.id(), "description",
                           description.size() ? description : " ");
    h5::create_float_attr(h5File_.id(), "raDeg", raDeg);
    h5::create_float_attr(h5File_.id(), "decDeg", decDeg);

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
    h5Time_ = h5::create_one_dim_space(h5File_.id(), "time", h5::get_type_id<float>(), CHUNK_SIZE);
    h5Scale_ = h5::create_one_dim_space(h5File_.id(), "scale", h5::get_type_id<float>(),
                                       CHUNK_SIZE);
    h5EigVal_ = h5::create_one_dim_space(h5File_.id(), "eigVal", h5EigValType_.id(), CHUNK_SIZE);
    h5EigVec_ = h5::create_one_dim_space(h5File_.id(), "eigVec", h5EigVecType_.id(), CHUNK_SIZE);
    h5UVW_ = h5::create_one_dim_space(h5File_.id(), "uvw", h5UVWType_.id(), CHUNK_SIZE);
  }

  // open existing dataset
  DatasetFileImpl(const std::string& fileName, bool readOnly)
      : fileName_(fileName), ctx_(BIPP_PU_CPU) {
    if (readOnly) {
      h5File_ = h5::check(H5Fopen(fileName.data(), H5F_ACC_RDONLY, H5P_DEFAULT));
    } else {
      h5File_ = h5::check(H5Fopen(fileName.data(), H5F_ACC_RDWR, H5P_DEFAULT));
    }

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
    raDeg_ = h5::read_float_attr(h5File_.id(), "raDeg");
    decDeg_ = h5::read_float_attr(h5File_.id(), "decDeg");

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
    h5Wl_ = h5::check(H5Dopen(h5File_.id(), "wl", H5P_DEFAULT));
    h5Time_ = h5::check(H5Dopen(h5File_.id(), "time", H5P_DEFAULT));
    h5Scale_ = h5::check(H5Dopen(h5File_.id(), "scale", H5P_DEFAULT));

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

  auto file_name() const noexcept -> const std::string& { return fileName_; }

  float ra_deg() const {return raDeg_;}

  float dec_deg() const { return decDeg_; }

  auto is_read_only() const -> bool {
    unsigned flag = 0;
    h5::check(H5Fget_intent(h5File_.id(), &flag));
    if (flag == H5F_ACC_RDONLY) {
      return true;
    }
    return false;
  }

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

  auto wl(std::size_t index) -> float {
    float value = 0;
    h5::read_single_element(index, h5Wl_.id(), h5::get_type_id<float>(), &value);
    return value;
  }

  auto time_stamp(std::size_t index) -> float {
    float value = 0;
    h5::read_single_element(index, h5Time_.id(), h5::get_type_id<float>(), &value);
    return value;
  }

  auto scale(std::size_t index) -> float {
    float value = 0;
    h5::read_single_element(index, h5Scale_.id(), h5::get_type_id<decltype(value)>(), &value);
    return value;
  }

  auto write(float timeStamp, float wl, float scale, const std::complex<float>* v, std::size_t ldv,
             const float* d, const float* uvw, std::size_t lduvw) -> void {
    ConstHostView<std::complex<float>, 2> vView(v, {nAntenna_, nBeam_}, {1, ldv});
    ConstHostView<float, 2> uvwView (uvw, {nAntenna_ * nAntenna_, 3}, {1, lduvw});

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

    // extend datasets
    const auto index = nSamples_;
    const auto newSize = index + 1;

    h5::check(H5Dextend(h5EigVal_.id(), &newSize));
    h5::check(H5Dextend(h5EigVec_.id(), &newSize));
    h5::check(H5Dextend(h5UVW_.id(), &newSize));
    h5::check(H5Dextend(h5Wl_.id(), &newSize));
    h5::check(H5Dextend(h5Time_.id(), &newSize));
    h5::check(H5Dextend(h5Scale_.id(), &newSize));

    // increase countr once datasets have been extended
    ++nSamples_;

    // write
    h5::write_single_element(index, h5EigVal_.id(), h5EigValType_.id(), d);
    h5::write_single_element(index, h5EigVec_.id(), h5EigVecType_.id(), v);
    h5::write_single_element(index, h5UVW_.id(), h5UVWType_.id(), uvwView.data());

    h5::write_single_element(index, h5Wl_.id(), h5::get_type_id<float>(), &wl);
    h5::write_single_element(index, h5Time_.id(), h5::get_type_id<float>(), &timeStamp);
    h5::write_single_element(index, h5Scale_.id(), h5::get_type_id<decltype(scale)>(),
                             &scale);
  }

private:
  std::string fileName_;

  ContextInternal ctx_;

  hsize_t nAntenna_ = 0;
  hsize_t nBeam_ = 0;
  hsize_t nSamples_ = 0;
  std::string description_;
  float raDeg_, decDeg_;

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
  h5::DataSet h5Wl_;
  h5::DataSet h5Time_;
  h5::DataSet h5Scale_;
};

void DatasetFile::DatasetFileImplDeleter::operator()(DatasetFileImpl* p) { delete p; }

DatasetFile::DatasetFile(DatasetFileImpl* impl) : impl_(impl) {}

DatasetFile DatasetFile::open(const std::string& fileName) {
  return DatasetFile(new DatasetFileImpl(fileName, true));
}

DatasetFile DatasetFile::create(const std::string& fileName, const std::string& description,
                                std::size_t nAntenna, std::size_t nBeam, float raDeg,
                                float decDeg) {
  return DatasetFile(new DatasetFileImpl(fileName, description, nAntenna, nBeam, raDeg, decDeg));
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

float DatasetFile::wl(std::size_t index) {
  if (impl_)
    return impl_->wl(index);
  else
    throw GenericError("DatasetFile: access after close");
}

float DatasetFile::time_stamp(std::size_t index) {
  if (impl_)
    return impl_->time_stamp(index);
  else
    throw GenericError("DatasetFile: access after close");
}

float DatasetFile::scale(std::size_t index) {
  if (impl_)
    return impl_->scale(index);
  else
    throw GenericError("DatasetFile: access after close");
}

void DatasetFile::write(float timeStamp, float wl, float scale, const std::complex<float>* v,
                        std::size_t ldv, const float* d, const float* uvw, std::size_t lduvw) {
  if (impl_) {
    if (impl_->is_read_only()) {
      std::string fileName = impl_->file_name();
      impl_.reset(); // close file
      // create read-write file
      impl_.reset(new DatasetFileImpl(fileName, false));
    }

    impl_->write(timeStamp, wl, scale, v, ldv, d, uvw, lduvw);
  } else {
    throw GenericError("DatasetFile: write after close");
  }
}

void DatasetFile::close() { return impl_.reset(); }

bool DatasetFile::is_open() const noexcept { return bool(impl_); }

float DatasetFile::ra_deg() const {
  if (impl_)
    return impl_->ra_deg();
  else
    throw GenericError("DatasetFile: access after close");
}

float DatasetFile::dec_deg() const {
  if (impl_)
    return impl_->dec_deg();
  else
    throw GenericError("DatasetFile: access after close");
}

}  // namespace bipp
