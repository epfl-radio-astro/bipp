
#include <hdf5.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "io/h5_util.hpp"
#include "bipp/image_prop_file.hpp"

namespace bipp {

namespace {

auto gatherMetaData(hid_t obj, const char* name, const H5O_info_t* info, void* opData) -> herr_t {
  auto& meta = *reinterpret_cast<std::unordered_map<std::string, ImagePropFile::MetaType>*>(opData);

  if (info->type == H5O_TYPE_DATASET) {
    h5::DataSet dset = h5::check(H5Dopen(obj, name, H5P_DEFAULT));
    h5::DataSpace dspace = h5::check(H5Dget_space(dset.id()));
    h5::DataType type = h5::check(H5Dget_type(dset.id()));

    auto typeClass = H5Tget_class(type.id());

    if (typeClass == H5T_INTEGER) {
      hsize_t size = 1;
      h5::DataSpace mspace = h5::check(H5Screate_simple(1, &size, &size));
      unsigned long long value = 0;
      h5::check(H5Dread(dset.id(), h5::get_type_id<decltype(value)>(), mspace.id(), dspace.id(),
                        H5P_DEFAULT, &value));

      meta.emplace(name, std::size_t(value));
    } else if (typeClass == H5T_FLOAT) {
      hsize_t size = 1;
      h5::DataSpace mspace = h5::check(H5Screate_simple(1, &size, &size));
      float value = 0;
      h5::check(H5Dread(dset.id(), h5::get_type_id<decltype(value)>(), mspace.id(), dspace.id(),
                        H5P_DEFAULT, &value));

      meta.emplace(name, value);
    } else if (typeClass == H5T_ARRAY) {
      const auto ndims = H5Tget_array_ndims(type.id());
      if (ndims == 1) {
        hsize_t numValues = 0;
        h5::check(H5Tget_array_dims(type.id(), &numValues));
        std::vector<float> values(numValues);

        std::array<hsize_t, 1> dims = {numValues};
        h5::DataType arrayType =
            h5::check(H5Tarray_create(h5::get_type_id<float>(), dims.size(), dims.data()));

        hsize_t size = 1;
        h5::DataSpace mspace = h5::check(H5Screate_simple(1, &size, &size));
        h5::check(H5Dread(dset.id(), arrayType.id(), mspace.id(), dspace.id(), H5P_DEFAULT,
                          values.data()));
        meta.emplace(name, std::move(values));
      }
    }
  }

  return 0;
};

auto check_hdf5_name(const std::string& name) {
  if (name.find('/') != std::string::npos) {
    throw InvalidParameterError("HDF5 name must not contain '/'");
  }
  if (name.find('.') != std::string::npos) {
    throw InvalidParameterError("HDF5 name must not contain '.'");
  }
  if (name.find(' ') != std::string::npos) {
    throw InvalidParameterError("HDF5 name must not contain spaces");
  }
}

}  // namespace

class ImagePropFile::ImagePropFileImpl {
public:
  // create new file
  ImagePropFileImpl(const std::string& fileName, std::size_t height, std::size_t width, float fovDeg,
                    const float* lmn, std::size_t ldlmn)
      : fileName_(fileName), height_(height), width_(width), fovDeg_(fovDeg) {
    h5::File h5File =
        h5::check(H5Fcreate(fileName.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));

    h5::create_size_attr(h5File.id(), "width", width);
    h5::create_size_attr(h5File.id(), "height", height);
    h5::create_float_attr(h5File.id(), "fovDeg", fovDeg);

    h5::Group h5MetaGroup =
        h5::check(H5Gcreate(h5File.id(), "meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    h5::DataSet h5PixelL =
        h5::create_fixed_one_dim_space(h5File.id(), "pixel_l", h5::get_type_id<float>(), width_ * height_);
    h5::DataSet h5PixelM =
        h5::create_fixed_one_dim_space(h5File.id(), "pixel_m", h5::get_type_id<float>(), width_ * height_);
    h5::DataSet h5PixelN =
        h5::create_fixed_one_dim_space(h5File.id(), "pixel_n", h5::get_type_id<float>(), width_ * height_);

    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelL.id()));
      h5::check(H5Dwrite(h5PixelL.id(), h5::get_type_id<float>(), dspace.id(), dspace.id(),
                         H5P_DEFAULT, lmn));
    }

    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelM.id()));
      h5::check(H5Dwrite(h5PixelM.id(), h5::get_type_id<float>(), dspace.id(), dspace.id(),
                         H5P_DEFAULT, lmn + ldlmn));
    }

    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelN.id()));
      h5::check(H5Dwrite(h5PixelN.id(), h5::get_type_id<float>(), dspace.id(), dspace.id(),
                         H5P_DEFAULT, lmn + 2 * ldlmn));
    }
  }

  // open file
  explicit ImagePropFileImpl(const std::string& fileName) : fileName_(fileName), height_(0), width_(0) {
    h5::File h5File = h5::check(H5Fopen(fileName.data(), H5F_ACC_RDONLY, H5P_DEFAULT));


    width_ = h5::read_size_attr(h5File.id(), "width");
    height_ = h5::read_size_attr(h5File.id(), "height");
    fovDeg_ = h5::read_float_attr(h5File.id(), "fovDeg");

  }

  std::size_t width() const {return width_;}

  std::size_t height() const {return height_;}

  float fov_deg() const {return fovDeg_;}

  void pixel_lmn(float* lmn, std::size_t ldlmn) {
    hsize_t size = width_ * height_;

    h5::File h5File = h5::check(H5Fopen(fileName_.data(), H5F_ACC_RDONLY, H5P_DEFAULT));
    h5::DataSet h5PixelL = h5::check(H5Dopen(h5File.id(), "pixel_l", H5P_DEFAULT));
    h5::DataSet h5PixelM = h5::check(H5Dopen(h5File.id(), "pixel_m", H5P_DEFAULT));
    h5::DataSet h5PixelN = h5::check(H5Dopen(h5File.id(), "pixel_n", H5P_DEFAULT));

    h5::DataSpace mspace = h5::check(H5Screate_simple(1, &size, &size));
    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelL.id()));

      h5::check(H5Dread(h5PixelL.id(), h5::get_type_id<float>(), mspace.id(), dspace.id(),
                        H5P_DEFAULT, lmn));
    }
    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelM.id()));

      h5::check(H5Dread(h5PixelM.id(), h5::get_type_id<float>(), mspace.id(), dspace.id(),
                        H5P_DEFAULT, lmn + ldlmn));
    }
    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelN.id()));

      h5::check(H5Dread(h5PixelN.id(), h5::get_type_id<float>(), mspace.id(), dspace.id(),
                        H5P_DEFAULT, lmn + 2 * ldlmn));
    }
  }

  std::unordered_map<std::string, MetaType> meta_data() const {
    std::unordered_map<std::string, MetaType> meta;

    h5::File h5File = h5::check(H5Fopen(fileName_.data(), H5F_ACC_RDONLY, H5P_DEFAULT));
    h5::Group h5MetaGroup = h5::check(H5Gopen(h5File.id(), "meta", H5P_DEFAULT));

// avoid using H5Ovisit1, because it requires different arguments
#if (H5_VERS_MAJOR >= 1) && (H5_VERS_MINOR >= 12)
    h5::check(H5Ovisit3(h5MetaGroup.id(), H5_INDEX_NAME, H5_ITER_NATIVE, gatherMetaData, &meta,
                       H5O_INFO_ALL));
#else
    h5::check(H5Ovisit2(h5MetaGroup.id(), H5_INDEX_NAME, H5_ITER_NATIVE, gatherMetaData, &meta,
                       H5O_INFO_ALL));
#endif

    return meta;
  }

  void set_meta(const std::string& name, const MetaType& data) {
    check_hdf5_name(name);

    h5::File h5File = h5::check(H5Fopen(fileName_.data(), H5F_ACC_RDWR, H5P_DEFAULT));
    h5::Group h5MetaGroup = h5::check(H5Gopen(h5File.id(), "meta", H5P_DEFAULT));

    if (H5Lexists(h5MetaGroup.id(), name.c_str(), H5P_DEFAULT) > 0) {
      throw InvalidParameterError("Meta data name already exists");
    }

    std::visit(
        [&](auto&& arg) -> void {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, std::size_t> || std::is_same_v<ArgType, float>) {
            ArgType value = arg;
            h5::DataSet dset = h5::create_fixed_one_dim_space(
                h5MetaGroup.id(), name.c_str(), h5::get_type_id<decltype(value)>(), 1);
            h5::DataSpace dspace = h5::check(H5Dget_space(dset.id()));
            h5::check(H5Dwrite(dset.id(), h5::get_type_id<decltype(value)>(), dspace.id(),
                               dspace.id(), H5P_DEFAULT, &value));

          } else if constexpr (std::is_same_v<ArgType, std::vector<float>>) {
            if (arg.empty()) {
              throw InvalidParameterError("Empty meta data");
            }
            std::array<hsize_t, 1> dims = {arg.size()};
            h5::DataType type =
                h5::check(H5Tarray_create(h5::get_type_id<float>(), dims.size(), dims.data()));
            h5::DataSet dset =
                h5::create_fixed_one_dim_space(h5MetaGroup.id(), name.c_str(), type.id(), 1);
            h5::DataSpace dspace = h5::check(H5Dget_space(dset.id()));
            h5::check(
                H5Dwrite(dset.id(), type.id(), dspace.id(), dspace.id(), H5P_DEFAULT, arg.data()));

          } else {
            throw InternalError("Unknown meta data type");
          }
        },
        data);
  }

private:
  std::string fileName_;
  std::size_t height_, width_;
  float fovDeg_;
};

auto ImagePropFile::ImageFileImplDeleter::operator()(ImagePropFileImpl* p) -> void {
  if (p) delete p;
}

ImagePropFile::ImagePropFile(ImagePropFileImpl* ptr) : impl_(ptr) {}

ImagePropFile ImagePropFile::create(const std::string& fileName, std::size_t height, std::size_t width, float fovDeg,
                    const float* lmn, std::size_t ldlmn) {
  return ImagePropFile(new ImagePropFileImpl(fileName, height, width, fovDeg, lmn, ldlmn));
}

ImagePropFile ImagePropFile::open(const std::string& fileName) {
  return ImagePropFile(new ImagePropFileImpl(fileName));
}

std::unordered_map<std::string, ImagePropFile::MetaType> ImagePropFile::meta_data() const {
  if (impl_)
    return impl_->meta_data();
  else
    throw GenericError("ImagePropFile: access after close");
}

void ImagePropFile::set_meta(const std::string& name, const MetaType& data) {
  if (impl_)
    impl_->set_meta(name, data);
  else
    throw GenericError("ImagePropFile: access after close");
}

void ImagePropFile::close() { impl_.reset(); }

bool ImagePropFile::is_open() const noexcept { return bool(impl_); }

std::size_t ImagePropFile::width() const {
  if (impl_)
    return impl_->width();
  else
    throw GenericError("ImagePropFile: access after close");
}

std::size_t ImagePropFile::height() const {
  if (impl_)
    return impl_->height();
  else
    throw GenericError("ImagePropFile: access after close");
}

float ImagePropFile::fov_deg() const {
  if (impl_)
    return impl_->fov_deg();
  else
    throw GenericError("ImagePropFile: access after close");
}

void ImagePropFile::pixel_lmn(float* lmn, std::size_t ldlmn) {
  if (impl_)
    impl_->pixel_lmn(lmn, ldlmn);
  else
    throw GenericError("ImagePropFile: access after close");
}

}  // namespace bipp
