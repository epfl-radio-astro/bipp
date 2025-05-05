
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
  ImagePropFileImpl(const std::string& fileName, std::size_t numPixel, const float* lmn,
                    std::size_t ldlmn)
      : fileName_(fileName), numPixel_(numPixel) {
    h5::File h5File =
        h5::check(H5Fcreate(fileName.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));

    h5::Group h5MetaGroup =
        h5::check(H5Gcreate(h5File.id(), "meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    h5::DataSet h5PixelL =
        h5::create_fixed_one_dim_space(h5File.id(), "pixel_l", h5::get_type_id<float>(), numPixel);
    h5::DataSet h5PixelM =
        h5::create_fixed_one_dim_space(h5File.id(), "pixel_m", h5::get_type_id<float>(), numPixel);
    h5::DataSet h5PixelN =
        h5::create_fixed_one_dim_space(h5File.id(), "pixel_n", h5::get_type_id<float>(), numPixel);

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
  explicit ImagePropFileImpl(const std::string& fileName) : fileName_(fileName), numPixel_(0) {
    h5::File h5File = h5::check(H5Fopen(fileName.data(), H5F_ACC_RDONLY, H5P_DEFAULT));

    h5::DataSet h5PixelL = h5::check(H5Dopen(h5File.id(), "pixel_l", H5P_DEFAULT));

    // retrieve the number of pixels
    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelL.id()));
      hsize_t dim = 0;
      hsize_t maxDim = 0;
      auto ndims = H5Sget_simple_extent_ndims(dspace.id());
      if (ndims != 1) {
        throw FileError("Invalid rank of dataset in image file. Expected one dimensional dataset.");
      }
      h5::check(H5Sget_simple_extent_dims(dspace.id(), &dim, &maxDim));

      numPixel_ = dim;
    }
  }

  std::size_t num_pixel() const { return numPixel_; }

  void pixel_lmn(float* lmn, std::size_t ldlmn) {
    hsize_t size = numPixel_;

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

    h5::check(H5Ovisit(h5MetaGroup.id(), H5_INDEX_NAME, H5_ITER_NATIVE, gatherMetaData, &meta,
                       H5O_INFO_ALL));

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
  std::size_t numPixel_;
};

auto ImagePropFile::ImageFileImplDeleter::operator()(ImagePropFileImpl* p) -> void {
  if (p) delete p;
}

ImagePropFile::ImagePropFile(ImagePropFileImpl* ptr) : impl_(ptr) {}

ImagePropFile ImagePropFile::create(const std::string& fileName, std::size_t numPixel,
                                    const float* lmn, std::size_t ldlmn) {
  return ImagePropFile(new ImagePropFileImpl(fileName, numPixel, lmn, ldlmn));
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

std::size_t ImagePropFile::num_pixel() const {
  if (impl_)
    return impl_->num_pixel();
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
