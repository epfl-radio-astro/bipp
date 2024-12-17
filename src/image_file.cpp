
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
#include "bipp/image_file.hpp"

namespace bipp {

namespace {

auto gatherDatasetNames(hid_t obj, const char* name, const H5O_info2_t* info, void* opData) -> herr_t {
  auto& names = *reinterpret_cast<std::vector<std::string>*>(opData);

  if (info->type == H5O_TYPE_DATASET) {
    names.emplace_back(name);
  }

  return 0;
};

auto gatherMetaData(hid_t obj, const char* name, const H5O_info2_t* info, void* opData) -> herr_t {
  auto& meta = *reinterpret_cast<std::unordered_map<std::string, ImageFile::MetaType>*>(opData);


  if (info->type == H5O_TYPE_DATASET) {
    h5::DataSet dset = h5::check(H5Dopen(obj, name, H5P_DEFAULT));
    h5::DataSpace dspace = h5::check(H5Dget_space(dset.id()));
    h5::DataType type = h5::check(H5Dget_type(dset.id()));

    auto typeClass = H5Tget_class(type.id());

    if(typeClass == H5T_INTEGER) {
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

class ImageFile::ImageFileImpl {
public:
  // create new file
  ImageFileImpl(const std::string& fileName, std::size_t numPixel, const float* lmn,
                std::size_t ldlmn)
      : numPixel_(numPixel) {
    h5File_ = h5::check(H5Fcreate(fileName.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));

    h5ImageGroup_ = h5::check(H5Gcreate(h5File_.id(), "images", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    h5MetaGroup_ = h5::check(H5Gcreate(h5File_.id(), "meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    h5PixelL_ =
        h5::create_fixed_one_dim_space(h5File_.id(), "pixel_l", h5::get_type_id<float>(), numPixel);
    h5PixelM_ =
        h5::create_fixed_one_dim_space(h5File_.id(), "pixel_m", h5::get_type_id<float>(), numPixel);
    h5PixelN_ =
        h5::create_fixed_one_dim_space(h5File_.id(), "pixel_n", h5::get_type_id<float>(), numPixel);

    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelL_.id()));
      h5::check(H5Dwrite(h5PixelL_.id(), h5::get_type_id<float>(), dspace.id(), dspace.id(),
                         H5P_DEFAULT, lmn));
    }

    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelM_.id()));
      h5::check(H5Dwrite(h5PixelM_.id(), h5::get_type_id<float>(), dspace.id(), dspace.id(),
                         H5P_DEFAULT, lmn + ldlmn));
    }

    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelN_.id()));
      h5::check(H5Dwrite(h5PixelN_.id(), h5::get_type_id<float>(), dspace.id(), dspace.id(),
                         H5P_DEFAULT, lmn + 2 * ldlmn));
    }
  }

  // open file
  explicit ImageFileImpl(const std::string& fileName) {
    h5File_ = h5::check(H5Fopen(fileName.data(), H5F_ACC_RDWR, H5P_DEFAULT));

    h5ImageGroup_ = h5::check(H5Gopen(h5File_.id(), "images", H5P_DEFAULT));
    h5MetaGroup_ = h5::check(H5Gopen(h5File_.id(), "meta", H5P_DEFAULT));

    h5PixelL_ = h5::check(H5Dopen(h5File_.id(), "pixel_l", H5P_DEFAULT));
    h5PixelM_ = h5::check(H5Dopen(h5File_.id(), "pixel_m", H5P_DEFAULT));
    h5PixelN_ = h5::check(H5Dopen(h5File_.id(), "pixel_n", H5P_DEFAULT));


    // retrieve the number of pixels
    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelL_.id()));
      hsize_t dim = 0;
      hsize_t maxDim = 0;
      auto ndims = H5Sget_simple_extent_ndims(dspace.id());
      if (ndims != 1) {
        throw FileError("Invalid rank of dataset in image file. Expected one dimensional dataset.");
      }
      h5::check(H5Sget_simple_extent_dims(dspace.id(), &dim, &maxDim));

      numPixel_ = dim;
    }

    // retrieve all existing images
    std::vector<std::string> tags;
    h5::check(H5Ovisit(h5ImageGroup_.id(), H5_INDEX_NAME, H5_ITER_NATIVE, gatherDatasetNames, &tags,
                       H5O_INFO_ALL));

    for(const auto& name : tags) {
      images_.insert({name, h5::check(H5Dopen(h5ImageGroup_.id(), name.c_str(), H5P_DEFAULT))});
    }

    for (auto& [name, dset] : images_) {
      h5::DataSpace dspace(h5::check(H5Dget_space(dset.id())));

      auto rank = H5Sget_simple_extent_ndims(dspace.id());
      if(rank != 1) {
        throw FileError("Invalid rank of dataset in image file. Expected one dimensional dataset.");
      }

      hsize_t dim, maxDim;
      h5::check(H5Sget_simple_extent_dims(dspace.id(), &dim, &maxDim));

      if (dim != numPixel_) {
        throw FileError("Mismatched image sizes in  image file. Expected same size for all images.");
      }
    }
  }

  std::vector<std::string> tags() const {
    std::vector<std::string> imageTags;

    for(const auto& [t, d]:images_){
      imageTags.emplace_back(t);
    }

    return imageTags;
  }

  std::size_t num_tags() const {
    return images_.size();
  }

  void get(const std::string& tag, float* image) {
    auto it = images_.find(tag);
    if(it == images_.end()){ 
        throw InvalidParameterError("Invalid image tag");
    }

    hsize_t size = numPixel_;

    h5::DataSpace dspace = h5::check(H5Dget_space(it->second.id()));
    h5::DataSpace mspace = h5::check(H5Screate_simple(1, &size, &size));

    h5::check(H5Dread(it->second.id(), h5::get_type_id<float>(), mspace.id(), dspace.id(),
                      H5P_DEFAULT, image));
  }

  void set(const std::string& tag, const float* image) {
    check_hdf5_name(tag);

    const auto it = images_.find(tag);
    if(it == images_.end()) {
      h5::DataSet dset = h5::create_fixed_one_dim_space(h5ImageGroup_.id(), tag,
                                                        h5::get_type_id<float>(), numPixel_);
      h5::DataSpace dspace = h5::check(H5Dget_space(dset.id()));
      h5::check(H5Dwrite(dset.id(), h5::get_type_id<float>(), dspace.id(), dspace.id(), H5P_DEFAULT,
                         image));

      images_.emplace(tag, std::move(dset));
    } else {
      auto& dset = it->second;
      h5::DataSpace dspace = h5::check(H5Dget_space(dset.id()));
      h5::check(H5Dwrite(dset.id(), h5::get_type_id<float>(), dspace.id(), dspace.id(), H5P_DEFAULT,
                         image));
    }
  }

  std::size_t num_pixel() const {
    return numPixel_;
  }

  void pixel_lmn(float* lmn, std::size_t ldlmn) {
    hsize_t size = numPixel_;

    h5::DataSpace mspace = h5::check(H5Screate_simple(1, &size, &size));
    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelL_.id()));

      h5::check(H5Dread(h5PixelL_.id(), h5::get_type_id<float>(), mspace.id(), dspace.id(),
                        H5P_DEFAULT, lmn));
    }
    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelM_.id()));

      h5::check(H5Dread(h5PixelM_.id(), h5::get_type_id<float>(), mspace.id(), dspace.id(),
                        H5P_DEFAULT, lmn + ldlmn));
    }
    {
      h5::DataSpace dspace = h5::check(H5Dget_space(h5PixelN_.id()));

      h5::check(H5Dread(h5PixelN_.id(), h5::get_type_id<float>(), mspace.id(), dspace.id(),
                        H5P_DEFAULT, lmn + 2*ldlmn));
    }
  }

  std::unordered_map<std::string, MetaType> meta_data() const {
    std::unordered_map<std::string, MetaType> meta;

    h5::check(H5Ovisit(h5MetaGroup_.id(), H5_INDEX_NAME, H5_ITER_NATIVE, gatherMetaData, &meta,
                       H5O_INFO_ALL));

    return meta;
  }

  void set_meta(const std::string& name, const MetaType& data) { check_hdf5_name(name);
    if (H5Lexists(h5MetaGroup_.id(), name.c_str(), H5P_DEFAULT) > 0) {
      throw InvalidParameterError("Meta data name already exists");
    }

    std::visit(
        [&](auto&& arg) -> void {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, std::size_t> || std::is_same_v<ArgType, float>) {
            ArgType value = arg;
            h5::DataSet dset = h5::create_fixed_one_dim_space(
                h5MetaGroup_.id(), name.c_str(), h5::get_type_id<decltype(value)>(), 1);
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
            h5::DataSet dset = h5::create_fixed_one_dim_space(
                h5MetaGroup_.id(), name.c_str(), type.id(), 1);
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
  std::size_t numPixel_;

  // file
  h5::File h5File_;

  // Groups
  h5::Group h5ImageGroup_;
  h5::Group h5MetaGroup_;

  // datasets
  h5::DataSet h5PixelL_;
  h5::DataSet h5PixelM_;
  h5::DataSet h5PixelN_;
  std::unordered_map<std::string, h5::DataSet> images_;
};

auto ImageFile::ImageFileImplDeleter::operator()(ImageFileImpl* p) -> void {
  if (p) delete p;
}

ImageFile::ImageFile(ImageFileImpl* ptr) : impl_(ptr) {}

ImageFile ImageFile::create(const std::string& fileName, std::size_t numPixel, const float* lmn,
                            std::size_t ldlmn) {
  return ImageFile(new ImageFileImpl(fileName, numPixel, lmn, ldlmn));
}

ImageFile ImageFile::open(const std::string& fileName) {
  return ImageFile(new ImageFileImpl(fileName));
}

std::unordered_map<std::string, ImageFile::MetaType> ImageFile::meta_data() const {
  if (impl_)
    return impl_->meta_data();
  else
    throw GenericError("ImageFile: access after close");
}

void ImageFile::set_meta(const std::string& name, const MetaType& data) {
  if (impl_)
    impl_->set_meta(name, data);
  else
    throw GenericError("ImageFile: access after close");
}

void ImageFile::close() {
  impl_.reset();
}

bool ImageFile::is_open() const noexcept {
  return bool(impl_);
}

std::vector<std::string> ImageFile::tags() const {
  if (impl_)
    return impl_->tags();
  else
    throw GenericError("ImageFile: access after close");
}

std::size_t ImageFile::num_tags() const  {
  if (impl_)
    return impl_->num_tags();
  else
    throw GenericError("ImageFile: access after close");
}

void ImageFile::get(const std::string& tag, float* image)  {
  if (impl_)
    impl_->get(tag, image);
  else
    throw GenericError("ImageFile: access after close");
}

void ImageFile::set(const std::string& tag, const float* image)  {
  if (impl_)
    impl_->set(tag, image);
  else
    throw GenericError("ImageFile: access after close");
}

std::size_t ImageFile::num_pixel() const  {
  if (impl_)
    return impl_->num_pixel();
  else
    throw GenericError("ImageFile: access after close");
}

void ImageFile::pixel_lmn(float* lmn, std::size_t ldlmn)  {
  if (impl_)
    impl_->pixel_lmn(lmn, ldlmn);
  else
    throw GenericError("ImageFile: access after close");
}

}  // namespace bipp
