
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
#include "bipp/image_data_file.hpp"

namespace bipp {

namespace {

auto gatherDatasetNames(hid_t obj, const char* name, const H5O_info_t* info, void* opData) -> herr_t {
  auto& names = *reinterpret_cast<std::vector<std::string>*>(opData);

  if (info->type == H5O_TYPE_DATASET) {
    names.emplace_back(name);
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

class ImageDataFile::ImageDataFileImpl {
public:
  // create new file
  ImageDataFileImpl(const std::string& fileName, std::size_t numPixel) : numPixel_(numPixel) {
    h5File_ = h5::check(H5Fcreate(fileName.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));

    h5ImageGroup_ =
        h5::check(H5Gcreate(h5File_.id(), "images", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    h5::create_size_attr(h5File_.id(), "nPixel", numPixel);
  }

  // open file
  explicit ImageDataFileImpl(const std::string& fileName) {
    h5File_ = h5::check(H5Fopen(fileName.data(), H5F_ACC_RDWR, H5P_DEFAULT));

    h5ImageGroup_ = h5::check(H5Gopen(h5File_.id(), "images", H5P_DEFAULT));

    numPixel_ = h5::read_size_attr(h5File_.id(), "nPixel");

    // retrieve all existing images
    std::vector<std::string> tags;
#ifdef BIPP_H5OVISIT2
    h5::check(H5Ovisit2(h5ImageGroup_.id(), H5_INDEX_NAME, H5_ITER_NATIVE, gatherDatasetNames,
                        &tags, H5O_INFO_ALL));
#else
    h5::check(H5Ovisit3(h5ImageGroup_.id(), H5_INDEX_NAME, H5_ITER_NATIVE, gatherDatasetNames,
                        &tags, H5O_INFO_ALL));
#endif

    for (const auto& name : tags) {
      images_.insert({name, h5::check(H5Dopen(h5ImageGroup_.id(), name.c_str(), H5P_DEFAULT))});
    }

    for (auto& [name, dset] : images_) {
      h5::DataSpace dspace(h5::check(H5Dget_space(dset.id())));

      auto rank = H5Sget_simple_extent_ndims(dspace.id());
      if (rank != 1) {
        throw FileError("Invalid rank of dataset in image file. Expected one dimensional dataset.");
      }

      hsize_t dim, maxDim;
      h5::check(H5Sget_simple_extent_dims(dspace.id(), &dim, &maxDim));

      if (dim != numPixel_) {
        throw FileError(
            "Mismatched image sizes in  image file. Expected same size for all images.");
      }
    }
  }

  std::vector<std::string> tags() const {
    std::vector<std::string> imageTags;

    for (const auto& [t, d] : images_) {
      imageTags.emplace_back(t);
    }

    return imageTags;
  }

  std::size_t num_tags() const { return images_.size(); }

  void get(const std::string& tag, float* image) {
    auto it = images_.find(tag);
    if (it == images_.end()) {
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
    if (it == images_.end()) {
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

  std::size_t num_pixel() const { return numPixel_; }

private:
  std::size_t numPixel_;
  h5::File h5File_;
  h5::Group h5ImageGroup_;
  std::unordered_map<std::string, h5::DataSet> images_;
};

auto ImageDataFile::ImageFileImplDeleter::operator()(ImageDataFileImpl* p) -> void {
  if (p) delete p;
}

ImageDataFile::ImageDataFile(ImageDataFileImpl* ptr) : impl_(ptr) {}

ImageDataFile ImageDataFile::create(const std::string& fileName, std::size_t numPixel) {
  return ImageDataFile(new ImageDataFileImpl(fileName, numPixel));
}

ImageDataFile ImageDataFile::open(const std::string& fileName) {
  return ImageDataFile(new ImageDataFileImpl(fileName));
}

void ImageDataFile::close() { impl_.reset(); }

bool ImageDataFile::is_open() const noexcept { return bool(impl_); }

std::vector<std::string> ImageDataFile::tags() const {
  if (impl_)
    return impl_->tags();
  else
    throw GenericError("ImageDataFile: access after close");
}

std::size_t ImageDataFile::num_tags() const {
  if (impl_)
    return impl_->num_tags();
  else
    throw GenericError("ImageDataFile: access after close");
}

void ImageDataFile::get(const std::string& tag, float* image) {
  if (impl_)
    impl_->get(tag, image);
  else
    throw GenericError("ImageDataFile: access after close");
}

void ImageDataFile::set(const std::string& tag, const float* image) {
  if (impl_)
    impl_->set(tag, image);
  else
    throw GenericError("ImageDataFile: access after close");
}

std::size_t ImageDataFile::num_pixel() const {
  if (impl_)
    return impl_->num_pixel();
  else
    throw GenericError("ImageDataFile: access after close");
}

}  // namespace bipp
