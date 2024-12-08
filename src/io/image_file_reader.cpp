#include "io/image_file_reader.hpp"

#include <hdf5.h>

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "io/h5_util.hpp"
#include "memory/view.hpp"

namespace bipp {

namespace {

auto gatherDatasetNames(hid_t obj, const char* name, const H5O_info2_t* info, void* opData) -> herr_t {
  auto& names = *reinterpret_cast<std::vector<std::string>*>(opData);

  if (info->type == H5O_TYPE_DATASET) {
    names.emplace_back(name);
  }

  return 0;
};

}  // namespace

class ImageFileReader::ImageFileReaderImpl {
public:
  explicit ImageFileReaderImpl(const std::string& fileName) {
    h5File_ = h5::check(H5Fopen(fileName.data(), H5F_ACC_RDONLY, H5P_DEFAULT));

    datasetFileName_ = h5::read_string_attr(h5File_.id(), "datasetFileName");
    // datasetDescription_ = h5::read_string_attr(h5File_.id(), "datasetDescription");

    h5::check(H5Ovisit(h5File_.id(), H5_INDEX_NAME, H5_ITER_NATIVE, gatherDatasetNames, &tags_,
                       H5O_INFO_ALL));

    for(const auto& name : tags_) {
      datasets_.insert({name, h5::check(H5Dopen(h5File_.id(), name.c_str(), H5P_DEFAULT))});
    }

    for(auto& [name, dset]: datasets_) {
      h5::DataSpace dspace(h5::check(H5Dget_space(dset.id())));

      auto rank = H5Sget_simple_extent_ndims(dspace.id());
      if(rank != 1) {
        throw FileError("Invalid rank of dataset in image file. Expected one dimensional dataset.");
      }

      hsize_t dim, maxDim;
      h5::check(H5Sget_simple_extent_dims(dspace.id(), &dim, &maxDim));

      if (!numPixel_) {
        numPixel_ = dim;
      }

      if(dim != numPixel_) {
        throw FileError("Mismatched image sizes in  image file. Expected same size for all images.");
      }
    }
  }

  auto read(const std::string& tag, HostView<float, 1> image) -> void {
    auto it = datasets_.find(tag);
    if(it == datasets_.end()){ 
        throw InvalidParameterError("Invalid image tag");
    }

    hsize_t size = image.size();

    h5::DataSpace dspace = h5::check(H5Dget_space(it->second.id()));
    h5::DataSpace mspace = h5::check(H5Screate_simple(1, &size, &size));

    h5::check(H5Dread(it->second.id(), h5::get_type_id<float>(), mspace.id(), dspace.id(), H5P_DEFAULT,
                  image.data()));
  }

  auto num_pixel() const -> std::size_t{
    return numPixel_;
  }

  auto tags() const -> const std::vector<std::string>& {
    return tags_;
  }

  auto dataset_file_name() const -> const std::string&{
    return datasetFileName_;
  }

  auto dataset_description() const -> const std::string&{
    return datasetDescription_;
  }

private:
  // file
  h5::File h5File_ = H5I_INVALID_HID;
  std::unordered_map<std::string, h5::DataSet> datasets_;
  std::vector<std::string> tags_;
  std::string datasetFileName_, datasetDescription_;
  std::size_t numPixel_ = 0;
};

auto ImageFileReader::ImageFileReaderImplDeleter::operator()(ImageFileReaderImpl* p) -> void {
  if (p) delete p;
}

ImageFileReader::ImageFileReader(const std::string& fileName)
    : impl_(new ImageFileReaderImpl(fileName)) {}

auto ImageFileReader::read(const std::string& tag, HostView<float, 1> image) -> void {
  impl_->read(tag, image);
}

auto ImageFileReader::dataset_file_name() const -> const std::string& {
  return impl_->dataset_file_name();
}

auto ImageFileReader::dataset_description() const -> const std::string& {
  return impl_->dataset_description();
}

auto ImageFileReader::num_pixel() const -> std::size_t { return impl_->num_pixel(); }

auto ImageFileReader::tags() const -> const std::vector<std::string>& { return impl_->tags(); }

}  // namespace bipp
