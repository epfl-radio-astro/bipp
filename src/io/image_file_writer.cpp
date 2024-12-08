#include "io/image_file_writer.hpp"

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
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
class ImageFileWriter::ImageFileWriterImpl {
public:
  ImageFileWriterImpl(const std::string& fileName, const std::string& datasetFileName,
                      const std::string& datasetDescription) {
    h5File_ = h5::check(H5Fcreate(fileName.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));

    // attributes
    h5::create_string_attr(h5File_.id(), "datasetFileName", datasetFileName);
    // h5::create_string_attr(h5File_.id(), "datasetDescription", datasetDescription);
  }

  auto write(const std::string& tag, ConstHostView<float, 1> image) -> void {
    h5::DataSet dset =
        h5::create_fixed_one_dim_space(h5File_.id(), tag, h5::get_type_id<float>(), image.size());

    h5::DataSpace dspace = h5::check(H5Dget_space(dset.id()));

    h5::check(H5Dwrite(dset.id(), h5::get_type_id<float>(), dspace.id(), dspace.id(), H5P_DEFAULT,
                       image.data()));
  }

private:
  // file
  h5::File h5File_ = H5I_INVALID_HID;
};

auto ImageFileWriter::ImageFileWriterImplDeleter::operator()(ImageFileWriterImpl* p) -> void {
  if (p) delete p;
}

ImageFileWriter::ImageFileWriter(const std::string& fileName, const std::string& datasetFileName,
                                 const std::string& datasetDescription)
    : impl_(new ImageFileWriterImpl(fileName, datasetFileName, datasetDescription)) {}

auto ImageFileWriter::write(const std::string& tag, ConstHostView<float, 1> image) -> void {
  impl_->write(tag, image);
}

}  // namespace bipp
