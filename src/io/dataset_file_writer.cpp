#include "io/dataset_file_writer.hpp"

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
class DatasetFileWriter::DatasetFileWriterImpl {
public:
  DatasetFileWriterImpl(const std::string& fileName, const std::string& description,
                        std::size_t nAntenna, std::size_t nBeam)
      : nAntenna_(nAntenna), nBeam_(nBeam) {
    h5File_ = h5::check(H5Fcreate(fileName.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));

    // attributes
    h5::create_size_attr(h5File_.id(), "formatVersionMajor", datasetFormatVersionMajor);
    h5::create_size_attr(h5File_.id(), "formatVersionMinor", datasetFormatVersionMinor);
    h5::create_size_attr(h5File_.id(), "nBeam", nBeam);
    h5::create_size_attr(h5File_.id(), "nAntenna", nAntenna);
    h5::create_string_attr(h5File_.id(), "description",
                           description.size() ? description : fileName);

    // create array types
    {
      std::array<hsize_t, 1> dims = {nBeam};
      h5EigValType_ =
          h5::check(H5Tarray_create(h5::get_type_id<ValueType>(), dims.size(), dims.data()));
    }
    {
      std::array<hsize_t, 2> dims = {nBeam, 2 * nAntenna};
      h5EigVecType_ =
          h5::check(H5Tarray_create(h5::get_type_id<ValueType>(), dims.size(), dims.data()));
    }
    {
      std::array<hsize_t, 2> dims = {3, nAntenna * nAntenna};
      h5UVWType_ =
          h5::check(H5Tarray_create(h5::get_type_id<ValueType>(), dims.size(), dims.data()));
    }

    {
      std::array<hsize_t, 2> dims = {3, nAntenna};
      h5XYZType_ =
          h5::check(H5Tarray_create(h5::get_type_id<ValueType>(), dims.size(), dims.data()));
    }

    // TODO: set optimal chunk size individually
    constexpr hsize_t CHUNK_SIZE = 5;

    // create data spaces
    h5Wl_ = h5::create_one_dim_space(h5File_.id(), "wl", h5::get_type_id<ValueType>(), CHUNK_SIZE);
    h5NVis_ =
        h5::create_one_dim_space(h5File_.id(), "nvis", h5::get_type_id<unsigned int>(), CHUNK_SIZE);
    h5MinU_ =
        h5::create_one_dim_space(h5File_.id(), "uMin", h5::get_type_id<ValueType>(), CHUNK_SIZE);
    h5MinV_ =
        h5::create_one_dim_space(h5File_.id(), "vMin", h5::get_type_id<ValueType>(), CHUNK_SIZE);
    h5MinW_ =
        h5::create_one_dim_space(h5File_.id(), "wMin", h5::get_type_id<ValueType>(), CHUNK_SIZE);
    h5MaxU_ =
        h5::create_one_dim_space(h5File_.id(), "uMax", h5::get_type_id<ValueType>(), CHUNK_SIZE);
    h5MaxV_ =
        h5::create_one_dim_space(h5File_.id(), "vMax", h5::get_type_id<ValueType>(), CHUNK_SIZE);
    h5MaxW_ =
        h5::create_one_dim_space(h5File_.id(), "wMax", h5::get_type_id<ValueType>(), CHUNK_SIZE);

    h5EigVal_ = h5::create_one_dim_space(h5File_.id(), "eigVal", h5EigValType_.id(), CHUNK_SIZE);
    h5EigVec_ = h5::create_one_dim_space(h5File_.id(), "eigVec", h5EigVecType_.id(), CHUNK_SIZE);
    h5UVW_ = h5::create_one_dim_space(h5File_.id(), "uvw", h5UVWType_.id(), CHUNK_SIZE);
    h5XYZ_ = h5::create_one_dim_space(h5File_.id(), "xyz", h5XYZType_.id(), CHUNK_SIZE);
  }

  auto write(ValueType wl, std::size_t nVis, ConstHostView<std::complex<ValueType>, 2> v,
             ConstHostView<ValueType, 1> d, ConstHostView<ValueType, 2> uvw,
             ConstHostView<ValueType, 2> xyz) -> void {
    if (!v.is_contiguous()) {
      throw InternalError("DatasetFileWriter: eigenvector view must be contiguous.");
    }
    if (!uvw.is_contiguous()) {
      throw InternalError("DatasetFileWriter: uvw view must be contiguous.");
    }
    if (!xyz.is_contiguous()) {
      throw InternalError("DatasetFileWriter: xyz view must be contiguous.");
    }

    const auto index = count_;
    const auto newSize = ++count_;

    h5::check(H5Dextend(h5EigVal_.id(), &newSize));
    h5::check(H5Dextend(h5EigVec_.id(), &newSize));
    h5::check(H5Dextend(h5UVW_.id(), &newSize));
    h5::check(H5Dextend(h5XYZ_.id(), &newSize));
    h5::check(H5Dextend(h5Wl_.id(), &newSize));
    h5::check(H5Dextend(h5NVis_.id(), &newSize));
    h5::check(H5Dextend(h5MinU_.id(), &newSize));
    h5::check(H5Dextend(h5MinV_.id(), &newSize));
    h5::check(H5Dextend(h5MinW_.id(), &newSize));
    h5::check(H5Dextend(h5MaxU_.id(), &newSize));
    h5::check(H5Dextend(h5MaxV_.id(), &newSize));
    h5::check(H5Dextend(h5MaxW_.id(), &newSize));

    h5::write_single_element(index, h5EigVal_.id(), h5EigValType_.id(), d.data());
    h5::write_single_element(index, h5EigVec_.id(), h5EigVecType_.id(), v.data());
    h5::write_single_element(index, h5UVW_.id(), h5UVWType_.id(), uvw.data());
    h5::write_single_element(index, h5XYZ_.id(), h5XYZType_.id(), xyz.data());

    h5::write_single_element(index, h5Wl_.id(), h5::get_type_id<ValueType>(), &wl);
    unsigned int nVisOut = static_cast<unsigned int>(nVis);
    h5::write_single_element(index, h5NVis_.id(), h5::get_type_id<decltype(nVisOut)>(), &nVisOut);

    const auto uMM = std::minmax_element(&uvw[{0, 0}], &uvw[{0, 0}] + uvw.shape(0));
    const auto vMM = std::minmax_element(&uvw[{0, 1}], &uvw[{0, 1}] + uvw.shape(0));
    const auto wMM = std::minmax_element(&uvw[{0, 2}], &uvw[{0, 2}] + uvw.shape(0));

    h5::write_single_element(index, h5MinU_.id(), h5::get_type_id<ValueType>(), uMM.first);
    h5::write_single_element(index, h5MaxU_.id(), h5::get_type_id<ValueType>(), uMM.second);

    h5::write_single_element(index, h5MinV_.id(), h5::get_type_id<ValueType>(), vMM.first);
    h5::write_single_element(index, h5MaxV_.id(), h5::get_type_id<ValueType>(), vMM.second);

    h5::write_single_element(index, h5MinW_.id(), h5::get_type_id<ValueType>(), wMM.first);
    h5::write_single_element(index, h5MaxW_.id(), h5::get_type_id<ValueType>(), wMM.second);
  }

private:
  hsize_t nAntenna_, nBeam_;

  // counter. Equal to outer dimension size
  hsize_t count_ = 0;

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

auto DatasetFileWriter::DatasetFileWriterImplDeleter::operator()(DatasetFileWriterImpl* p) -> void {
  if (p) delete p;
}

DatasetFileWriter::DatasetFileWriter(const std::string& fileName, const std::string& description,
                                     std::size_t nAntenna, std::size_t nBeam)
    : impl_(new DatasetFileWriterImpl(fileName, description, nAntenna, nBeam)) {}

auto DatasetFileWriter::write(ValueType wl, std::size_t nVis,
                              ConstHostView<std::complex<ValueType>, 2> v,
                              ConstHostView<ValueType, 1> d, ConstHostView<ValueType, 2> uvw,
                              ConstHostView<ValueType, 2> xyz) -> void {
  impl_->write(wl, nVis, v, d, uvw, xyz);
}

}  // namespace bipp
