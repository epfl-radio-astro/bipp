#include "io/dataset_writer.hpp"

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
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {

namespace {


class DataSet {
  public:
  DataSet(hid_t identifier) noexcept : id_(identifier) {}

  DataSet() noexcept : id_(H5I_INVALID_HID) {}

  DataSet(const DataSet&) = delete;
  DataSet(DataSet&& s) { *this = std::move(s); }

  auto operator=(const DataSet&) -> DataSet& = delete;
  auto operator=(DataSet&& s) noexcept -> DataSet& {
    if(id_ != H5I_INVALID_HID) {
        H5Dclose(id_);
    }
    id_ = s.id_;
    s.id_ = H5I_INVALID_HID;
    return *this;
  }

  ~DataSet() noexcept {
    if(id_ != H5I_INVALID_HID) {
        H5Dclose(id_);
    }
  }

  inline auto id() noexcept -> hid_t {
    return id_;
  }

private:
  hid_t id_;
};

class DataSpace {
  public:
  DataSpace(hid_t identifier) noexcept : id_(identifier) {}

  DataSpace() noexcept : id_(H5I_INVALID_HID) {}

  DataSpace(const DataSpace&) = delete;
  DataSpace(DataSpace&& s) { *this = std::move(s); }

  auto operator=(const DataSpace&) -> DataSpace& = delete;
  auto operator=(DataSpace&& s) noexcept -> DataSpace& {
    if(id_ != H5I_INVALID_HID) {
        H5Dclose(id_);
    }
    id_ = s.id_;
    s.id_ = H5I_INVALID_HID;
    return *this;
  }

  ~DataSpace() noexcept {
    if(id_ != H5I_INVALID_HID) {
        H5Sclose(id_);
    }
  }

  inline auto id() noexcept -> hid_t {
    return id_;
  }

private:
  hid_t id_;
};


template<typename T>
inline auto get_type_id() -> hid_t {
  if constexpr (std::is_same_v<T, float>) {
    return H5T_NATIVE_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return H5T_NATIVE_DOUBLE;
  } else if constexpr (std::is_same_v<T, unsigned long long>) {
    return H5T_NATIVE_ULLONG;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return H5T_NATIVE_UINT;
  } else {
    throw InternalError("unkown hdf5 type");
  }
  return H5I_INVALID_HID;
}

// inline auto destroy_hdf5(hid_t h) noexcept -> void {
//   if (h != H5I_INVALID_HID) {
//     H5O_info2_t info;
//     if (H5Oget_info(h, &info, H5O_INFO_BASIC) < 0) return;
//     switch (info.type) {
//       case H5O_TYPE_GROUP:
//         printf("H5O_TYPE_GROUP\n");
//         break;
//       case H5O_TYPE_DATASET:
//         printf("H5O_TYPE_DATASET\n");
//         break;
//       case H5O_TYPE_NAMED_DATATYPE:
//         printf("H5O_TYPE_NAMED_DATATYPE\n");
//         break;
//       case H5O_TYPE_MAP:
//         printf("H5O_TYPE_MAP\n");
//         break;
//       case H5O_TYPE_NTYPES:
//         printf("H5O_TYPE_NTYPES\n");
//         break;
//       case H5O_TYPE_UNKNOWN:
//         printf("H5O_TYPE_UNKNOWN\n");
//         break;
//     }
//   }
// }

inline auto check_hdf5(hid_t h) ->hid_t  {
  if(h == H5I_INVALID_HID) {
    throw HDF5Error();
  }
  return h;
}

inline auto check_hdf5(int h) -> void {
  if (h < 0) {
    throw HDF5Error();
  }
}

auto create_string_attr(hid_t hid, const std::string& name, const std::string& value) -> void {
  auto attSpace = check_hdf5(H5Screate(H5S_SCALAR));

  auto attType = check_hdf5(H5Tcopy(H5T_C_S1));

  check_hdf5(H5Tset_size(attType, value.size()));

  auto attId = check_hdf5(H5Acreate(hid, name.data(), attType, attSpace, H5P_DEFAULT, H5P_DEFAULT));

  check_hdf5(H5Awrite(attId, attType, value.data()));

  check_hdf5(H5Sclose(attSpace));
  check_hdf5(H5Tclose(attType));
  check_hdf5(H5Aclose(attId));
}

auto create_size_attr(hid_t hid, const std::string& name, unsigned int size) -> void {
  auto attSpace = check_hdf5(H5Screate(H5S_SCALAR));

  auto attId = check_hdf5(
      H5Acreate(hid, name.data(), get_type_id<decltype(size)>(), attSpace, H5P_DEFAULT, H5P_DEFAULT));

  check_hdf5(H5Awrite(attId, get_type_id<decltype(size)>(), &size));

  check_hdf5(H5Sclose(attSpace));
  check_hdf5(H5Aclose(attId));
}

auto create_one_dim_space(hid_t fd, const std::string& name, hid_t type,
                          hsize_t chunkSize) -> hid_t {
  std::array<hsize_t, 1> chunks = {chunkSize};
  std::array<hsize_t, 1> dims = {0};
  std::array<hsize_t, 1> maxDims = {H5S_UNLIMITED};

  auto dataspace = check_hdf5(H5Screate_simple(1, dims.data(), maxDims.data()));
  auto cparms = check_hdf5(H5Pcreate(H5P_DATASET_CREATE));
  check_hdf5(H5Pset_chunk(cparms, 1, chunks.data()));

  auto arr =
      check_hdf5(H5Dcreate(fd, name.data(), type, dataspace, H5P_DEFAULT, cparms, H5P_DEFAULT));

  H5Sclose(dataspace);
  H5Pclose(cparms);

  return arr;
}

auto write_single_element(hsize_t index, hid_t dset, hid_t type, const void* data) {
  const hsize_t one = 1;
  DataSpace dspace(check_hdf5(H5Dget_space(dset)));
  check_hdf5(H5Sselect_elements(dspace.id(), H5S_SELECT_SET, 1, &index));

  DataSpace mspace = check_hdf5(H5Screate_simple(1, &one, &one));

  check_hdf5(
      H5Dwrite(dset, type, mspace.id(), dspace.id(), H5P_DEFAULT, data));
}

}

class DatasetWriter::DatasetWriterImpl {
public:
  DatasetWriterImpl(const std::string& fileName, const std::string& datasetName,
                    std::size_t nAntenna, std::size_t nBeam)
      : nAntenna_(nAntenna), nBeam_(nBeam) {
    h5File_ = check_hdf5(H5Fcreate(fileName.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));

    // attributes
    create_size_attr(h5File_, "formatVersion", datasetFormatVersion);
    create_size_attr(h5File_, "nBeam", nBeam);
    create_size_attr(h5File_, "nAntenna", nAntenna);
    create_string_attr(h5File_, "name", datasetName);


    // create array types
    {
      std::array<hsize_t, 1> dims = {nBeam};
      h5EigValType_ = check_hdf5(H5Tarray_create(get_type_id<float>(), dims.size(), dims.data()));
    }
    {
      std::array<hsize_t, 2> dims = {nBeam, 2 * nAntenna};
      h5EigVecType_ = check_hdf5(H5Tarray_create(get_type_id<float>(), dims.size(), dims.data()));
    }
    {
      std::array<hsize_t, 2> dims = {3, nAntenna * nAntenna};
      h5UVWType_ = check_hdf5(H5Tarray_create(get_type_id<float>(), dims.size(), dims.data()));
    }

    // TODO: set optimal chunk size individually
    constexpr hsize_t CHUNK_SIZE = 5;

    // create data spaces
    h5Wl_ = create_one_dim_space(h5File_, "wl", get_type_id<float>(), CHUNK_SIZE);
    h5NVis_ = create_one_dim_space(h5File_, "nvis", get_type_id<unsigned int>(), CHUNK_SIZE);
    h5MinU_ = create_one_dim_space(h5File_, "uMin", get_type_id<float>(), CHUNK_SIZE);
    h5MinV_ = create_one_dim_space(h5File_, "vMin", get_type_id<float>(), CHUNK_SIZE);
    h5MinW_ = create_one_dim_space(h5File_, "wMin", get_type_id<float>(), CHUNK_SIZE);
    h5MaxU_ = create_one_dim_space(h5File_, "uMax", get_type_id<float>(), CHUNK_SIZE);
    h5MaxV_ = create_one_dim_space(h5File_, "vMax", get_type_id<float>(), CHUNK_SIZE);
    h5MaxW_ = create_one_dim_space(h5File_, "wMax", get_type_id<float>(), CHUNK_SIZE);

    h5EigVal_ = create_one_dim_space(h5File_, "eigVal", h5EigValType_, CHUNK_SIZE);
    h5EigVec_ = create_one_dim_space(h5File_, "eigVec", h5EigVecType_, CHUNK_SIZE);
    h5UVW_ = create_one_dim_space(h5File_, "uvw", h5UVWType_, CHUNK_SIZE);
  }

  ~DatasetWriterImpl() {
    if (h5File_ != H5I_INVALID_HID) {
      H5Fclose(h5File_);
    }
  }

  auto write(ValueType wl, std::size_t nVis, ConstHostView<std::complex<ValueType>, 2> v,
             ConstHostView<ValueType, 1> d, ConstHostView<ValueType, 2> uvw) -> void {
    const auto index = count_;
    const auto newSize = ++count_;

    check_hdf5(H5Dextend(h5EigVal_.id(), &newSize));
    check_hdf5(H5Dextend(h5EigVec_.id(), &newSize));
    check_hdf5(H5Dextend(h5UVW_.id(), &newSize));
    check_hdf5(H5Dextend(h5Wl_.id(), &newSize));
    check_hdf5(H5Dextend(h5NVis_.id(), &newSize));
    check_hdf5(H5Dextend(h5MinU_.id(), &newSize));
    check_hdf5(H5Dextend(h5MinV_.id(), &newSize));
    check_hdf5(H5Dextend(h5MinW_.id(), &newSize));
    check_hdf5(H5Dextend(h5MaxU_.id(), &newSize));
    check_hdf5(H5Dextend(h5MaxV_.id(), &newSize));
    check_hdf5(H5Dextend(h5MaxW_.id(), &newSize));

    write_single_element(index, h5EigVal_.id(), h5EigValType_, d.data());
    if (v.is_contiguous()) {
      write_single_element(index, h5EigVec_.id(), h5EigVecType_, v.data());
    } else {
      // TODO
    }
    if (uvw.is_contiguous()) {
      write_single_element(index, h5UVW_.id(), h5UVWType_, uvw.data());
    } else {
      // TODO
    }

    write_single_element(index, h5Wl_.id(), get_type_id<float>(), &wl);
    unsigned int nVisOut = static_cast<unsigned int>(nVis);
    write_single_element(index, h5NVis_.id(), get_type_id<decltype(nVisOut)>(), &nVisOut);

    const auto uMM = std::minmax_element(&uvw[{0, 0}], &uvw[{0, 0}] + uvw.shape(0));
    const auto vMM = std::minmax_element(&uvw[{0, 1}], &uvw[{0, 1}] + uvw.shape(0));
    const auto wMM = std::minmax_element(&uvw[{0, 2}], &uvw[{0, 2}] + uvw.shape(0));

    write_single_element(index, h5MinU_.id(), get_type_id<float>(), uMM.first);
    write_single_element(index, h5MaxU_.id(), get_type_id<float>(), uMM.second);

    write_single_element(index, h5MinV_.id(), get_type_id<float>(), vMM.first);
    write_single_element(index, h5MaxV_.id(), get_type_id<float>(), vMM.second);

    write_single_element(index, h5MinW_.id(), get_type_id<float>(), wMM.first);
    write_single_element(index, h5MaxW_.id(), get_type_id<float>(), wMM.second);
  }

private:
  hsize_t nAntenna_, nBeam_;

  // counter. Equal to outer dimension size
  hsize_t count_ = 0;

  // array types
  hid_t h5EigValType_ = H5I_INVALID_HID;
  hid_t h5EigVecType_ = H5I_INVALID_HID;
  hid_t h5UVWType_ = H5I_INVALID_HID;

  // datasets
  DataSet h5EigVal_;
  DataSet h5EigVec_;
  DataSet h5UVW_;
  DataSet h5Wl_;
  DataSet h5NVis_;
  DataSet h5MinU_;
  DataSet h5MinV_;
  DataSet h5MinW_;
  DataSet h5MaxU_;
  DataSet h5MaxV_;
  DataSet h5MaxW_;

  // file
  hid_t h5File_ = H5I_INVALID_HID;
};

auto DatasetWriter::DatasetWriterImplDeleter::operator()(DatasetWriterImpl* p) -> void {
  if (p) delete p;
}

DatasetWriter::DatasetWriter(const std::string& fileName, const std::string& datasetName,
                             std::size_t nAntenna, std::size_t nBeam)
    : impl_(new DatasetWriterImpl(fileName, datasetName, nAntenna, nBeam)) {}

auto DatasetWriter::write(ValueType wl, std::size_t nVis,
                          ConstHostView<std::complex<ValueType>, 2> v,
                          ConstHostView<ValueType, 1> d, ConstHostView<ValueType, 2> uvw) -> void {
  impl_->write(wl, nVis, v, d, uvw);
}

}  // namespace bipp
