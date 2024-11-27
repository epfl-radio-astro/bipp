#pragma once

#include <hdf5.h>

#include <array>
#include <string>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"

namespace bipp {
namespace h5 {

class IdWrapper {
public:
  IdWrapper(hid_t identifier, herr_t (*close)(hid_t)) noexcept : id_(identifier), close_(close) {}

  IdWrapper() noexcept : id_(H5I_INVALID_HID), close_(nullptr) {}

  IdWrapper(const IdWrapper&) = delete;
  IdWrapper(IdWrapper&& s) { *this = std::move(s); }

  auto operator=(const IdWrapper&) -> IdWrapper& = delete;

  auto operator=(IdWrapper&& s) noexcept -> IdWrapper& {
    if (id_ != H5I_INVALID_HID) {
      close_(id_);
    }
    id_ = s.id_;
    close_ = s.close_;
    s.id_ = H5I_INVALID_HID;
    s.close_ = nullptr;
    return *this;
  }

  ~IdWrapper() noexcept {
    if (id_ != H5I_INVALID_HID) {
      close_(id_);
    }
  }

  inline auto id() noexcept -> hid_t { return id_; }

private:
  hid_t id_;
  herr_t (*close_)(hid_t);
};

class DataSet {
public:
  DataSet(hid_t identifier) noexcept : wrapper_(identifier, H5Dclose) {}

  DataSet() = default;

  inline auto id() noexcept -> hid_t { return wrapper_.id(); }

private:
  IdWrapper wrapper_;
};

class DataSpace {
public:
  DataSpace(hid_t identifier) noexcept : wrapper_(identifier, H5Sclose) {}

  DataSpace() = default;

  inline auto id() noexcept -> hid_t { return wrapper_.id(); }

private:
  IdWrapper wrapper_;
};

class DataType {
public:
  DataType(hid_t identifier) noexcept : wrapper_(identifier, H5Tclose) {}

  DataType() = default;

  inline auto id() noexcept -> hid_t { return wrapper_.id(); }

private:
  IdWrapper wrapper_;
};

class File {
public:
  File(hid_t identifier) noexcept : wrapper_(identifier, H5Fclose) {}

  File() = default;

  inline auto id() noexcept -> hid_t { return wrapper_.id(); }

private:
  IdWrapper wrapper_;
};

template <typename T>
inline auto get_type_id() -> hid_t {
  if constexpr (std::is_same_v<T, float>) {
    return H5T_NATIVE_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return H5T_NATIVE_DOUBLE;
  } else if constexpr (std::is_same_v<T, unsigned long long>) {
    return H5T_NATIVE_ULLONG;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return H5T_NATIVE_ULONG;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return H5T_NATIVE_UINT;
  } else if constexpr (std::is_same_v<T, int>) {
    return H5T_NATIVE_INT;
  } else if constexpr (std::is_same_v<T, long>) {
    return H5T_NATIVE_LONG;
  } else if constexpr (std::is_same_v<T, long long>) {
    return H5T_NATIVE_LLONG;
  } else {
    throw InternalError("unkown hdf5 type");
  }
  return H5I_INVALID_HID;
}

inline auto check(hid_t h) -> hid_t {
  if (h == H5I_INVALID_HID) {
    throw HDF5Error();
  }
  return h;
}

inline auto check(int h) -> void {
  if (h < 0) {
    throw HDF5Error();
  }
}

inline auto create_string_attr(hid_t hid, const std::string& name, const std::string& value) -> void {
  auto attSpace = check(H5Screate(H5S_SCALAR));

  auto attType = check(H5Tcopy(H5T_C_S1));

  check(H5Tset_size(attType, value.size()));

  auto attId = check(H5Acreate(hid, name.data(), attType, attSpace, H5P_DEFAULT, H5P_DEFAULT));

  check(H5Awrite(attId, attType, value.data()));

  check(H5Sclose(attSpace));
  check(H5Tclose(attType));
  check(H5Aclose(attId));
}

inline auto create_size_attr(hid_t hid, const std::string& name, unsigned int size) -> void {
  auto attSpace = check(H5Screate(H5S_SCALAR));

  auto attId = check(H5Acreate(hid, name.data(), get_type_id<decltype(size)>(), attSpace,
                                    H5P_DEFAULT, H5P_DEFAULT));

  check(H5Awrite(attId, get_type_id<decltype(size)>(), &size));

  check(H5Sclose(attSpace));
  check(H5Aclose(attId));
}

inline auto create_one_dim_space(hid_t fd, const std::string& name, hid_t type,
                          hsize_t chunkSize) -> hid_t {
  std::array<hsize_t, 1> chunks = {chunkSize};
  std::array<hsize_t, 1> dims = {0};
  std::array<hsize_t, 1> maxDims = {H5S_UNLIMITED};

  auto dataspace = check(H5Screate_simple(1, dims.data(), maxDims.data()));
  auto cparms = check(H5Pcreate(H5P_DATASET_CREATE));
  check(H5Pset_chunk(cparms, 1, chunks.data()));

  auto arr =
      check(H5Dcreate(fd, name.data(), type, dataspace, H5P_DEFAULT, cparms, H5P_DEFAULT));

  H5Sclose(dataspace);
  H5Pclose(cparms);

  return arr;
}

inline auto write_single_element(hsize_t index, hid_t dset, hid_t type, const void* data) {
  const hsize_t one = 1;
  DataSpace dspace(check(H5Dget_space(dset)));
  check(H5Sselect_elements(dspace.id(), H5S_SELECT_SET, 1, &index));

  DataSpace mspace = check(H5Screate_simple(1, &one, &one));

  check(H5Dwrite(dset, type, mspace.id(), dspace.id(), H5P_DEFAULT, data));
}

}  // namespace h5
}  // namespace bipp
