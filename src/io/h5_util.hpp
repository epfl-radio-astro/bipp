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

  IdWrapper() noexcept = default;

  IdWrapper(const IdWrapper&) = delete;
  IdWrapper(IdWrapper&& s) noexcept { *this = std::move(s); }

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

  inline auto id() const noexcept -> const hid_t& { return id_; }

private:
  hid_t id_ = H5I_INVALID_HID;
  herr_t (*close_)(hid_t) = nullptr;
};

class Group {
public:
  Group(hid_t identifier) noexcept : wrapper_(identifier, H5Gclose) {}

  Group() = default;

  inline auto id() const noexcept -> hid_t { return wrapper_.id(); }

private:
  IdWrapper wrapper_;
};

class DataSet {
public:
  DataSet(hid_t identifier) noexcept : wrapper_(identifier, H5Dclose) {}

  DataSet() = default;

  inline auto id() const noexcept -> hid_t { return wrapper_.id(); }

private:
  IdWrapper wrapper_;
};

class DataSpace {
public:
  DataSpace(hid_t identifier) noexcept : wrapper_(identifier, H5Sclose) {}

  DataSpace() = default;

  inline auto id() const noexcept -> hid_t { return wrapper_.id(); }

private:
  IdWrapper wrapper_;
};

class DataType {
public:
  DataType(hid_t identifier) noexcept : wrapper_(identifier, H5Tclose) {}

  DataType() = default;

  inline auto id() const noexcept -> hid_t { return wrapper_.id(); }

private:
  IdWrapper wrapper_;
};

class Attribute {
public:
  Attribute(hid_t identifier) noexcept : wrapper_(identifier, H5Aclose) {}

  Attribute() = default;

  inline auto id() const noexcept -> hid_t { return wrapper_.id(); }

private:
  IdWrapper wrapper_;
};

class Property {
public:
  Property(hid_t identifier) noexcept : wrapper_(identifier, H5Pclose) {}

  Property() = default;

  inline auto id() const noexcept -> hid_t { return wrapper_.id(); }

private:
  IdWrapper wrapper_;
};

class File {
public:
  File(hid_t identifier) noexcept : wrapper_(identifier, H5Fclose) {}

  File() = default;

  inline auto id() const noexcept -> hid_t { return wrapper_.id(); }

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

inline auto create_string_attr(hid_t hid, const std::string& name, const std::string& value) -> void {
  DataSpace attSpace = check(H5Screate(H5S_SCALAR));

  DataType attType = check(H5Tcopy(H5T_C_S1));

  check(H5Tset_size(attType.id(), value.size()));

  Attribute attr = check(H5Acreate(hid, name.data(), attType.id(), attSpace.id(), H5P_DEFAULT, H5P_DEFAULT));

  check(H5Awrite(attr.id(), attType.id(), value.data()));
}

inline auto read_string_attr(hid_t hid, const std::string& name) -> std::string {
  Attribute attr = check(H5Aopen(hid, name.data(), H5P_DEFAULT));
  DataType type = check(H5Aget_type(attr.id()));
  auto size = H5Tget_size(type.id());
  if (size < 0) throw HDF5Error();

  std::string value;
  value.resize(size);
  check(H5Aread(attr.id(), type.id(), value.data()));

  return value;
}

inline auto create_size_attr(hid_t hid, const std::string& name, unsigned int size) -> void {
  DataSpace attSpace = check(H5Screate(H5S_SCALAR));

  Attribute attr = check(H5Acreate(hid, name.data(), get_type_id<decltype(size)>(), attSpace.id(),
                                    H5P_DEFAULT, H5P_DEFAULT));

  check(H5Awrite(attr.id(), get_type_id<decltype(size)>(), &size));
}

inline auto read_size_attr(hid_t hid, const std::string& name) -> unsigned int {
  Attribute attr = check(H5Aopen(hid, name.data(), H5P_DEFAULT));

  unsigned int size = 0;
  check(H5Aread(attr.id(), get_type_id<decltype(size)>(), &size));

  return size;
}

inline auto create_float_attr(hid_t hid, const std::string& name, float value) -> void {
  DataSpace attSpace = check(H5Screate(H5S_SCALAR));

  Attribute attr = check(H5Acreate(hid, name.data(), get_type_id<decltype(value)>(), attSpace.id(),
                                    H5P_DEFAULT, H5P_DEFAULT));

  check(H5Awrite(attr.id(), get_type_id<decltype(value)>(), &value));
}

inline auto read_float_attr(hid_t hid, const std::string& name) -> float {
  Attribute attr = check(H5Aopen(hid, name.data(), H5P_DEFAULT));

  float value = 0;
  check(H5Aread(attr.id(), get_type_id<decltype(value)>(), &value));

  return value;
}

inline auto create_fixed_one_dim_space(hid_t fd, const std::string& name, hid_t type, hsize_t size) -> hid_t {
  DataSpace dataspace = check(H5Screate_simple(1, &size, &size));

  auto arr = check(
      H5Dcreate(fd, name.data(), type, dataspace.id(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  return arr;
}

inline auto create_one_dim_space(hid_t fd, const std::string& name, hid_t type,
                          hsize_t chunkSize) -> hid_t {
  std::array<hsize_t, 1> chunks = {chunkSize};
  std::array<hsize_t, 1> dims = {0};
  std::array<hsize_t, 1> maxDims = {H5S_UNLIMITED};

  DataSpace dataspace = check(H5Screate_simple(1, dims.data(), maxDims.data()));
  Property cparms = check(H5Pcreate(H5P_DATASET_CREATE));
  check(H5Pset_chunk(cparms.id(), 1, chunks.data()));

  auto arr = check(
      H5Dcreate(fd, name.data(), type, dataspace.id(), H5P_DEFAULT, cparms.id(), H5P_DEFAULT));

  return arr;
}

inline auto write_single_element(hsize_t index, hid_t dset, hid_t type, const void* data) {
  const hsize_t one = 1;
  DataSpace dspace(check(H5Dget_space(dset)));
  check(H5Sselect_elements(dspace.id(), H5S_SELECT_SET, 1, &index));

  DataSpace mspace = check(H5Screate_simple(1, &one, &one));

  check(H5Dwrite(dset, type, mspace.id(), dspace.id(), H5P_DEFAULT, data));
}

inline auto read_single_element(hsize_t index, hid_t dset, hid_t type, void* data) {
  const hsize_t one = 1;
  DataSpace dspace(check(H5Dget_space(dset)));
  check(H5Sselect_elements(dspace.id(), H5S_SELECT_SET, 1, &index));

  DataSpace mspace = check(H5Screate_simple(1, &one, &one));

  check(H5Dread(dset, type, mspace.id(), dspace.id(), H5P_DEFAULT, data));
}

}  // namespace h5
}  // namespace bipp
