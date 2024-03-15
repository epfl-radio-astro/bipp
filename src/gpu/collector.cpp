#include "gpu/collector.hpp"

#include <cassert>
#include <complex>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "collector_interface.hpp"
#include "context_internal.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace gpu {

namespace {
template <typename T>
struct SerialInfo {
  T wl;
  std::size_t nVis;
  std::size_t vShape[2];
  std::size_t dMaskedShape[2];
  std::size_t xyzUvwShape[2];
};
}  // namespace

template <typename T>
Collector<T>::Collector(std::shared_ptr<ContextInternal> ctx) : ctx_(std::move(ctx)) {
  assert(ctx_->processing_unit() == BIPP_PU_GPU);
  wlData_.reserve(100);
  nVisData_.reserve(100);
  vData_.reserve(100);
  dMaskedData_.reserve(100);
  xyzUvwData_.reserve(100);
}

template <typename T>
auto Collector<T>::clear() -> void {
  wlData_.clear();
  nVisData_.clear();
  vData_.clear();
  dMaskedData_.clear();
  xyzUvwData_.clear();
  wlData_.reserve(100);
  nVisData_.reserve(100);
  vData_.reserve(100);
  dMaskedData_.reserve(100);
  xyzUvwData_.reserve(100);
}

template <typename T>
auto Collector<T>::collect(T wl, const std::size_t nVis, ConstView<std::complex<T>, 2> v,
                           ConstHostView<T, 2> dMasked, ConstView<T, 2> xyzUvw) -> void {

  auto& queue = ctx_->gpu_queue();

  wlData_.emplace_back(wl);

  nVisData_.emplace_back(nVis);

  vData_.emplace_back(queue.create_device_array<std::complex<T>, 2>(v.shape()));
  copy(queue, v, vData_.back());

  dMaskedData_.emplace_back(queue.create_host_array<T,2>(dMasked.shape()));
  copy(queue, dMasked, dMaskedData_.back());

  xyzUvwData_.emplace_back(queue.create_pinned_array<T, 2>(xyzUvw.shape()));
  copy(queue, xyzUvw, xyzUvwData_.back());

  queue.sync();
}

template <typename T>
auto Collector<T>::get_data() const -> std::vector<typename CollectorInterface<T>::Data> {
  using DataType = typename CollectorInterface<T>::Data;
  std::vector<DataType> data;
  data.reserve(this->size());
  for(std::size_t i = 0; i < this->size(); ++i) {
    data.emplace_back(wlData_[i], nVisData_[i], vData_[i], dMaskedData_[i], xyzUvwData_[i]);
  }
  return data;
}

template <typename T>
auto Collector<T>::serialize() const -> HostArray<char, 1> {
  // serialize all content
  // Note: to avoid alignment issues, only read / write through memcpy

  auto& queue = ctx_->gpu_queue();

  std::size_t totalNumBytes = 0;

  // number of collected steps at first position
  totalNumBytes += sizeof(std::size_t);

  for(std::size_t i = 0; i < wlData_.size(); ++i) {
    totalNumBytes += sizeof(SerialInfo<T>);
    totalNumBytes += sizeof(std::size_t);
    totalNumBytes += vData_[i].size_in_bytes();
    totalNumBytes += dMaskedData_[i].size_in_bytes();
    totalNumBytes += xyzUvwData_[i].size_in_bytes();
  }


  // create serilized host array
  auto data = queue.create_pinned_array<char, 1>(totalNumBytes);

  // copy number of steps first
  const std::size_t numSteps = wlData_.size();
  std::memcpy(data.data(), &numSteps, sizeof(decltype(numSteps)));


  // copy each step data sequentially
  std::size_t currentOffset = sizeof(decltype(numSteps));
  for (std::size_t i = 0; i < numSteps; ++i) {
    SerialInfo<T> info;
    info.wl = wlData_[i];
    info.nVis = nVisData_[i];
    info.vShape[0] = vData_[i].shape(0);
    info.vShape[1] = vData_[i].shape(1);
    info.dMaskedShape[0] = dMaskedData_[i].shape(0);
    info.dMaskedShape[1] = dMaskedData_[i].shape(1);
    info.xyzUvwShape[0] = xyzUvwData_[i].shape(0);
    info.xyzUvwShape[1] = xyzUvwData_[i].shape(1);

    std::memcpy(data.data() + currentOffset, &info, sizeof(decltype(info)));
    currentOffset += sizeof(SerialInfo<T>);

    copy(queue, vData_[i],
         HostView<std::complex<T>, 2>(
             reinterpret_cast<std::complex<T>*>(data.data() + currentOffset), vData_[i].shape(),
             {1, vData_[i].shape(0)}));
    currentOffset += vData_[i].size_in_bytes();

    copy(dMaskedData_[i], HostView<T, 2>(reinterpret_cast<T*>(data.data() + currentOffset),
                                         dMaskedData_[i].shape(), {1, dMaskedData_[i].shape(0)}));
    currentOffset += dMaskedData_[i].size_in_bytes();

    copy(xyzUvwData_[i], HostView<T, 2>(reinterpret_cast<T*>(data.data() + currentOffset),
                                        xyzUvwData_[i].shape(), {1, xyzUvwData_[i].shape(0)}));
    currentOffset += xyzUvwData_[i].size_in_bytes();

    assert(currentOffset <= data.size_in_bytes());
  }

  queue.sync();

  return data;
}

template <typename T>
auto Collector<T>::deserialize(ConstHostView<char, 1> serialData) -> void {
  // Note: to avoid alignment issues, only read / write through memcpy

  auto& queue = ctx_->gpu_queue();

  // clear all current data
  wlData_.clear();
  nVisData_.clear();
  vData_.clear();
  dMaskedData_.clear();
  xyzUvwData_.clear();

  // read total number of steps
  std::size_t numSteps = 0;
  std::memcpy(&numSteps, serialData.data(), sizeof(decltype(numSteps)));

  // reserve for received number of steps
  wlData_.reserve(numSteps);
  nVisData_.reserve(numSteps);
  vData_.reserve(numSteps);
  dMaskedData_.reserve(numSteps);
  xyzUvwData_.reserve(numSteps);

  std::size_t currentOffset = sizeof(decltype(numSteps));
  for (std::size_t i = 0; i < numSteps; ++i) {
    SerialInfo<T> info;

    std::memcpy(&info, serialData.data() + currentOffset, sizeof(decltype(info)));
    currentOffset += sizeof(SerialInfo<T>);

    wlData_.emplace_back(info.wl);

    nVisData_.emplace_back(info.nVis);

    vData_.emplace_back(
        queue.create_device_array<std::complex<T>, 2>({info.vShape[0], info.vShape[1]}));
    copy(queue,
         ConstHostView<std::complex<T>, 2>(
             reinterpret_cast<const std::complex<T>*>(serialData.data() + currentOffset),
             vData_[i].shape(), {1, vData_[i].shape(0)}),
         vData_[i]);
    currentOffset += vData_[i].size_in_bytes();

    dMaskedData_.emplace_back(
        queue.create_host_array<T, 2>({info.dMaskedShape[0], info.dMaskedShape[1]}));
    copy(ConstHostView<T, 2>(reinterpret_cast<const T*>(serialData.data() + currentOffset),
                             dMaskedData_[i].shape(), {1, dMaskedData_[i].shape(0)}),
         dMaskedData_[i]);
    currentOffset += dMaskedData_[i].size_in_bytes();

    xyzUvwData_.emplace_back(
        queue.create_pinned_array<T, 2>({info.xyzUvwShape[0], info.xyzUvwShape[1]}));
    copy(ConstHostView<T, 2>(reinterpret_cast<const T*>(serialData.data() + currentOffset),
                             xyzUvwData_[i].shape(), {1, xyzUvwData_[i].shape(0)}),
         xyzUvwData_[i]);
    currentOffset += xyzUvwData_[i].size_in_bytes();

    assert(currentOffset <= serialData.size_in_bytes());
  }

  queue.sync();
}

template class Collector<float>;
template class Collector<double>;

}  // namespace gpu
}  // namespace bipp
