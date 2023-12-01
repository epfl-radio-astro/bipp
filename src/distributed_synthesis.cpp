#include "distributed_synthesis.hpp"

#include <complex>
#include <cstddef>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "communicator_internal.hpp"
#include "context_internal.hpp"
#include "host/domain_partition.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/device_accessor.hpp"
#endif

namespace bipp {

template <typename T>
DistributedSynthesis<T>::DistributedSynthesis(
    std::shared_ptr<CommunicatorInternal> comm, std::shared_ptr<ContextInternal> ctx,
    std::optional<NufftSynthesisOptions> nufftOpt, std::size_t nLevel,
    ConstHostView<BippFilter, 1> filter, ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
    ConstHostView<T, 1> pixelZ)
    : comm_(std::move(comm)),
      ctx_(std::move(ctx)),
      totalCollectCount_(0),
      img_(ctx_->host_alloc(), {pixelX.size(), nLevel, filter.size()}),
      filter_(ctx_->host_alloc(), filter.shape()),
      imgPartition_(host::DomainPartition::none(ctx_, pixelX.size())),
      type_(nufftOpt ? SynthesisType::NUFFT : SynthesisType::Standard) {
  if(!comm_->is_root()) throw InvalidParameterError();

  assert(comm_->comm().rank() == 0);
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());

  copy(filter, filter_);

  imgPartition_ = host::DomainPartition::grid<T, 3>(
      ctx_, {std::size_t(comm_->comm().size()) - 1, 1, 1}, {pixelX, pixelY, pixelZ});

  HostArray <T, 2> pixel(ctx_->host_alloc(), {pixelX.size(), 3});
  imgPartition_.apply(pixelX, pixel.slice_view(0));
  imgPartition_.apply(pixelY, pixel.slice_view(1));
  imgPartition_.apply(pixelZ, pixel.slice_view(2));

  ConstHostView<PartitionGroup, 1> groups(imgPartition_.groups().data(), imgPartition_.groups().size(), 1);
  id_ = comm_->send_synthesis_init<T>(std::move(nufftOpt), nLevel, filter, pixel.slice_view(0),
                                      pixel.slice_view(1), pixel.slice_view(2), groups);
}

template <typename T>
auto DistributedSynthesis<T>::collect(T wl, ConstView<std::complex<T>, 2> vView,
                                      ConstHostView<T, 2> dMasked, ConstView<T, 2> xyzUvwView)
    -> void {
  if (ctx_->processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    auto& queue = ctx_->gpu_queue();
    ConstHostAccessor<T, 2> xyzUvwDevice(queue, xyzUvwView);
    ConstHostAccessor<std::complex<T>, 2> vDevice(queue, vView);

    auto v = vDevice.view();
    auto xyzUvw = xyzUvwDevice.view();
    queue.sync();
    comm_->send_synthesis_collect<T>(id_, wl, v, dMasked, xyzUvw);
#else
    throw GPUSupportError();
#endif
  } else {
    auto v = ConstHostView<std::complex<T>, 2>(vView);
    auto xyzUvw = ConstHostView<T, 2>(xyzUvwView);
    comm_->send_synthesis_collect<T>(id_, wl, v, dMasked, xyzUvw);
  }
  ++totalCollectCount_;
}

template <typename T>
auto DistributedSynthesis<T>::get(BippFilter f, View<T, 2> out) -> void {
  std::size_t index = filter_.size();
  for (std::size_t i = 0; i < filter_.size(); ++i) {
    if (filter_[{i}] == f) {
      index = i;
      break;
    }
  }
  if (index == filter_.size()) throw InvalidParameterError();

  auto filterImg = img_.slice_view(index);

  ConstHostView<PartitionGroup, 1> groups(imgPartition_.groups().data(),
                                          imgPartition_.groups().size(), 1);
  comm_->gather_image<T>(id_, index, groups, filterImg);

  const T scale =
      totalCollectCount_ ? static_cast<T>(1.0 / static_cast<double>(totalCollectCount_)) : 0;

  HostArray<T, 1> buffer(ctx_->host_alloc(), filterImg.shape(0));
  for(std::size_t idxLevel = 0; idxLevel < filterImg.shape(1); ++idxLevel) {
    auto levelImg = filterImg.slice_view(idxLevel);
    imgPartition_.reverse<T>(levelImg, buffer);

    T* __restrict__ imgPtr = levelImg.data();
    const T* __restrict__ bufferPtr = buffer.data();
    for(std::size_t i = 0; i < levelImg.size();++i) {
      imgPtr[i] = bufferPtr[i] * scale;
    }
  }

  if (ctx_->processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    auto& queue = ctx_->gpu_queue();
    copy(queue, filterImg, out);
#else
    throw GPUSupportError();
#endif
  } else {
    copy(filterImg, HostView<T, 2>(out));
  }
}

template class DistributedSynthesis<float>;
template class DistributedSynthesis<double>;


}  // namespace bipp
