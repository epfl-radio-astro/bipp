#include "distributed_synthesis.hpp"

#include <complex>
#include <cstddef>
#include <variant>

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
    std::variant<NufftSynthesisOptions, StandardSynthesisOptions> opt, std::size_t nLevel,
    ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY, ConstHostView<T, 1> pixelZ)
    : comm_(std::move(comm)),
      ctx_(std::move(ctx)),
      totalCollectCount_(0),
      img_(ctx_->host_alloc(), {pixelX.size(), nLevel}),
      imgPartition_(host::DomainPartition::none(ctx_, pixelX.size())),
      type_(std::holds_alternative<NufftSynthesisOptions>(opt) ? SynthesisType::NUFFT
                                                               : SynthesisType::Standard) {
  if (!comm_->is_root()) throw InvalidParameterError();

  assert(comm_->comm().rank() == 0);
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());

  imgPartition_ = host::DomainPartition::grid<T, 3>(
      ctx_, {std::size_t(comm_->comm().size()) - 1, 1, 1}, {pixelX, pixelY, pixelZ});

  if (imgPartition_.groups().size() != comm_->comm().size() - 1)
    throw InternalError("Distributed image partition failed.");

  HostArray <T, 2> pixel(ctx_->host_alloc(), {pixelX.size(), 3});
  imgPartition_.apply(pixelX, pixel.slice_view(0));
  imgPartition_.apply(pixelY, pixel.slice_view(1));
  imgPartition_.apply(pixelZ, pixel.slice_view(2));

  ConstHostView<PartitionGroup, 1> groups(imgPartition_.groups().data(), imgPartition_.groups().size(), 1);

  if(std::holds_alternative<StandardSynthesisOptions>(opt)) {
    id_ = comm_->send_standard_synthesis_init<T>(std::get<StandardSynthesisOptions>(opt), nLevel,
                                                 pixel.slice_view(0), pixel.slice_view(1),
                                                 pixel.slice_view(2), groups);
  } else {
    id_ = comm_->send_nufft_synthesis_init<T>(std::get<NufftSynthesisOptions>(opt), nLevel,
                                              pixel.slice_view(0), pixel.slice_view(1),
                                              pixel.slice_view(2), groups);
  }
}

template <typename T>
auto DistributedSynthesis<T>::process(CollectorInterface<T>& collector) -> void {
  totalCollectCount_ += collector.size();
  comm_->send_synthesis_collect<T>(id_, collector);
}

template <typename T>
auto DistributedSynthesis<T>::get(View<T, 2> out) -> void {
  ConstHostView<PartitionGroup, 1> groups(imgPartition_.groups().data(),
                                          imgPartition_.groups().size(), 1);
  comm_->gather_image<T>(id_, groups, img_);

  const T scale =
      totalCollectCount_ ? static_cast<T>(1.0 / static_cast<double>(totalCollectCount_)) : 0;

  HostArray<T, 1> buffer(ctx_->host_alloc(), img_.shape(0));
  for(std::size_t idxLevel = 0; idxLevel < img_.shape(1); ++idxLevel) {
    auto levelImg = img_.slice_view(idxLevel);
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
    copy(queue, img_, out);
#else
    throw GPUSupportError();
#endif
  } else {
    copy(img_, HostView<T, 2>(out));
  }
}

template class DistributedSynthesis<float>;
template class DistributedSynthesis<double>;

}  // namespace bipp
