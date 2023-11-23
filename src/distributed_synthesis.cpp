#include "distributed_synthesis.hpp"

#include <complex>
#include <cstddef>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "communicator_internal.hpp"
#include "context_internal.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"
#include "host/domain_partition.hpp"


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
      count_(0),
      img_(ctx_->host_alloc(), {filter.size(), nLevel, pixelX.size()}),
      imgPartition_(host::DomainPartition::none(ctx_, pixelX.size())) {
  if(!comm_->is_root()) throw InvalidParameterError();

  assert(comm_->comm().rank() == 0);
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());

  img_.zero();

  imgPartition_ = host::DomainPartition::grid<T, 3>(
      ctx_, {std::size_t(comm_->comm().size()) - 1, 1, 1}, {pixelX, pixelY, pixelZ});

  ConstHostView<PartitionGroup, 1> groups(imgPartition_.groups().data(), imgPartition_.groups().size(), 1);
  id_ = comm_->send_synthesis_init<T>(std::move(nufftOpt), nLevel, filter, pixelX, pixelY, pixelZ,
                                      groups);
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
    comm_->send_synthesis_collect<T>(id_, v, dMasked, xyzUvw);
#else
    throw GPUSupportError();
#endif
  } else {
    auto v = ConstHostView<std::complex<T>, 2>(vView);
    auto xyzUvw = ConstHostView<T, 2>(xyzUvwView);
    comm_->send_synthesis_collect<T>(id_, v, dMasked, xyzUvw);
  }
}

template <typename T>
auto DistributedSynthesis<T>::get(BippFilter f, View<T, 2> out) -> void {
  //TODO
}

template class DistributedSynthesis<float>;
template class DistributedSynthesis<double>;


}  // namespace bipp
