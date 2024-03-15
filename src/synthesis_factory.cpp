
#include "synthesis_factory.hpp"

#include <cstddef>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "bipp/nufft_synthesis.hpp"
#include "context_internal.hpp"
#include "host/nufft_synthesis.hpp"
#include "host/standard_synthesis.hpp"
#include "memory/view.hpp"
#include "synthesis_interface.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/nufft_synthesis.hpp"
#include "gpu/standard_synthesis.hpp"
#include "gpu/util/device_accessor.hpp"
#include "gpu/util/device_guard.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

#ifdef BIPP_MPI
#include "distributed_synthesis.hpp"
#endif

namespace bipp {

template <typename T>
auto SynthesisFactory<T>::create_standard_synthesis(std::shared_ptr<ContextInternal> ctx,
                                                    StandardSynthesisOptions opt,
                                                    std::size_t nImages, ConstView<T, 1> pixelX,
                                                    ConstView<T, 1> pixelY, ConstView<T, 1> pixelZ)
    -> std::unique_ptr<SynthesisInterface<T>> {
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());
  assert(ctx);

  auto t = ctx->logger().scoped_timing(BIPP_LOG_LEVEL_INFO, "Create StandardSynthesis");

  if (ctx->processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    gpu::DeviceGuard deviceGuard(ctx->device_id());

    auto& queue = ctx->gpu_queue();
    // Syncronize with default stream.
    queue.sync_with_stream(nullptr);
    // syncronize with stream to be synchronous with host before exiting
    auto syncGuard = queue.sync_guard();

    auto pixelArray = queue.create_device_array<T, 2>({pixelX.size(), 3});
    copy(queue, pixelX, pixelArray.slice_view(0));
    copy(queue, pixelY, pixelArray.slice_view(1));
    copy(queue, pixelZ, pixelArray.slice_view(2));

    return std::make_unique<gpu::StandardSynthesis<T>>(ctx, std::move(opt), nImages,
                                                       std::move(pixelArray));
#else
    throw GPUSupportError();
#endif
  }

  return std::make_unique<host::StandardSynthesis<T>>(
      std::move(ctx), std::move(opt), nImages, ConstHostView<T, 1>(pixelX),
      ConstHostView<T, 1>(pixelY), ConstHostView<T, 1>(pixelZ));
}

template <typename T>
auto SynthesisFactory<T>::create_nufft_synthesis(std::shared_ptr<ContextInternal> ctx,
                                                 NufftSynthesisOptions opt, std::size_t nImages,
                                                 ConstView<T, 1> pixelX, ConstView<T, 1> pixelY,
                                                 ConstView<T, 1> pixelZ)
    -> std::unique_ptr<SynthesisInterface<T>> {
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());
  assert(ctx);

  auto t = ctx->logger().scoped_timing(BIPP_LOG_LEVEL_INFO, "Create NUFFTSynthesis");

  if (ctx->processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    gpu::DeviceGuard deviceGuard(ctx->device_id());

    auto& queue = ctx->gpu_queue();
    // Syncronize with default stream.
    queue.sync_with_stream(nullptr);
    // syncronize with stream to be synchronous with host before exiting
    auto syncGuard = queue.sync_guard();

    auto pixelArray = queue.create_device_array<T, 2>({pixelX.size(), 3});
    copy(queue, pixelX, pixelArray.slice_view(0));
    copy(queue, pixelY, pixelArray.slice_view(1));
    copy(queue, pixelZ, pixelArray.slice_view(2));

    return std::make_unique<gpu::NufftSynthesis<T>>(ctx, std::move(opt), nImages,
                                                    std::move(pixelArray));
#else
    throw GPUSupportError();
#endif
  }

  return std::make_unique<host::NufftSynthesis<T>>(
      std::move(ctx), opt, nImages, ConstHostView<T, 1>(pixelX), ConstHostView<T, 1>(pixelY),
      ConstHostView<T, 1>(pixelZ));
}


#ifdef BIPP_MPI

template <typename T>
auto SynthesisFactory<T>::create_distributed_standard_synthesis(
    std::shared_ptr<CommunicatorInternal> comm, std::shared_ptr<ContextInternal> ctx,
    StandardSynthesisOptions opt, std::size_t nImages, ConstView<T, 1> pixelX,
    ConstView<T, 1> pixelY, ConstView<T, 1> pixelZ) -> std::unique_ptr<SynthesisInterface<T>> {
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());
  assert(ctx);

  auto t = ctx->logger().scoped_timing(BIPP_LOG_LEVEL_INFO, "Create Distributed StandardSynthesis");

  if (ctx->processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    gpu::DeviceGuard deviceGuard(ctx->device_id());

    auto& queue = ctx->gpu_queue();
    // Syncronize with default stream.
    queue.sync_with_stream(nullptr);

    auto pixelArray = queue.create_device_array<T, 2>({pixelX.size(), 3});
    copy(queue, pixelX, pixelArray.slice_view(0));
    copy(queue, pixelY, pixelArray.slice_view(1));
    copy(queue, pixelZ, pixelArray.slice_view(2));

    ConstHostAccessor<T, 1> pixelXHost(queue, pixelX);
    ConstHostAccessor<T, 1> pixelYHost(queue, pixelY);
    ConstHostAccessor<T, 1> pixelZHost(queue, pixelZ);
    queue.sync();

    return std::make_unique<DistributedSynthesis<T>>(std::move(comm), std::move(ctx), opt, nImages,
                                                     pixelXHost.view(), pixelYHost.view(),
                                                     pixelZHost.view());
#else
    throw GPUSupportError();
#endif
  }

  return std::make_unique<DistributedSynthesis<T>>(
      std::move(comm), std::move(ctx), opt, nImages, ConstHostView<T, 1>(pixelX),
      ConstHostView<T, 1>(pixelY), ConstHostView<T, 1>(pixelZ));
}

template <typename T>
auto SynthesisFactory<T>::create_distributed_nufft_synthesis(
    std::shared_ptr<CommunicatorInternal> comm, std::shared_ptr<ContextInternal> ctx,
    NufftSynthesisOptions opt, std::size_t nImages, ConstView<T, 1> pixelX, ConstView<T, 1> pixelY,
    ConstView<T, 1> pixelZ) -> std::unique_ptr<SynthesisInterface<T>> {
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());
  assert(ctx);

  auto t = ctx->logger().scoped_timing(BIPP_LOG_LEVEL_INFO, "Create Distributed NufftSynthesis");

  if (ctx->processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    gpu::DeviceGuard deviceGuard(ctx->device_id());

    auto& queue = ctx->gpu_queue();
    // Syncronize with default stream.
    queue.sync_with_stream(nullptr);

    auto pixelArray = queue.create_device_array<T, 2>({pixelX.size(), 3});
    copy(queue, pixelX, pixelArray.slice_view(0));
    copy(queue, pixelY, pixelArray.slice_view(1));
    copy(queue, pixelZ, pixelArray.slice_view(2));

    ConstHostAccessor<T, 1> pixelXHost(queue, pixelX);
    ConstHostAccessor<T, 1> pixelYHost(queue, pixelY);
    ConstHostAccessor<T, 1> pixelZHost(queue, pixelZ);
    queue.sync();

    return std::make_unique<DistributedSynthesis<T>>(std::move(comm), std::move(ctx), opt, nImages,
                                                     pixelXHost.view(), pixelYHost.view(),
                                                     pixelZHost.view());
#else
    throw GPUSupportError();
#endif
  }

  return std::make_unique<DistributedSynthesis<T>>(
      std::move(comm), std::move(ctx), opt, nImages, ConstHostView<T, 1>(pixelX),
      ConstHostView<T, 1>(pixelY), ConstHostView<T, 1>(pixelZ));
}

#endif




template struct SynthesisFactory<float>;

template struct SynthesisFactory<double>;

}  // namespace bipp
