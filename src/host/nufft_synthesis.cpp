#include "host/nufft_synthesis.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstring>
#include <limits>
#include <memory>
#include <neonufft/plan.hpp>
#include <type_traits>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "host/domain_partition.hpp"
#include "host/nufft.hpp"
#include "host/uvw_partition.hpp"
#include "host/virtual_vis.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "nufft_util.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/nufft.hpp"
#endif

namespace bipp {
namespace host {

static auto system_memory() -> unsigned long long {
  unsigned long long pages = sysconf(_SC_PHYS_PAGES);
  unsigned long long pageSize = sysconf(_SC_PAGE_SIZE);
  unsigned long long memory = pages * pageSize;
  return memory > 0 ? memory : 8ull * 1024ull * 1024ull * 1024ull;
}

auto read_eig_val(ContextInternal& ctx, Dataset& dataset, std::size_t index, HostView<float, 1> d) {
  dataset.eig_val(index, d.data());
}

auto read_eig_val(ContextInternal& ctx, Dataset& dataset, std::size_t index,
                  HostView<double, 1> d) {
  HostArray<float, 1> buffer(ctx.host_alloc(), d.shape());
  dataset.eig_val(index, buffer.data());
  copy(buffer, d);
}

auto read_eig_vec(ContextInternal& ctx, Dataset& dataset, std::size_t index,
                  HostView<std::complex<float>, 2> v) {
  dataset.eig_vec(index, v.data(), v.strides(1));
}

auto read_eig_vec(ContextInternal& ctx, Dataset& dataset, std::size_t index,
                  HostView<std::complex<double>, 2> v) {
  HostArray<std::complex<float>, 2> buffer(ctx.host_alloc(), v.shape());
  dataset.eig_vec(index, buffer.data(), buffer.strides(1));
  copy(buffer, v);
}

auto read_uvw(ContextInternal& ctx, Dataset& dataset, std::size_t index, HostView<float, 2> uvw) {
  dataset.uvw(index, uvw.data(), uvw.strides(1));
}

auto read_uvw(ContextInternal& ctx, Dataset& dataset, std::size_t index, HostView<double, 2> uvw) {
  HostArray<float, 2> buffer(ctx.host_alloc(), uvw.shape());
  dataset.uvw(index, buffer.data(), buffer.strides(1));
  copy(buffer, uvw);
}

template <typename T>
void nufft_synthesis(std::shared_ptr<ContextInternal> ctxPtr, const NufftSynthesisOptions& opt,
                     Dataset& dataset,
                     ConstHostView<std::pair<std::size_t, const float*>, 1> samples,
                     ConstHostView<float, 1> pixelX, ConstHostView<float, 1> pixelY,
                     ConstHostView<float, 1> pixelZ, const std::string& imageTag,
                     Image& imageWriter) {
  auto& ctx = *ctxPtr;
  auto funcTimer = ctx.logger().scoped_timing(BIPP_LOG_LEVEL_INFO, "host::nufft_synthesis");
  ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "prepare");

  const auto numPixel = pixelX.size();
  assert(pixelY.size() == numPixel);
  assert(pixelZ.size() == numPixel);

  ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "input extent");

  std::array<T, 3> input_min_float, input_max_float;
  input_min_float.fill(std::numeric_limits<float>::max());
  input_max_float.fill(std::numeric_limits<float>::lowest());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto id = samples[i].first;

    std::array<float, 3> sample_input_max, sample_input_min;

    dataset.uvw_min_max(id, sample_input_min.data(), sample_input_max.data());

    input_min_float[0] = std::min<float>(input_min_float[0], sample_input_min[0]);
    input_min_float[1] = std::min<float>(input_min_float[1], sample_input_min[1]);
    input_min_float[2] = std::min<float>(input_min_float[2], sample_input_min[2]);

    input_max_float[0] = std::max<float>(input_max_float[0], sample_input_max[0]);
    input_max_float[1] = std::max<float>(input_max_float[1], sample_input_max[1]);
    input_max_float[2] = std::max<float>(input_max_float[2], sample_input_max[2]);
  }
  std::array<T, 3> input_min, input_max;
  std::copy(input_min_float.begin(), input_min_float.end(), input_min.begin());
  std::copy(input_max_float.begin(), input_max_float.end(), input_max.begin());
  ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "input extent");

  ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "input \"u\" min {}, max {}", input_min[0], input_max[0]);
  ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "input \"v\" min {}, max {}", input_min[1], input_max[1]);
  ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "input \"w\" min {}, max {}", input_min[2], input_max[2]);

  auto inputGroups = std::visit(
      [&](auto&& arg) -> std::vector<UVWGroup<T>> {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
          ctx.logger().log(BIPP_LOG_LEVEL_INFO, "input partition: grid ({}, {}, {})",
                           arg.dimensions[0], arg.dimensions[1], arg.dimensions[2]);
          return create_uvw_partitions(arg.dimensions, input_min, input_max);
        } else if constexpr (std::is_same_v<ArgType, Partition::None> ||
                             std::is_same_v<ArgType, Partition::Auto>) {
          ctx.logger().log(BIPP_LOG_LEVEL_INFO, "input partition: none");
          return create_uvw_partitions({1, 1, 1}, input_min, input_max);
        }
      },
      opt.localUVWPartition.method);

  const auto nBeam = dataset.num_beam();
  const auto nAntenna = dataset.num_antenna();

  ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "image partition");
  DomainPartition imagePartition = std::visit(
      [&](auto&& arg) -> DomainPartition {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
          ctx.logger().log(BIPP_LOG_LEVEL_INFO, "image partition: grid ({}, {}, {})",
                           arg.dimensions[0], arg.dimensions[1], arg.dimensions[2]);
          return DomainPartition::grid<float, 3>(ctx.host_alloc(), arg.dimensions,
                                                 {pixelX, pixelY, pixelZ});
        } else if constexpr (std::is_same_v<ArgType, Partition::None> ||
                             std::is_same_v<ArgType, Partition::Auto>) {
          ctx.logger().log(BIPP_LOG_LEVEL_INFO, "image partition: none");
          return DomainPartition::none(ctx.host_alloc(), numPixel);
        }
      },
      opt.localImagePartition.method);

  HostArray<T, 1> pixelXPermuted(ctx.host_alloc(), numPixel);
  HostArray<T, 1> pixelYPermuted(ctx.host_alloc(), numPixel);
  HostArray<T, 1> pixelZPermuted(ctx.host_alloc(), numPixel);

  imagePartition.apply(pixelX, pixelXPermuted);
  imagePartition.apply(pixelY, pixelYPermuted);
  imagePartition.apply(pixelZ, pixelZPermuted);
  ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "image partition");

  HostArray<T, 2> uvwArray(ctx.host_alloc(), {nAntenna * nAntenna, 3});
  HostArray<std::complex<T>, 1> virtualVisArray(ctx.host_alloc(), nAntenna * nAntenna);
  HostArray<std::complex<T>, 2> vArray(ctx.host_alloc(), {nAntenna, nBeam});
  HostArray<T, 1> dScaledArray(ctx.host_alloc(), nBeam);

  auto neo_opt = neonufft::Options();
  neo_opt.sort_input = false;
  neo_opt.sort_output = false;

  // const auto grid_memory_size = neonufft::PlanT3<T, 3>::grid_memory_size(
  //     neo_opt, input_min, input_max, output_min, output_max);

  // printf("grid_memory_size = %llu\n", grid_memory_size);

  HostArray<float, 1> imageReal(ctx.host_alloc(), numPixel);
  imageReal.zero();

  ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "prepare");

  ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "nufft");
  for (const auto& [imgBegin, imgSize] : imagePartition.groups()) {
    if (!imgSize) continue;

    ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "output extent");
    auto outMM = std::minmax_element(pixelXPermuted.data() + imgBegin,
                                     pixelXPermuted.data() + imgBegin + imgSize);
    std::array<T, 3> output_min, output_max;
    output_min[0] = *(outMM.first);
    output_max[0] = *(outMM.second);

    outMM = std::minmax_element(pixelYPermuted.data() + imgBegin,
                                pixelYPermuted.data() + imgBegin + imgSize);
    output_min[1] = *(outMM.first);
    output_max[1] = *(outMM.second);

    outMM = std::minmax_element(pixelZPermuted.data() + imgBegin,
                                pixelZPermuted.data() + imgBegin + imgSize);
    output_min[2] = *(outMM.first);
    output_max[2] = *(outMM.second);
    ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "output extent");

    ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "sample loop");
    for (const auto& inGroup : inputGroups) {
      ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "output \"l\" min {}, max {}", output_min[0],
                       output_max[0]);
      ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "output \"m\" min {}, max {}", output_min[1],
                       output_max[1]);
      ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "output \"n\" min {}, max {}", output_min[2],
                       output_max[2]);
      std::unique_ptr<NUFFTInterface<T>> nufft;

      if(ctx.processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
        ctx.gpu_queue().sync();  // make sure previous memory is freed
        nufft.reset(new gpu::NUFFT<T>(ctxPtr, 40, nAntenna * nAntenna, neo_opt, 1,
                                       inGroup.lowerBoundInclusive, inGroup.upperBoundExclusive,
                                       output_min, output_max));
#else
    throw GPUSupportError();
#endif
      } else {
        nufft.reset(new host::NUFFT<T>(ctxPtr, 10, nAntenna * nAntenna, neo_opt, 1,
                                       inGroup.lowerBoundInclusive, inGroup.upperBoundExclusive,
                                       output_min, output_max));
      }

      nufft->set_output_points(pixelXPermuted.sub_view(imgBegin, imgSize),
                               pixelYPermuted.sub_view(imgBegin, imgSize),
                               pixelZPermuted.sub_view(imgBegin, imgSize));

      for (std::size_t i = 0; i < samples.size(); ++i) {
        ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "read dataset");
        const auto id = samples[i].first;

        ConstHostView<float, 1> d(samples[i].second, nBeam, 1);

        if (opt.apply_scaling) {
          const T scale = dataset.scale(id);
          ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "applying scale {}", scale);
          for (std::size_t j = 0; j < nBeam; ++j) {
            dScaledArray[j] = scale * d[j];
          }
        } else {
          copy(d, dScaledArray);
        }

        read_uvw(ctx, dataset, id, uvwArray);

        read_eig_vec(ctx, dataset, id, vArray);
        ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "read dataset");

        ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvalues", d);
        ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "scaled eigenvalues", dScaledArray);
        ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", vArray);
        ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "u", uvwArray.slice_view(0));
        ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "v", uvwArray.slice_view(1));
        ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "w", uvwArray.slice_view(2));

        ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "virtual vis");
        virtual_vis<T>(ctx, dScaledArray, vArray, virtualVisArray);
        ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "virtual vis");

        ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "partition uvw");
        auto [uvwView, virtualVisView] =
            inputGroups.size() > 1 ? apply_uvw_partition(inGroup, uvwArray, virtualVisArray)
                                   : std::pair<HostView<T, 2>, HostView<std::complex<T>, 1>>(
                                         uvwArray, virtualVisArray);
        ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "partition uvw");

        if (virtualVisView.size()) {
          ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "virutal vis", virtualVisView);
          nufft->add_input(uvwView, virtualVisView);
        }
      }

      nufft->transform_and_add(imageReal.sub_view(imgBegin, imgSize));
      ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "sample loop");
    }
  }
  ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "nufft");

  if (opt.normalizeImage) {
    ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "scale image");
    const T scale = T(1) / T(dataset.num_samples());
    for (std::size_t j = 0; j < imageReal.size(); ++j) {
      imageReal[j] *= scale;
    }
    ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "scale image");
  }

  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "scaled image", imageReal);

  ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "permut image");
  imagePartition.reverse(imageReal);
  ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "permut image");

  ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "write image");
  imageWriter.set(imageTag, imageReal.data());
  ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "write image");
}

template void nufft_synthesis<float>(std::shared_ptr<ContextInternal> ctxPtr,
                                     const NufftSynthesisOptions& opt, Dataset& dataset,
                                     ConstHostView<std::pair<std::size_t, const float*>, 1> samples,
                                     ConstHostView<float, 1> pixelX, ConstHostView<float, 1> pixelY,
                                     ConstHostView<float, 1> pixelZ, const std::string& imageTag,
                                     Image& imageWriter);

template void nufft_synthesis<double>(
    std::shared_ptr<ContextInternal> ctxPtr, const NufftSynthesisOptions& opt, Dataset& dataset,
    ConstHostView<std::pair<std::size_t, const float*>, 1> samples, ConstHostView<float, 1> pixelX,
    ConstHostView<float, 1> pixelY, ConstHostView<float, 1> pixelZ, const std::string& imageTag,
    Image& imageWriter);

}  // namespace host
}  // namespace bipp
