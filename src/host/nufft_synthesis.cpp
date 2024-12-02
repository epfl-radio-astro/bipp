#include "host/nufft_synthesis.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstring>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>
#include <limits>
#include <neonufft/plan.hpp>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "host/blas_api.hpp"
#include "host/eigensolver.hpp"
#include "host/gram_matrix.hpp"
#include "host/kernels/nuft_sum.hpp"
#include "host/virtual_vis.hpp"
#include "memory/copy.hpp"
#include "memory/array.hpp"
#include "nufft_util.hpp"

namespace bipp {
namespace host {

namespace {
  template<typename T, std::size_t DIM>
  class DatasetReadArray;

  template <std::size_t DIM>
  class DatasetReadArray<float, DIM> {
  public:
    using ValueType = float;
    DatasetReadArray(std::shared_ptr<Allocator> alloc,
                 const typename HostArray<ValueType, DIM>::IndexType& shape)
        : array_(std::move(alloc), shape) {}

    auto read_view() -> HostView<float, DIM> { return array_; }

    auto process_view() -> HostView<ValueType, DIM> { return array_; }

  private:
    HostArray<ValueType, DIM> array_;
  };

  template <std::size_t DIM>
  class DatasetReadArray<double, DIM> {
  public:
    using ValueType = double;
    DatasetReadArray(std::shared_ptr<Allocator> alloc,
                 const typename HostArray<ValueType, DIM>::IndexType& shape)
        : readArray_(alloc, shape), array_(std::move(alloc), shape) {}

    auto read_view() -> HostView<float, DIM> { return readArray_; }

    auto process_view() -> HostView<ValueType, DIM> {
      copy(readArray_, array_);
      return array_;
    }

  private:
    HostArray<float, DIM> readArray_;
    HostArray<ValueType, DIM> array_;
  };

  template <std::size_t DIM>
  class DatasetReadArray<std::complex<float>, DIM> {
  public:
    using ValueType = std::complex<float>;
    DatasetReadArray(std::shared_ptr<Allocator> alloc,
                 const typename HostArray<ValueType, DIM>::IndexType& shape)
        : array_(std::move(alloc), shape) {}

    auto read_view() -> HostView<std::complex<float>, DIM> { return array_; }

    auto process_view() -> HostView<ValueType, DIM> { return array_; }

  private:
    HostArray<ValueType, DIM> array_;
  };

  template <std::size_t DIM>
  class DatasetReadArray<std::complex<double>, DIM> {
  public:
    using ValueType = std::complex<double>;
    DatasetReadArray(std::shared_ptr<Allocator> alloc,
                 const typename HostArray<ValueType, DIM>::IndexType& shape)
        : readArray_(alloc, shape), array_(std::move(alloc), shape) {}

    auto read_view() -> HostView<std::complex<float>, DIM> { return readArray_; }

    auto process_view() -> HostView<ValueType, DIM> {
      copy(readArray_, array_);
      return array_;
    }

  private:
    HostArray<std::complex<float>, DIM> readArray_;
    HostArray<ValueType, DIM> array_;
  };
}

template <typename T>
void nufft_synthesis(const Communicator& comm, ContextInternal& ctx,
                     const NufftSynthesisOptions& opt, DatasetFileReader& datasetReader,
                     ConstHostView<std::pair<std::size_t, const float*>, 1> samples,
                     ConstHostView<float, 1> pixelX, ConstHostView<float, 1> pixelY,
                     ConstHostView<float, 1> pixelZ, const std::string& imageTag,
                     ImageFileWriter& imageWriter) {
  const auto numPixel = pixelX.size();
  assert(pixelY.size() == numPixel);
  assert(pixelZ.size() == numPixel);

  std::array<std::array<T, 2>, 3> input_min_max;
  std::array<std::array<T, 2>, 3> output_min_max;

  auto outMM = std::minmax_element(pixelX.data(), pixelX.data() + pixelX.size());
  output_min_max[0][0] = *(outMM.first);
  output_min_max[0][1] = *(outMM.second);

  outMM = std::minmax_element(pixelY.data(), pixelY.data() + pixelY.size());
  output_min_max[1][0] = *(outMM.first);
  output_min_max[1][1] = *(outMM.second);

  outMM = std::minmax_element(pixelZ.data(), pixelZ.data() + pixelZ.size());
  output_min_max[2][0] = *(outMM.first);
  output_min_max[2][1] = *(outMM.second);

  for(std::size_t dim = 0; dim < 3; ++dim) {
    input_min_max[dim][0] = std::numeric_limits<T>::max();
    input_min_max[dim][1] = std::numeric_limits<T>::lowest();
  }

  for (std::size_t i = 0; i < samples.size(); ++i) {
      const auto id = samples[i].first;
      input_min_max[0][0] = std::min<T>(input_min_max[0][0], datasetReader.read_u_min(id));
      input_min_max[0][1] = std::max<T>(input_min_max[0][1], datasetReader.read_u_max(id));

      input_min_max[1][0] = std::min<T>(input_min_max[1][0], datasetReader.read_v_min(id));
      input_min_max[1][1] = std::max<T>(input_min_max[1][1], datasetReader.read_v_max(id));

      input_min_max[2][0] = std::min<T>(input_min_max[2][0], datasetReader.read_w_min(id));
      input_min_max[2][1] = std::max<T>(input_min_max[2][1], datasetReader.read_w_max(id));
  }


  const auto nBeam = datasetReader.num_beam();
  const auto nAntenna = datasetReader.num_antenna();

  neonufft::PlanT3<T, 3> plan(neonufft::Options(), 1, input_min_max, output_min_max);
  if constexpr (std::is_same_v<T, float>) {
    plan.set_output_points(numPixel, {pixelX.data(), pixelY.data(), pixelZ.data()});
  } else {
    HostArray<T, 1> pixelXDouble(ctx.host_alloc(), pixelX.size());
    HostArray<T, 1> pixelYDouble(ctx.host_alloc(), pixelY.size());
    HostArray<T, 1> pixelZDouble(ctx.host_alloc(), pixelZ.size());
    copy(pixelX, pixelXDouble);
    copy(pixelY, pixelYDouble);
    copy(pixelZ, pixelZDouble);
    plan.set_output_points(numPixel,
                           {pixelXDouble.data(), pixelYDouble.data(), pixelZDouble.data()});
  }

  {
    DatasetReadArray<T, 2> uvwArray(ctx.host_alloc(), {nAntenna * nAntenna, 3});
    HostArray<std::complex<T>, 1> virtualVisArray(ctx.host_alloc(), nAntenna * nAntenna);
    DatasetReadArray<std::complex<T>, 2> vArray(ctx.host_alloc(), {nAntenna, nBeam});

    HostArray<T, 1> dScaledArray(ctx.host_alloc(), nBeam);


    for (std::size_t i = 0; i < samples.size(); ++i) {
      const auto id = samples[i].first;

      ConstHostView<float, 1> d(samples[i].second, nBeam, 1);

      if (opt.normalizeImageNvis) {
        const auto nVis = datasetReader.read_n_vis(id);
        const T scale = nVis ? 1 / T(nVis) : 0;
        for (std::size_t j = 0; j < nBeam; ++j) {
          dScaledArray[j] = scale * d[j];
        }
      } else {
        copy(d, dScaledArray);
      }

      datasetReader.read_uvw(id, uvwArray.read_view());
      auto uvw = uvwArray.process_view();

      datasetReader.read_eig_vec(id, vArray.read_view());
      auto v = vArray.process_view();

      virtual_vis<T>(ctx, dScaledArray, v, virtualVisArray);

      // TODO add samples
      plan.set_input_points(uvw.shape(0), {&uvw[{0, 0}], &uvw[{0, 1}], &uvw[{0, 2}]});
      plan.add_input(virtualVisArray.data());
    }
  }

  // TODO: write image
  HostArray<std::complex<T>, 1> image(ctx.host_alloc(), numPixel);
  plan.transform(image.data());

  HostArray<float, 1> imageReal(ctx.host_alloc(), numPixel);

  if(opt.normalizeImage) {
    const T scale = T(1) / T(datasetReader.num_samples());
    for (std::size_t j = 0; j < numPixel; ++j) {
      imageReal[j] = scale* image[j].real();
    }
  } else {
    for (std::size_t j = 0; j < numPixel; ++j) {
      imageReal[j] = image[j].real();
    }
  }

  imageWriter.write(imageTag, imageReal);
}

template void nufft_synthesis<float>(const Communicator& comm, ContextInternal& ctx,
                                     const NufftSynthesisOptions& opt,
                                     DatasetFileReader& datasetReader,
                                     ConstHostView<std::pair<std::size_t, const float*>, 1> samples,
                                     ConstHostView<float, 1> pixelX, ConstHostView<float, 1> pixelY,
                                     ConstHostView<float, 1> pixelZ, const std::string& imageTag,
                                     ImageFileWriter& imageWriter);

template void nufft_synthesis<double>(
    const Communicator& comm, ContextInternal& ctx, const NufftSynthesisOptions& opt,
    DatasetFileReader& datasetReader,
    ConstHostView<std::pair<std::size_t, const float*>, 1> samples, ConstHostView<float, 1> pixelX,
    ConstHostView<float, 1> pixelY, ConstHostView<float, 1> pixelZ, const std::string& imageTag,
    ImageFileWriter& imageWriter);

/*
static auto system_memory() -> unsigned long long {
  unsigned long long pages = sysconf(_SC_PHYS_PAGES);
  unsigned long long pageSize = sysconf(_SC_PAGE_SIZE);
  unsigned long long memory = pages * pageSize;
  return memory > 0 ? memory : 8ull * 1024ull * 1024ull * 1024ull;
}

template <typename T>
NufftSynthesis<T>::NufftSynthesis(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
                                  std::size_t nImages, ConstHostView<T, 1> pixelX,
                                  ConstHostView<T, 1> pixelY, ConstHostView<T, 1> pixelZ)
    : ctx_(std::move(ctx)),
      opt_(std::move(opt)),
      nImages_(nImages),
      nPixel_(pixelX.shape()),
      pixel_(ctx_->host_alloc(), {pixelX.size(), 3}),
      img_(ctx_->host_alloc(), {nPixel_, nImages_}),
      imgPartition_(DomainPartition::none(ctx_, nPixel_)),
      totalCollectCount_(0),
      totalVisibilityCount_(0) {
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());

  img_.zero();

  // Only partition image if explicitly set. Auto defaults to no partition.
  std::visit(
      [&](auto&& arg) -> void {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "image partition: grid ({}, {}, {})",
                             arg.dimensions[0], arg.dimensions[1], arg.dimensions[2]);
          imgPartition_ =
              DomainPartition::grid<T, 3>(ctx_, arg.dimensions, {pixelX, pixelY, pixelZ});
        } else if constexpr (std::is_same_v<ArgType, Partition::None> ||
                             std::is_same_v<ArgType, Partition::Auto>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "image partition: none");
        }
      },
      opt_.localImagePartition.method);

  imgPartition_.apply(pixelX, pixel_.slice_view(0));
  imgPartition_.apply(pixelY, pixel_.slice_view(1));
  imgPartition_.apply(pixelZ, pixel_.slice_view(2));
}

template <typename T>
auto NufftSynthesis<T>::process(CollectorInterface<T>& collector) -> void {
  ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "computing nufft for collected data");

  auto data = collector.get_data();
  if (data.empty()) return;

  std::size_t collectPoints = 0;
  std::size_t visibilityCount = 0;
  for (const auto& s : data) {
    collectPoints += s.xyzUvw.shape(0);
    visibilityCount += s.nVis;
    assert(s.v.shape(0) * s.v.shape(0) == s.xyzUvw.shape(0));
  }

  // compute virtual visibilities
  HostArray<std::complex<T>, 2> virtualVis(ctx_->host_alloc(), {collectPoints, nImages_});
  {
    std::size_t currentCount = 0;
    for (const auto& s : data) {
      const auto nAntenna = s.v.shape(0);
      auto virtVisCurrent =
          virtualVis.sub_view({currentCount, 0}, {nAntenna * nAntenna, virtualVis.shape(1)});
      virtual_vis<T>(*ctx_, s.nVis, s.dMasked, ConstHostView<std::complex<T>, 2>(s.v),
virtVisCurrent); currentCount += s.xyzUvw.shape(0);
    }
  }

  // copy uvw into contiguous buffer
  HostArray<T, 2> uvw(ctx_->host_alloc(), {collectPoints, 3});
  {
    std::size_t currentCount = 0;
    for (const auto& s : data) {
      copy(ConstHostView<T, 2>(s.xyzUvw), uvw.sub_view({currentCount, 0}, {s.xyzUvw.shape(0), 3}));
      currentCount += s.xyzUvw.shape(0);
    }
  }

  auto uvwX = uvw.slice_view(0).sub_view({0}, {collectPoints});
  auto uvwY = uvw.slice_view(1).sub_view({0}, {collectPoints});
  auto uvwZ = uvw.slice_view(2).sub_view({0}, {collectPoints});


  auto pixelX = pixel_.slice_view(0);
  auto pixelY = pixel_.slice_view(1);
  auto pixelZ = pixel_.slice_view(2);

  auto inputPartition = std::visit(
      [&](auto&& arg) -> DomainPartition {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: grid ({}, {}, {})",
                             arg.dimensions[0], arg.dimensions[1], arg.dimensions[2]);
          return DomainPartition::grid<T, 3>(ctx_, arg.dimensions, {uvwX, uvwY, uvwZ});
        } else if constexpr (std::is_same_v<ArgType, Partition::None>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: none");
          return DomainPartition::none(ctx_, collectPoints);

        } else if constexpr (std::is_same_v<ArgType, Partition::Auto>) {
          std::array<double, 3> uvwExtent{};
          std::array<double, 3> imgExtent{};

          auto minMaxIt = std::minmax_element(uvwX.data(), uvwX.data() + collectPoints);
          uvwExtent[0] = *minMaxIt.second - *minMaxIt.first;
          minMaxIt = std::minmax_element(uvwY.data(), uvwY.data() + collectPoints);
          uvwExtent[1] = *minMaxIt.second - *minMaxIt.first;
          minMaxIt = std::minmax_element(uvwZ.data(), uvwZ.data() + collectPoints);
          uvwExtent[2] = *minMaxIt.second - *minMaxIt.first;

          minMaxIt = std::minmax_element(pixelX.data(), pixelX.data() + nPixel_);
          imgExtent[0] = *minMaxIt.second - *minMaxIt.first;
          minMaxIt = std::minmax_element(pixelY.data(), pixelY.data() + nPixel_);
          imgExtent[1] = *minMaxIt.second - *minMaxIt.first;
          minMaxIt = std::minmax_element(pixelZ.data(), pixelZ.data() + nPixel_);
          imgExtent[2] = *minMaxIt.second - *minMaxIt.first;

          // Use at most 12.5% of total memory for fft grid
          const auto gridSize = optimal_nufft_input_partition(
              uvwExtent, imgExtent, system_memory() / (8 * sizeof(std::complex<T>)));

          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: grid ({}, {}, {})", gridSize[0],
                             gridSize[1], gridSize[2]);

          // set partition method to grid and create grid partition
          opt_.localUVWPartition.method = Partition::Grid{gridSize};
          return DomainPartition::grid<T>(ctx_, gridSize, {uvwX, uvwY, uvwZ});
        }
      },
      opt_.localUVWPartition.method);

  for (std::size_t j = 0; j < nImages_; ++j) {
    inputPartition.apply(virtualVis.slice_view(j));
  }

  inputPartition.apply(uvwX);
  inputPartition.apply(uvwY);
  inputPartition.apply(uvwZ);
  auto output = HostArray<std::complex<T>, 1>(ctx_->host_alloc(), nPixel_);

  for (const auto& [inputBegin, inputSize] : inputPartition.groups()) {
    if (!inputSize) continue;
    auto uvwXSlice = uvwX.sub_view(inputBegin, inputSize);
    auto uvwYSlice = uvwY.sub_view(inputBegin, inputSize);
    auto uvwZSlice = uvwZ.sub_view(inputBegin, inputSize);

    if (inputSize <= 32) {
      // Direct evaluation of sum for small input sizes
      ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "nufft size {}, direct evaluation", inputSize);
      for (std::size_t j = 0; j < nImages_; ++j) {
        auto virtVisCurrentSlice =
            virtualVis.slice_view(j).sub_view(inputBegin, inputSize);
        auto* imgPtr = &img_[{0, j}];
        nuft_sum<T>(1.0, inputSize, virtVisCurrentSlice.data(), uvwXSlice.data(), uvwYSlice.data(),
                    uvwZSlice.data(), img_.shape(0), pixelX.data(), pixelY.data(), pixelZ.data(),
                    imgPtr);
      }
    } else {
      // Compute Nufft for each input and output partition combination
      for (const auto& [imgBegin, imgSize] : imgPartition_.groups()) {
        if (!imgSize) continue;

        auto pixelXSlice = pixelX.sub_view(imgBegin, imgSize);
        auto pixelYSlice = pixelY.sub_view(imgBegin, imgSize);
        auto pixelZSlice = pixelZ.sub_view(imgBegin, imgSize);

        ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT input coordinate x", uvwXSlice);
        ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT input coordinate y", uvwYSlice);
        ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT input coordinate z", uvwZSlice);
        ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT output coordinate x", pixelXSlice);
        ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT output coordinate y", pixelYSlice);
        ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT output coordinate z", pixelZSlice);

        // Approximate sum through nufft
        ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "nufft size {}, calling fiNUFFT", inputSize);
        Nufft3d3<T> transform(1, opt_.tolerance, 1, inputSize, uvwXSlice.data(), uvwYSlice.data(),
                              uvwZSlice.data(), imgSize, pixelXSlice.data(), pixelYSlice.data(),
                              pixelZSlice.data());

          for (std::size_t j = 0; j < nImages_; ++j) {
            auto virtVisCurrentSlice =
                virtualVis.slice_view(j).sub_view(inputBegin, inputSize);
            ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT input", virtVisCurrentSlice);
            transform.execute(virtVisCurrentSlice.data(), output.data());
            ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT output",
                                      output.sub_view({0}, {imgSize}));

            auto* __restrict__ outputPtr = output.data();
            auto* __restrict__ imgPtr = &img_[{imgBegin, j}];
            for (std::size_t k = 0; k < imgSize; ++k) {
              imgPtr[k] += outputPtr[k].real();
            }
          }
      }
    }
  }
  totalCollectCount_ += data.size();
  totalVisibilityCount_ += visibilityCount;
}

template <typename T>
auto NufftSynthesis<T>::get(View<T, 2> out) -> void {
  assert(out.shape(0) == nPixel_);
  assert(out.shape(1) == nImages_);

  HostView<T, 2> outHost(out);

  const T scale =
      totalCollectCount_ ? static_cast<T>(1.0 / static_cast<double>(totalCollectCount_)) : 0;

  ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                     "NufftSynthesis<T>::get totalVisibilityCount_ = {}, totalCollectCount_ = {},
scale = {}", totalVisibilityCount_, totalCollectCount_, scale);

  for (std::size_t i = 0; i < nImages_; ++i) {
    auto currentImg = img_.slice_view(i);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image permuted", currentImg);

    imgPartition_.reverse<T>(currentImg, outHost.slice_view(i));

    if (opt_.normalizeImage) {
      T* __restrict__ outPtr = &outHost[{0, i}];
      for (std::size_t j = 0; j < nPixel_; ++j) {
          outPtr[j] *= scale;
      }
    }

    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image output", outHost.slice_view(i));
  }
}

template class NufftSynthesis<float>;
template class NufftSynthesis<double>;

*/
}  // namespace host
}  // namespace bipp
