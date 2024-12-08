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
#include "host/virtual_vis.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "nufft_util.hpp"

namespace bipp {
namespace host {

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
void nufft_synthesis(ContextInternal& ctx, const NufftSynthesisOptions& opt, Dataset& dataset,
                     ConstHostView<std::pair<std::size_t, const float*>, 1> samples,
                     ConstHostView<float, 1> pixelX, ConstHostView<float, 1> pixelY,
                     ConstHostView<float, 1> pixelZ, const std::string& imageTag,
                     ImageFileWriter& imageWriter) {
  auto funcTimer = ctx.logger().scoped_timing(BIPP_LOG_LEVEL_INFO, "host::nufft_synthesis");

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

  ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "output extent");
  auto outMM = std::minmax_element(pixelX.data(), pixelX.data() + pixelX.size());

  std::array<T, 3> output_min, output_max;
  output_min[0] = *(outMM.first);
  output_max[0] = *(outMM.second);

  outMM = std::minmax_element(pixelY.data(), pixelY.data() + pixelY.size());
  output_min[1] = *(outMM.first);
  output_max[1] = *(outMM.second);

  outMM = std::minmax_element(pixelZ.data(), pixelZ.data() + pixelZ.size());
  output_min[2] = *(outMM.first);
  output_max[2] = *(outMM.second);

  ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "output extent");

  ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "output \"l\" min {}, max {}", output_min[0],
                   output_max[0]);
  ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "output \"m\" min {}, max {}", output_min[1],
                   output_max[1]);
  ctx.logger().log(BIPP_LOG_LEVEL_DEBUG, "output \"n\" min {}, max {}", output_min[2],
                   output_max[2]);

  const auto nBeam = dataset.num_beam();
  const auto nAntenna = dataset.num_antenna();

  ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "nufft: create plan");
  auto neo_opt = neonufft::Options();
  neo_opt.sort_input = false;
  neo_opt.sort_output = false;
  neonufft::PlanT3<T, 3> plan(neo_opt, 1, input_min, input_max, output_min, output_max);
  ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "nufft: create plan");

  ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "nufft: set image points");
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
  ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "nufft: set image points");

  {
    auto loopTimer = ctx.logger().scoped_timing(BIPP_LOG_LEVEL_INFO, "loop");
    HostArray<T, 2> uvwArray(ctx.host_alloc(), {nAntenna * nAntenna, 3});
    HostArray<std::complex<T>, 1> virtualVisArray(ctx.host_alloc(), nAntenna * nAntenna);
    HostArray<std::complex<T>, 2> vArray(ctx.host_alloc(), {nAntenna, nBeam});

    HostArray<T, 1> dScaledArray(ctx.host_alloc(), nBeam);

    for (std::size_t i = 0; i < samples.size(); ++i) {
      const auto id = samples[i].first;

      ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "scale eigenvalues");
      ConstHostView<float, 1> d(samples[i].second, nBeam, 1);

      if (opt.normalizeImageNvis) {
        const auto nVis = dataset.n_vis(id);
        const T scale = nVis ? 1 / T(nVis) : 0;
        for (std::size_t j = 0; j < nBeam; ++j) {
          dScaledArray[j] = scale * d[j];
        }
      } else {
        copy(d, dScaledArray);
      }
      ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "scale eigenvalues");

      ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "read dataset");
      read_uvw(ctx, dataset, id, uvwArray);

      read_eig_vec(ctx, dataset, id, vArray);
      ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "read dataset");

      ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvalues", d);
      ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "scaled eigenvalues", dScaledArray);
      ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "eigenvectors", vArray);
      ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "u", uvwArray.slice_view(0));
      ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "v", uvwArray.slice_view(1));
      ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "w", uvwArray.slice_view(2));

      ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "compute virtual vis");
      virtual_vis<T>(ctx, dScaledArray, vArray, virtualVisArray);
      ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "compute virtual vis");

      ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "virutal vis", virtualVisArray);

      ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "nufft: set input points");
      plan.set_input_points(uvwArray.shape(0),
                            {&uvwArray[{0, 0}], &uvwArray[{0, 1}], &uvwArray[{0, 2}]});
      ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "nufft: set input points");

      ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "nufft: add data");
      plan.add_input(virtualVisArray.data());
      ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "nufft: add data");
    }
  }

  ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "nufft: transform");
  HostArray<std::complex<T>, 1> image(ctx.host_alloc(), numPixel);
  plan.transform(image.data());
  ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "nufft: transform");
  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft output", image);

  HostArray<float, 1> imageReal(ctx.host_alloc(), numPixel);

  ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "extract image");
  if (opt.normalizeImage) {
    const T scale = T(1) / T(dataset.num_samples());
    for (std::size_t j = 0; j < numPixel; ++j) {
      imageReal[j] = scale * image[j].real();
    }
  } else {
    for (std::size_t j = 0; j < numPixel; ++j) {
      imageReal[j] = image[j].real();
    }
  }
  ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "extract image");

  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "scaled image", imageReal);

  ctx.logger().start_timing(BIPP_LOG_LEVEL_INFO, "write image");
  imageWriter.write(imageTag, imageReal);
  ctx.logger().stop_timing(BIPP_LOG_LEVEL_INFO, "write image");
}

template void nufft_synthesis<float>(ContextInternal& ctx, const NufftSynthesisOptions& opt,
                                     Dataset& dataset,
                                     ConstHostView<std::pair<std::size_t, const float*>, 1> samples,
                                     ConstHostView<float, 1> pixelX, ConstHostView<float, 1> pixelY,
                                     ConstHostView<float, 1> pixelZ, const std::string& imageTag,
                                     ImageFileWriter& imageWriter);

template void nufft_synthesis<double>(
    ContextInternal& ctx, const NufftSynthesisOptions& opt, Dataset& dataset,
    ConstHostView<std::pair<std::size_t, const float*>, 1> samples, ConstHostView<float, 1> pixelX,
    ConstHostView<float, 1> pixelY, ConstHostView<float, 1> pixelZ, const std::string& imageTag,
    ImageFileWriter& imageWriter);

}  // namespace host
}  // namespace bipp
