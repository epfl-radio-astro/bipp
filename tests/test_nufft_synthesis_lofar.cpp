#include <complex>
#include <fstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bipp/bipp.hpp"
#include "bipp/communicator.hpp"
#include "bipp/image_synthesis.hpp"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

static auto get_lofar_input_json() -> const nlohmann::json& {
  static nlohmann::json data = []() {
    std::ifstream file(std::string(BIPP_TEST_DATA_DIR) + "/lofar_input.json");
    nlohmann::json j;
    file >> j;
    return j;
  }();

  return data;
}

template <typename T>
static auto get_lofar_nufft_output() -> const nlohmann::json& {
  static nlohmann::json data = []() {
    if constexpr (std::is_same_v<T, float>) {
      std::ifstream file(std::string(BIPP_TEST_DATA_DIR) + "/lofar_nufft_output_single.json");
      nlohmann::json j;
      file >> j;
      return j;
    } else {
      std::ifstream file(std::string(BIPP_TEST_DATA_DIR) + "/lofar_nufft_output_double.json");
      nlohmann::json j;
      file >> j;
      return j;
    }
  }();
  return data;
}

template <typename T, typename JSON>
static auto read_json_complex_2d(JSON jReal, JSON jImag) -> std::vector<std::complex<T>> {
  std::vector<std::complex<T>> w;

  auto wRealCol = jReal.begin();
  auto wImagCol = jImag.begin();
  for (; wRealCol != jReal.end(); ++wRealCol, ++wImagCol) {
    auto wReal = wRealCol->begin();
    auto wImag = wImagCol->begin();
    for (; wReal != wRealCol->end(); ++wReal, ++wImag) {
      w.emplace_back(*wReal, *wImag);
    }
  }
  return w;
}

template <typename T, typename JSON>
static auto read_json_scalar_2d(JSON j) -> std::vector<T> {
  std::vector<T> w;

  for (auto& col : j) {
    for (auto& val : col) {
      w.emplace_back(val);
    }
  }

  return w;
}

template <typename T, typename JSON>
static auto read_json_scalar_1d(JSON j) -> std::vector<T> {
  std::vector<T> vec;

  for (auto& val : j) {
    vec.emplace_back(val);
  }

  return vec;
}

template <typename T>
class NufftSynthesisLofar : public ::testing::TestWithParam<std::tuple<BippProcessingUnit>> {
protected:
  using ValueType = float;

  NufftSynthesisLofar() : ctx_(std::get<0>(GetParam())) {}

  auto test_intensity() -> void {
    const auto data = get_lofar_input_json();
    const auto output_data = get_lofar_nufft_output<T>();

    const ValueType wl = ValueType(data["wl"]);
    const ValueType tol = ValueType(data["eps"]);
    const std::size_t nAntenna = data["n_antenna"];
    const std::size_t nBeam = data["n_beam"];
    const std::size_t nEig = data["n_eig_int"];
    const std::size_t nIntervals = data["intervals_int"].size();
    const auto intervals = read_json_scalar_2d<ValueType>(data["intervals_int"]);

    const auto imgRef = read_json_scalar_2d<ValueType>(output_data[std::string("int_") + "lsq"]);
    const auto lmnX = read_json_scalar_1d<ValueType>(output_data["lmn_x"]);
    const auto lmnY = read_json_scalar_1d<ValueType>(output_data["lmn_y"]);
    const auto lmnZ = read_json_scalar_1d<ValueType>(output_data["lmn_z"]);

    const std::size_t nPixel = imgRef.size() / nIntervals;

    // map intervals to mask
    auto eigMaskFunc = [&](std::size_t idxBin, std::size_t nEigOut, ValueType* d) -> void {
      const ValueType dMin = intervals[idxBin * 2];
      const ValueType dMax = intervals[idxBin * 2 + 1];

      std::size_t idxEig = 0;
      for(; idxEig < nEig; ++idxEig) {
        const auto val = d[idxEig];
        d[idxEig] *= (val >= dMin && val <= dMax);
      }
      for(; idxEig < nEigOut ; ++idxEig) {
        d[idxEig] = 0;
      }
    };


    // create dataset
    const std::string datasetFileName = "test_nufft_synthesis_lofar.h5";
    auto dataset = bipp::DatasetFile::create(datasetFileName, "", nAntenna, nBeam);

    std::vector<ValueType> eigValues(nBeam);
    std::vector<std::complex<ValueType>> eigVec(nBeam * nAntenna);
    for (const auto& itData : data["data"]) {
      auto xyz = read_json_scalar_2d<ValueType>(itData["xyz"]);
      auto uvw = read_json_scalar_2d<ValueType>(itData["uvw"]);
      auto w = read_json_complex_2d<ValueType>(itData["w_real"], itData["w_imag"]);
      auto s = read_json_complex_2d<ValueType>(itData["s_real"], itData["s_imag"]);

      auto info =
          bipp::eigh<ValueType>(ctx_, wl, nAntenna, nBeam, s.data(), nBeam, w.data(), nAntenna,
                                xyz.data(), nAntenna, eigValues.data(), eigVec.data(), nAntenna);

      dataset.write(wl, info.second, eigVec.data(), nAntenna, eigValues.data(), xyz.data(),
                    nAntenna, uvw.data(), nAntenna * nAntenna);
    }

    // create selection
    std::unordered_map<std::string, std::vector<std::pair<std::size_t, const float*>>> selection;
    std::vector<std::vector<float>> eigenvalues;
    for (std::size_t idxInterval = 0; idxInterval < nIntervals; ++idxInterval) {
      const std::string tag = std::string("image_") + std::to_string(idxInterval);

      for (std::size_t idxSample = 0; idxSample < dataset.num_samples(); ++idxSample) {
        eigenvalues.emplace_back(dataset.num_beam());
        dataset.eig_val(idxSample, eigenvalues.back().data());
        eigMaskFunc(idxInterval, eigenvalues.back().size(), eigenvalues.back().data());
        selection[tag].emplace_back(idxSample, eigenvalues.back().data());
      }
    }

    bipp::NufftSynthesisOptions opt;
    opt.set_precision(std::is_same_v<T, float> ? BIPP_PRECISION_SINGLE : BIPP_PRECISION_DOUBLE);
    // opt.set_local_image_partition(bipp::Partition::Grid{{2,2,1}});

    auto comm = bipp::Communicator::local();

    std::vector<ValueType> lmn(3 * nPixel);
    std::copy(lmnX.begin(), lmnX.end(), lmn.begin());
    std::copy(lmnY.begin(), lmnY.end(), lmn.begin() + nPixel);
    std::copy(lmnZ.begin(), lmnZ.end(), lmn.begin() + 2 * nPixel);

    auto imgFile =
        bipp::ImageFile::create("test_nufft_synthesis_lofar_image.h5", nPixel, lmn.data(), nPixel);

    bipp::image_synthesis(ctx_, opt, dataset, std::move(selection), imgFile);

    std::vector<ValueType> img(nPixel);
    for (std::size_t idxInterval = 0; idxInterval < nIntervals; ++idxInterval) {
      const std::string tag = std::string("image_") + std::to_string(idxInterval);
      imgFile.get(tag, img.data());
      for (std::size_t i = 0; i < nPixel; ++i) {
        // Single precision is very inaccurate due to different summation orders
        // Use twice the absolute error for single precision
        ASSERT_NEAR(img[i], imgRef[i + idxInterval * nPixel], 0.05 * (4.0 / sizeof(ValueType)));
      }
    }
  }

  bipp::Context ctx_;
};

using NufftSynthesisLofarSingle = NufftSynthesisLofar<float>;
using NufftSynthesisLofarDouble = NufftSynthesisLofar<double>;

TEST_P(NufftSynthesisLofarSingle, Intensity_LSQ) { this->test_intensity(); }
TEST_P(NufftSynthesisLofarDouble, Intensity_LSQ) { this->test_intensity(); }

static auto param_type_names(const ::testing::TestParamInfo<std::tuple<BippProcessingUnit>>& info)
    -> std::string {
  std::stringstream stream;

  if (std::get<0>(info.param) == BIPP_PU_CPU)
    stream << "CPU";
  else
    stream << "GPU";

  return stream.str();
}

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#define TEST_PROCESSING_UNITS BIPP_PU_CPU, BIPP_PU_GPU
#else
#define TEST_PROCESSING_UNITS BIPP_PU_CPU
#endif

INSTANTIATE_TEST_SUITE_P(Lofar, NufftSynthesisLofarSingle,
                         ::testing::Combine(::testing::Values(TEST_PROCESSING_UNITS)),
                         param_type_names);

INSTANTIATE_TEST_SUITE_P(Lofar, NufftSynthesisLofarDouble,
                         ::testing::Combine(::testing::Values(TEST_PROCESSING_UNITS)),
                         param_type_names);
