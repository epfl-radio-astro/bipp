#include <cmath>
#include <complex>
#include <fstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include "bipp/bipp.hpp"
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
static auto get_lofar_ss_output_json() -> const nlohmann::json& {
  static nlohmann::json data = []() {
    if constexpr (std::is_same_v<T, float>) {
      std::ifstream file(std::string(BIPP_TEST_DATA_DIR) + "/lofar_ss_output_single.json");
      nlohmann::json j;
      file >> j;
      return j;
    } else {
      std::ifstream file(std::string(BIPP_TEST_DATA_DIR) + "/lofar_ss_output_double.json");
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
class StandardSynthesisLofar : public ::testing::TestWithParam<std::tuple<BippProcessingUnit>> {
protected:
  using ValueType = T;

  StandardSynthesisLofar() : ctx_(std::get<0>(GetParam())) {}

  auto test_intensity() -> void {
    const auto data = get_lofar_input_json();
    const auto output_data = get_lofar_ss_output_json<T>();

    const T wl = ValueType(data["wl"]);
    const std::size_t nAntenna = data["n_antenna"];
    const std::size_t nBeam = data["n_beam"];
    const std::size_t nEig = data["n_eig_int"];
    const std::size_t nIntervals = data["intervals_int"].size();
    const auto intervals = read_json_scalar_2d<T>(data["intervals_int"]);

    const auto imgRef = read_json_scalar_2d<T>(output_data[std::string("int_") + "lsq"]);
    const auto pixelX = read_json_scalar_1d<T>(output_data["pixel_x"]);
    const auto pixelY = read_json_scalar_1d<T>(output_data["pixel_y"]);
    const auto pixelZ = read_json_scalar_1d<T>(output_data["pixel_z"]);
    const std::size_t nPixel = imgRef.size() / nIntervals;

    bipp::StandardSynthesis<T> imager(ctx_, bipp::StandardSynthesisOptions(), nIntervals, nPixel,
                                      pixelX.data(), pixelY.data(), pixelZ.data());

    // map intervals to mask
    auto eigMaskFunc = [&](std::size_t idxBin, std::size_t nEigOut, T* d) -> void {
      const T dMin = intervals[idxBin * 2];
      const T dMax = intervals[idxBin * 2 + 1];

      std::size_t idxEig = 0;
      for(; idxEig < nEigOut - nEig; ++idxEig) {
        d[idxEig] = 0;
      }
      for(; idxEig < nEigOut; ++idxEig) {
        const auto val = d[idxEig];
        d[idxEig] *= (val >= dMin && val <= dMax);
      }
    };

    std::size_t nEpochs = 0;
    for (const auto& itData : data["data"]) {
      auto xyz = read_json_scalar_2d<ValueType>(itData["xyz"]);
      auto w = read_json_complex_2d<ValueType>(itData["w_real"], itData["w_imag"]);
      auto s = read_json_complex_2d<ValueType>(itData["s_real"], itData["s_imag"]);

      imager.collect(nAntenna, nBeam, wl, eigMaskFunc, s.data(), nBeam, w.data(), nAntenna,
                     xyz.data(), nAntenna);
      ++nEpochs;
      }

      std::vector<T> img(imgRef.size());
      imager.get(img.data(), nPixel);

      for (std::size_t i = 0; i < img.size(); ++i) {
        // Single precision is very inaccurate due to different summation orders
        // Use twice the absolute error for single precision
        ASSERT_NEAR(img[i], imgRef[i], 0.05 * (4.0 / sizeof(T)));
      }
    }

    bipp::Context ctx_;
};

using StandardSynthesisLofarSingle = StandardSynthesisLofar<float>;
using StandardSynthesisLofarDouble = StandardSynthesisLofar<double>;

TEST_P(StandardSynthesisLofarSingle, Intensity_LSQ) { this->test_intensity(); }
TEST_P(StandardSynthesisLofarDouble, Intensity_LSQ) { this->test_intensity(); }

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

INSTANTIATE_TEST_SUITE_P(Lofar, StandardSynthesisLofarSingle,
                         ::testing::Combine(::testing::Values(TEST_PROCESSING_UNITS)),
                         param_type_names);

INSTANTIATE_TEST_SUITE_P(Lofar, StandardSynthesisLofarDouble,
                         ::testing::Combine(::testing::Values(TEST_PROCESSING_UNITS)),
                         param_type_names);
