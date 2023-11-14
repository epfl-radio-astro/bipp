#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <chrono>
#include <functional>
#include <cstring>

#include "bipp/bipp.hpp"

using namespace bipp;
namespace py = pybind11;

// Helper
namespace {
template <typename TARGET, typename SOURCE,
          typename = std::enable_if_t<std::is_integral_v<TARGET> && std::is_integral_v<SOURCE> &&
                                      !std::is_signed_v<TARGET>>>
inline auto safe_cast(SOURCE s) {
  if constexpr (std::is_signed_v<SOURCE>) {
    if (s < 0)
      throw std::underflow_error("Integer underflow due to input size or stride being negative.");
  }

  if ((sizeof(TARGET) >= sizeof(SOURCE) &&
       static_cast<TARGET>(s) > std::numeric_limits<TARGET>::max()) ||
      (sizeof(TARGET) < sizeof(SOURCE) &&
       s > static_cast<SOURCE>(std::numeric_limits<TARGET>::max())))
    throw std::overflow_error("Integer overflow due to input size or stride.");
  return static_cast<TARGET>(s);
}

template <typename T, int STYLE>
auto check_1d_array(const py::array_t<T, STYLE>& a, long shape = 0) -> void {
  if (a.ndim() != 1) throw InvalidParameterError();
  if (shape && a.shape(0) != shape) throw InvalidParameterError();
}

template <typename T, int STYLE>
auto check_2d_array(const py::array_t<T, STYLE>& a, std::array<long, 2> shape = {0, 0}) -> void {
  if (a.ndim() != 2) throw InvalidParameterError();
  if (shape[0] && a.shape(0) != shape[0]) throw InvalidParameterError();
  if (shape[1] && a.shape(1) != shape[1]) throw InvalidParameterError();
}

template <typename T, int STYLE>
auto check_3d_array(const py::array_t<T, STYLE>& a, std::array<long, 3> shape = {0, 0, 0}) -> void {
  if (a.ndim() != 3) throw InvalidParameterError();
  if (shape[0] && a.shape(0) != shape[0]) throw InvalidParameterError();
  if (shape[1] && a.shape(1) != shape[1]) throw InvalidParameterError();
  if (shape[2] && a.shape(2) != shape[2]) throw InvalidParameterError();
}

auto string_to_processing_unit(const std::string& pu) -> BippProcessingUnit {
  if (pu == "CPU" || pu == "cpu") return BIPP_PU_CPU;
  if (pu == "GPU" || pu == "gpu") return BIPP_PU_GPU;
  if (pu == "AUTO" || pu == "auto") return BIPP_PU_AUTO;

  throw InvalidParameterError();
}

auto string_to_filter(const std::string& f) -> BippFilter {
  if (f == "LSQ" || f == "lsq") return BIPP_FILTER_LSQ;
  if (f == "STD" || f == "std") return BIPP_FILTER_STD;
  if (f == "SQRT" || f == "sqrt") return BIPP_FILTER_SQRT;
  if (f == "INV" || f == "inv") return BIPP_FILTER_INV;
  if (f == "INV_SQ" || f == "inv_sq") return BIPP_FILTER_INV_SQ;

  throw InvalidParameterError();
}

auto processing_unit_to_string(BippProcessingUnit pu) -> std::string {
  if (pu == BIPP_PU_CPU) return "CPU";
  if (pu == BIPP_PU_GPU) return "GPU";
  if (pu == BIPP_PU_AUTO) return "AUTO";

  throw InvalidParameterError();
}

auto create_context(const std::string& pu) -> Context {
  return Context(string_to_processing_unit(pu));
}

template <typename T>
auto call_gram_matrix(Context& ctx, const py::array_t<T, py::array::f_style>& xyz,
                      const py::array_t<std::complex<T>, py::array::f_style>& w, T wl) {
  check_2d_array(w);
  check_2d_array(xyz, {w.shape(0), 3});

  auto g = py::array_t<std::complex<T>, py::array::f_style>({w.shape(1), w.shape(1)});

  gram_matrix(ctx, safe_cast<std::size_t>(w.shape(0)), safe_cast<std::size_t>(w.shape(1)),
              w.data(0), safe_cast<std::size_t>(w.strides(1) / w.itemsize()), xyz.data(0),
              safe_cast<std::size_t>(xyz.strides(1) / xyz.itemsize()), wl, g.mutable_data(0),
              safe_cast<std::size_t>(g.strides(1) / g.itemsize()));

  return g;
}

template <typename T>
auto call_eigh(Context& ctx, T wl, const py::array_t<std::complex<T>, py::array::f_style>& s,
               const py::array_t<std::complex<T>, py::array::f_style>& w,
               const py::array_t<T, py::array::f_style>& xyz) {
  check_2d_array(w);
  auto nAntenna = w.shape(0);
  auto nBeam = w.shape(1);
  check_2d_array(xyz, {nAntenna, 3});
  check_2d_array(s, {nBeam, nBeam});

  auto d = py::array_t<T, py::array::f_style>({py::ssize_t(nBeam)});
  std::size_t nEigOut = 0;

  nEigOut = eigh<T>(ctx, wl, nAntenna, nBeam, s.data(0),
                    safe_cast<std::size_t>(s.strides(1) / s.itemsize()), w.data(0),
                    safe_cast<std::size_t>(w.strides(1) / w.itemsize()), xyz.data(0),
                    safe_cast<std::size_t>(xyz.strides(1) / xyz.itemsize()), d.mutable_data(0));

  d.resize({nEigOut});

  return d;
}

struct StandardSynthesisDispatcher {
  StandardSynthesisDispatcher(Context& ctx, std::size_t nAntenna, std::size_t nBeam,
                              std::size_t nIntervals, const std::vector<std::string>& filter,
                              const py::array& pixelX, const py::array& pixelY,
                              const py::array& pixelZ, const std::string& precision)
      : nIntervals_(nIntervals), nPixel_(pixelX.shape(0)) {
    std::vector<BippFilter> filterEnums;
    for (const auto& f : filter) {
      filterEnums.emplace_back(string_to_filter(f));
    }
    if (precision == "single" || precision == "SINGLE") {
      py::array_t<float, pybind11::array::f_style | py::array::forcecast> pixelXArray(pixelX);
      py::array_t<float, pybind11::array::f_style | py::array::forcecast> pixelYArray(pixelY);
      py::array_t<float, pybind11::array::f_style | py::array::forcecast> pixelZArray(pixelZ);
      check_1d_array(pixelXArray);
      check_1d_array(pixelYArray, pixelXArray.shape(0));
      check_1d_array(pixelZArray, pixelXArray.shape(0));
      plan_ = StandardSynthesis<float>(
          ctx, nAntenna, nBeam, nIntervals, filterEnums.size(), filterEnums.data(),
          pixelXArray.shape(0), pixelXArray.data(0), pixelYArray.data(0), pixelZArray.data(0));
    } else if (precision == "double" || precision == "DOUBLE") {
      py::array_t<double, pybind11::array::f_style | py::array::forcecast> pixelXArray(pixelX);
      py::array_t<double, pybind11::array::f_style | py::array::forcecast> pixelYArray(pixelY);
      py::array_t<double, pybind11::array::f_style | py::array::forcecast> pixelZArray(pixelZ);
      check_1d_array(pixelXArray);
      check_1d_array(pixelYArray, pixelXArray.shape(0));
      check_1d_array(pixelZArray, pixelXArray.shape(0));
      plan_ = StandardSynthesis<double>(
          ctx, nAntenna, nBeam, nIntervals, filterEnums.size(), filterEnums.data(),
          pixelXArray.shape(0), pixelXArray.data(0), pixelYArray.data(0), pixelZArray.data(0));
    } else {
      throw InvalidParameterError();
    }
  }

  StandardSynthesisDispatcher(StandardSynthesisDispatcher&&) = default;

  StandardSynthesisDispatcher(const StandardSynthesisDispatcher&) = delete;

  StandardSynthesisDispatcher& operator=(StandardSynthesisDispatcher&&) = default;

  StandardSynthesisDispatcher& operator=(const StandardSynthesisDispatcher&) = delete;

  auto collect(double wl, const std::function<pybind11::array(std::size_t, pybind11::array)>& eigMaskFunc, pybind11::array w,
               pybind11::array xyz, std::optional<pybind11::array> s) -> void {
    std::visit(
        [&](auto&& arg) -> void {
          using variantType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<variantType, StandardSynthesis<float>> ||
                        std::is_same_v<variantType, StandardSynthesis<double>>) {
            using T = typename variantType::valueType;
            py::array_t<std::complex<T>, py::array::f_style | py::array::forcecast> wArray(w);
            check_2d_array(wArray);
            auto nAntenna = wArray.shape(0);
            auto nBeam = wArray.shape(1);
            py::array_t<T, py::array::f_style | py::array::forcecast> xyzArray(xyz);
            check_2d_array(xyzArray, {nAntenna, 3});
            std::optional<py::array_t<std::complex<T>, py::array::f_style | py::array::forcecast>>
                sArray;
            if (s) {
              sArray = py::array_t<std::complex<T>, py::array::f_style | py::array::forcecast>(
                  s.value());
              check_2d_array(sArray.value(), {nBeam, nBeam});
            }

            auto eigMaskFuncLambda = [&](std::size_t nBins, std::size_t nEig, const T* d, int* maskOut) {
              py::array_t<T> dArray(nEig);
              std::memcpy(dArray.mutable_data(0), d, nEig * sizeof(T));

              py::array_t<int, py::array::f_style | py::array::forcecast> mask(
                  eigMaskFunc(nBins, dArray));
              check_2d_array(mask, {long(nEig), long(nBins)});

              const auto* maskPtr = mask.data(0);
              const auto ldmask = safe_cast<std::size_t>(mask.strides(1) / mask.itemsize());

              for (std::size_t i = 0; i < nBins; ++i) {
                std::memcpy(maskOut + i * nEig, maskPtr + i * ldmask, nEig * sizeof(int));
              }
            };

            std::get<StandardSynthesis<T>>(plan_).collect(
                wl, eigMaskFuncLambda, s ? sArray.value().data(0) : nullptr,
                safe_cast<std::size_t>(sArray.value().strides(1) / sArray.value().itemsize()),
                wArray.data(0), safe_cast<std::size_t>(wArray.strides(1) / wArray.itemsize()),
                xyzArray.data(0),
                safe_cast<std::size_t>(xyzArray.strides(1) / xyzArray.itemsize()));

          } else {
            throw InternalError();
          }
        },
        plan_);
  }

  auto get(const std::string& fString) -> py::array {
    const auto f = string_to_filter(fString);
    return std::visit(
        [&](auto&& arg) -> pybind11::array {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, StandardSynthesis<double>>) {
            py::array_t<double> out({nIntervals_, nPixel_});
            std::get<StandardSynthesis<double>>(plan_).get(
                f, out.mutable_data(0), safe_cast<std::size_t>(out.strides(0) / out.itemsize()));
            return out;
          } else if constexpr (std::is_same_v<T, StandardSynthesis<float>>) {
            py::array_t<float> out({nIntervals_, nPixel_});
            std::get<StandardSynthesis<float>>(plan_).get(
                f, out.mutable_data(0), safe_cast<std::size_t>(out.strides(0) / out.itemsize()));
            return out;
          } else {
            throw InternalError();
            return py::array_t<double, py::array::f_style>();
          }
        },
        plan_);
  }

  std::variant<std::monostate, StandardSynthesis<float>, StandardSynthesis<double>> plan_;
  std::size_t nIntervals_, nPixel_;
};

struct NufftSynthesisDispatcher {
  NufftSynthesisDispatcher(Context& ctx, NufftSynthesisOptions opt, std::size_t nAntenna,
                           std::size_t nBeam, std::size_t nIntervals,
                           const std::vector<std::string>& filter, const py::array& lmnX,
                           const py::array& lmnY, const py::array& lmnZ,
                           const std::string& precision)
      : nIntervals_(nIntervals), nPixel_(lmnX.shape(0)) {
    std::vector<BippFilter> filterEnums;
    for (const auto& f : filter) {
      filterEnums.emplace_back(string_to_filter(f));
    }
    if (precision == "single" || precision == "SINGLE") {
      py::array_t<float, pybind11::array::f_style | py::array::forcecast> lmnXArray(lmnX);
      py::array_t<float, pybind11::array::f_style | py::array::forcecast> lmnYArray(lmnY);
      py::array_t<float, pybind11::array::f_style | py::array::forcecast> lmnZArray(lmnZ);
      check_1d_array(lmnXArray);
      check_1d_array(lmnYArray, lmnXArray.shape(0));
      check_1d_array(lmnZArray, lmnXArray.shape(0));
      plan_ = NufftSynthesis<float>(ctx, std::move(opt), nAntenna, nBeam, nIntervals,
                                    filterEnums.size(), filterEnums.data(), lmnXArray.shape(0),
                                    lmnXArray.data(0), lmnYArray.data(0), lmnZArray.data(0));
    } else if (precision == "double" || precision == "DOUBLE") {
      py::array_t<double, pybind11::array::f_style | py::array::forcecast> lmnXArray(lmnX);
      py::array_t<double, pybind11::array::f_style | py::array::forcecast> lmnYArray(lmnY);
      py::array_t<double, pybind11::array::f_style | py::array::forcecast> lmnZArray(lmnZ);
      check_1d_array(lmnXArray);
      check_1d_array(lmnYArray, lmnXArray.shape(0));
      check_1d_array(lmnZArray, lmnXArray.shape(0));
      plan_ = NufftSynthesis<double>(ctx, std::move(opt), nAntenna, nBeam, nIntervals,
                                     filterEnums.size(), filterEnums.data(), lmnXArray.shape(0),
                                     lmnXArray.data(0), lmnYArray.data(0), lmnZArray.data(0));
    } else {
      throw InvalidParameterError();
    }
  }

  NufftSynthesisDispatcher(NufftSynthesisDispatcher&&) = default;

  NufftSynthesisDispatcher(const NufftSynthesisDispatcher&) = delete;

  NufftSynthesisDispatcher& operator=(NufftSynthesisDispatcher&&) = default;

  NufftSynthesisDispatcher& operator=(const NufftSynthesisDispatcher&) = delete;

  auto collect(std::size_t nEig, double wl, pybind11::array intervals, pybind11::array w,
               pybind11::array xyz, pybind11::array uvw, std::optional<pybind11::array> s) -> void {
    std::visit(
        [&](auto&& arg) -> void {
          using variantType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<variantType, NufftSynthesis<float>> ||
                        std::is_same_v<variantType, NufftSynthesis<double>>) {
            using T = typename variantType::valueType;
            py::array_t<T, py::array::c_style | py::array::forcecast> intervalsArray(intervals);
            check_2d_array(intervalsArray, {static_cast<long>(nIntervals_), 2});
            py::array_t<std::complex<T>, py::array::f_style | py::array::forcecast> wArray(w);
            check_2d_array(wArray);
            auto nAntenna = wArray.shape(0);
            auto nBeam = wArray.shape(1);
            py::array_t<T, py::array::f_style | py::array::forcecast> xyzArray(xyz);
            check_2d_array(xyzArray, {nAntenna, 3});
            py::array_t<T, py::array::f_style | py::array::forcecast> uvwArray(uvw);
            check_2d_array(uvwArray, {nAntenna * nAntenna, 3});

            std::optional<py::array_t<std::complex<T>, py::array::f_style | py::array::forcecast>>
                sArray;
            if (s) {
              sArray = py::array_t<std::complex<T>, py::array::f_style | py::array::forcecast>(
                  s.value());
              check_2d_array(sArray.value(), {nBeam, nBeam});
            }
            std::get<NufftSynthesis<T>>(plan_).collect(
                nEig, wl, intervalsArray.data(0),
                safe_cast<std::size_t>(intervals.strides(0) / intervals.itemsize()),
                s ? sArray.value().data(0) : nullptr,
                s ? safe_cast<std::size_t>(sArray.value().strides(1) / sArray.value().itemsize())
                  : 0,
                wArray.data(0), safe_cast<std::size_t>(wArray.strides(1) / wArray.itemsize()),
                xyzArray.data(0), safe_cast<std::size_t>(xyzArray.strides(1) / xyzArray.itemsize()),
                uvwArray.data(0),
                safe_cast<std::size_t>(uvwArray.strides(1) / uvwArray.itemsize()));

          } else {
            throw InternalError();
          }
        },
        plan_);
  }

  auto get(const std::string& fString) -> py::array {
    const auto f = string_to_filter(fString);
    return std::visit(
        [&](auto&& arg) -> pybind11::array {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, NufftSynthesis<double>>) {
            py::array_t<double> out({nIntervals_, nPixel_});
            std::get<NufftSynthesis<double>>(plan_).get(
                f, out.mutable_data(0), safe_cast<std::size_t>(out.strides(0) / out.itemsize()));
            return out;
          } else if constexpr (std::is_same_v<T, NufftSynthesis<float>>) {
            py::array_t<float> out({nIntervals_, nPixel_});
            std::get<NufftSynthesis<float>>(plan_).get(
                f, out.mutable_data(0), safe_cast<std::size_t>(out.strides(0) / out.itemsize()));
            return out;
          } else {
            throw InternalError();
            return py::array_t<double, py::array::f_style>();
          }
        },
        plan_);
  }

  std::variant<std::monostate, NufftSynthesis<float>, NufftSynthesis<double>> plan_;
  std::size_t nIntervals_, nPixel_;
};

}  // namespace

// Create module

// NOTE: Function overloading does NOT generally try to find the best matching
// types and the declaration order matters. Always declare single precision
// functions first, as otherwise double precision versions will always be
// selected.
PYBIND11_MODULE(pybipp, m) {
  m.doc() = R"pbdoc(
        Bipp
    )pbdoc";
#ifdef BIPP_VERSION
  m.attr("__version__") = BIPP_VERSION;
#else
  m.attr("__version__") = "dev";
#endif
  pybind11::class_<Context>(m, "Context")
      .def(py::init(&create_context), pybind11::arg("pu"))
      .def_property_readonly("processing_unit",
           [](const Context& ctx) { return processing_unit_to_string(ctx.processing_unit()); });

  pybind11::class_<Partition>(m, "Partition")
      .def(py::init())
      .def_static("auto", []() { return Partition{Partition::Auto()}; })
      .def_static("none", []() { return Partition{Partition::None()}; })
      .def_static(
          "grid",
          [](std::array<std::size_t, 3> dimensions) {
            return Partition{Partition::Grid{dimensions}};
          },
          pybind11::arg("dimensions"));

  pybind11::class_<NufftSynthesisOptions>(m, "NufftSynthesisOptions")
      .def(py::init())
      .def_readwrite("tolerance", &NufftSynthesisOptions::tolerance)
      .def("set_tolerance", &NufftSynthesisOptions::set_tolerance)
      .def_readwrite("collect_group_size", &NufftSynthesisOptions::collectGroupSize)
      .def("set_collect_group_size", &NufftSynthesisOptions::set_collect_group_size)
      .def_readwrite("local_image_partition", &NufftSynthesisOptions::localImagePartition)
      .def("set_local_image_partition", &NufftSynthesisOptions::set_local_image_partition)
      .def_readwrite("local_uvw_partition", &NufftSynthesisOptions::localUVWPartition)
      .def("set_local_uvw_partition", &NufftSynthesisOptions::set_local_uvw_partition);

  pybind11::class_<NufftSynthesisDispatcher>(m, "NufftSynthesis")
      .def(pybind11::init<Context&, NufftSynthesisOptions, std::size_t, std::size_t, std::size_t,
                          const std::vector<std::string>&, const py::array&, const py::array&,
                          const py::array&, const std::string&>(),
           pybind11::arg("ctx"), pybind11::arg("opt"), pybind11::arg("n_antenna"),
           pybind11::arg("n_beam"), pybind11::arg("n_intervals"), pybind11::arg("filter"),
           pybind11::arg("lmn_x"), pybind11::arg("lmn_y"), pybind11::arg("lmn_y"),
           pybind11::arg("precision"))
      .def("collect", &NufftSynthesisDispatcher::collect, pybind11::arg("n_eig"),
           pybind11::arg("wl"), pybind11::arg("intervals"), pybind11::arg("w"),
           pybind11::arg("xyz"), pybind11::arg("uvw"),
           pybind11::arg("s") = std::optional<pybind11::array>())
      .def("get", &NufftSynthesisDispatcher::get, pybind11::arg("f"));

  pybind11::class_<StandardSynthesisDispatcher>(m, "StandardSynthesis")
      .def(pybind11::init<Context&, std::size_t, std::size_t, std::size_t,
                          const std::vector<std::string>&, const py::array&, const py::array&,
                          const py::array&, const std::string&>(),
           pybind11::arg("ctx"), pybind11::arg("n_antenna"), pybind11::arg("n_beam"),
           pybind11::arg("n_intervals"), pybind11::arg("filter"), pybind11::arg("lmn_x"),
           pybind11::arg("lmn_y"), pybind11::arg("lmn_y"), pybind11::arg("precision"))
      .def("collect", &StandardSynthesisDispatcher::collect, pybind11::arg("wl"),
           pybind11::arg("mask"), pybind11::arg("w"), pybind11::arg("xyz"),
           pybind11::arg("s") = std::optional<pybind11::array>())
      .def("get", &StandardSynthesisDispatcher::get, pybind11::arg("f"));

  m.def(
       "gram_matrix",
       [](Context& ctx, const py::array_t<float, pybind11::array::f_style>& xyz,
          const py::array_t<std::complex<float>, pybind11::array::f_style>& w,
          float wl) { return call_gram_matrix(ctx, xyz, w, wl); },
       pybind11::arg("ctx"), pybind11::arg("XYZ"), pybind11::arg("W"), pybind11::arg("wl"))
      .def(
          "gram_matrix",
          [](Context& ctx, const py::array_t<double, pybind11::array::f_style>& xyz,
             const py::array_t<std::complex<double>, pybind11::array::f_style>& w,
             double wl) { return call_gram_matrix(ctx, xyz, w, wl); },
          pybind11::arg("ctx"), pybind11::arg("XYZ"), pybind11::arg("W"), pybind11::arg("wl"));

  m.def(
       "eigh",
       [](Context& ctx, float wl, const py::array_t<std::complex<float>, py::array::f_style>& s,
          const py::array_t<std::complex<float>, py::array::f_style>& w,
          const py::array_t<float, py::array::f_style>& xyz) {
         return call_eigh(ctx, wl, s, w, xyz);
       },
       pybind11::arg("ctx"), pybind11::arg("wl"), pybind11::arg("s"), pybind11::arg("w"),
       pybind11::arg("xyz"))
      .def(
          "eigh",
          [](Context& ctx, double wl,
             const py::array_t<std::complex<double>, py::array::f_style>& s,
             const py::array_t<std::complex<double>, py::array::f_style>& w,
             const py::array_t<double, py::array::f_style>& xyz) {
            return call_eigh(ctx, wl, s, w, xyz);
          },
          pybind11::arg("ctx"), pybind11::arg("wl"), pybind11::arg("s"), pybind11::arg("w"),
          pybind11::arg("xyz"));

  // m.def(
  //     "test",
  //     [](const std::function<py::array_t<int, py::array::f_style>(
  //            std::size_t, py::array_t<float, py::array::f_style>)>& func) {
  //       py::array_t<float, pybind11::array::f_style | py::array::forcecast> d(15);
  //       auto start = std::chrono::high_resolution_clock::now();
  //       func(5, d);
  //       auto end = std::chrono::high_resolution_clock::now();
  //       const std::chrono::duration<double> elapsed_seconds{end - start};
  //       printf("time; %f\n", elapsed_seconds.count());
  //     },
  //     pybind11::arg("func"));
}
