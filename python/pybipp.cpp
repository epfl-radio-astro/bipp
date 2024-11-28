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

auto processing_unit_to_string(BippProcessingUnit pu) -> std::string {
  if (pu == BIPP_PU_CPU) return "CPU";
  if (pu == BIPP_PU_GPU) return "GPU";
  if (pu == BIPP_PU_AUTO) return "AUTO";

  throw InvalidParameterError();
}

void center_array(py::array_t<double> input_array) {
  py::buffer_info buf_info = input_array.request();
  double *ptr = static_cast<double *>(buf_info.ptr);
  const auto N = buf_info.shape[0];
  const auto M = buf_info.shape[1];
  for (auto j=0; j<M; j++) {
    double mean = 0.0;
    for (auto i=j*N; i<(j+1)*N; i++) {
      mean += ptr[i];
    }
    mean /= N;
    for (auto i=j*N; i<(j+1)*N; i++) {
      ptr[i] = ptr[i] - mean;
    }
  }
}

auto create_context(const std::string& pu) -> Context {
  return Context(string_to_processing_unit(pu));
}

auto create_distributed_context(const std::string& pu, Communicator comm) -> Context {
  return Context(string_to_processing_unit(pu), std::move(comm));
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
  std::pair<std::size_t, std::size_t> pev{0, 0};
  pev = eigh<T>(ctx, wl, nAntenna, nBeam, s.data(0),
                    safe_cast<std::size_t>(s.strides(1) / s.itemsize()), w.data(0),
                    safe_cast<std::size_t>(w.strides(1) / w.itemsize()), xyz.data(0),
                    safe_cast<std::size_t>(xyz.strides(1) / xyz.itemsize()), d.mutable_data(0));
  d.resize({pev.first});

  return d;
}

struct StandardSynthesisDispatcher {
  StandardSynthesisDispatcher(Context& ctx, StandardSynthesisOptions opt, std::size_t nImages,
                              const py::array& pixelX, const py::array& pixelY,
                              const py::array& pixelZ, const std::string& precision)
      : nImages_(nImages), nPixel_(pixelX.shape(0)) {
    if (precision == "single" || precision == "SINGLE") {
      py::array_t<float, pybind11::array::f_style | py::array::forcecast> pixelXArray(pixelX);
      py::array_t<float, pybind11::array::f_style | py::array::forcecast> pixelYArray(pixelY);
      py::array_t<float, pybind11::array::f_style | py::array::forcecast> pixelZArray(pixelZ);
      check_1d_array(pixelXArray);
      check_1d_array(pixelYArray, pixelXArray.shape(0));
      check_1d_array(pixelZArray, pixelXArray.shape(0));
      plan_ = StandardSynthesis<float>(ctx, opt, nImages, pixelXArray.shape(0), pixelXArray.data(0),
                                       pixelYArray.data(0), pixelZArray.data(0));
    } else if (precision == "double" || precision == "DOUBLE") {
      py::array_t<double, pybind11::array::f_style | py::array::forcecast> pixelXArray(pixelX);
      py::array_t<double, pybind11::array::f_style | py::array::forcecast> pixelYArray(pixelY);
      py::array_t<double, pybind11::array::f_style | py::array::forcecast> pixelZArray(pixelZ);
      check_1d_array(pixelXArray);
      check_1d_array(pixelYArray, pixelXArray.shape(0));
      check_1d_array(pixelZArray, pixelXArray.shape(0));
      plan_ =
          StandardSynthesis<double>(ctx, opt, nImages, pixelXArray.shape(0), pixelXArray.data(0),
                                    pixelYArray.data(0), pixelZArray.data(0));
    } else {
      throw InvalidParameterError();
    }
  }

  StandardSynthesisDispatcher(StandardSynthesisDispatcher&&) = default;

  StandardSynthesisDispatcher(const StandardSynthesisDispatcher&) = delete;

  StandardSynthesisDispatcher& operator=(StandardSynthesisDispatcher&&) = default;

  StandardSynthesisDispatcher& operator=(const StandardSynthesisDispatcher&) = delete;

  auto collect(double wl,
               const std::function<pybind11::array(std::size_t, pybind11::array)>& eigMaskFunc,
               pybind11::array s, pybind11::array w, pybind11::array xyz) -> void {
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
            // Always center xyz array in double precision
            // TODO: unify this and the C interface
            center_array(xyz);
            py::array_t<T, py::array::f_style | py::array::forcecast> xyzArray(xyz);
            check_2d_array(xyzArray, {nAntenna, 3});
            py::array_t<std::complex<T>, py::array::f_style | py::array::forcecast> sArray(s);
            check_2d_array(sArray, {nBeam, nBeam});

            auto eigMaskFuncLambda = [&](std::size_t idxBin, std::size_t nEig, T* d) {
              py::array_t<T> dArray(nEig);
              std::memcpy(dArray.mutable_data(0), d, nEig * sizeof(T));

              py::array_t<T, py::array::f_style | py::array::forcecast> dNew(
                  eigMaskFunc(idxBin, dArray));
              check_1d_array(dNew, nEig);

              std::memcpy(d, dNew.data(0), nEig * sizeof(T));
            };

            std::get<StandardSynthesis<T>>(plan_).collect(
                nAntenna, nBeam, wl, eigMaskFuncLambda, sArray.data(0),
                safe_cast<std::size_t>(sArray.strides(1) / sArray.itemsize()), wArray.data(0),
                safe_cast<std::size_t>(wArray.strides(1) / wArray.itemsize()), xyzArray.data(0),
                safe_cast<std::size_t>(xyzArray.strides(1) / xyzArray.itemsize()));

          } else {
            throw InternalError();
          }
        },
        plan_);
  }

  auto get() -> py::array {
    return std::visit(
        [&](auto&& arg) -> pybind11::array {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, StandardSynthesis<double>>) {
            py::array_t<double> out({nImages_, nPixel_});
            std::get<StandardSynthesis<double>>(plan_).get(
                out.mutable_data(0), safe_cast<std::size_t>(out.strides(0) / out.itemsize()));
            return out;
          } else if constexpr (std::is_same_v<T, StandardSynthesis<float>>) {
            py::array_t<float> out({nImages_, nPixel_});
            std::get<StandardSynthesis<float>>(plan_).get(
                out.mutable_data(0), safe_cast<std::size_t>(out.strides(0) / out.itemsize()));
            return out;
          } else {
            throw InternalError();
            return py::array_t<double, py::array::f_style>();
          }
        },
        plan_);
  }

  std::variant<std::monostate, StandardSynthesis<float>, StandardSynthesis<double>> plan_;
  std::size_t nImages_, nPixel_;
};

struct DatasetCreatorDispatcher {
  DatasetCreatorDispatcher(const std::string& fileName, const std::string& description,
                           std::size_t nAntenna, std::size_t nBeam)
      : creator_(fileName, description, nAntenna, nBeam) {}
  auto process_and_write(const std::string& precision, double wl,
               pybind11::array s, pybind11::array w, pybind11::array xyz, pybind11::array uvw) -> void{
    if (precision == "single" || precision == "SINGLE") {
      process_and_write_t<float>(wl, s, w, xyz, uvw);
    } else if (precision == "double" || precision == "DOUBLE") {
      process_and_write_t<double>(wl, s, w, xyz, uvw);
    } else {
      throw InvalidParameterError();
    }
  }

  template <typename T>
  auto process_and_write_t(double wl, pybind11::array s, pybind11::array w, pybind11::array xyz,
                           pybind11::array uvw) -> void{
            py::array_t<std::complex<T>, py::array::f_style | py::array::forcecast> wArray(w);
            check_2d_array(wArray);
            long nAntenna = creator_.num_antenna();
            long nBeam = creator_.num_beam();
            py::array_t<T, py::array::f_style | py::array::forcecast> xyzArray(xyz);
            check_2d_array(xyzArray, {nAntenna, 3});
            py::array_t<T, py::array::f_style | py::array::forcecast> uvwArray(uvw);
            check_2d_array(uvwArray, {nAntenna * nAntenna, 3});

            auto sArray =
                py::array_t<std::complex<T>, py::array::f_style | py::array::forcecast>(s);
            check_2d_array(sArray, {nBeam, nBeam});

            creator_.process_and_write(
                wl, sArray.data(0), safe_cast<std::size_t>(sArray.strides(1) / sArray.itemsize()),
                wArray.data(0), safe_cast<std::size_t>(wArray.strides(1) / wArray.itemsize()),
                xyzArray.data(0), safe_cast<std::size_t>(xyzArray.strides(1) / xyzArray.itemsize()),
                uvwArray.data(0),
                safe_cast<std::size_t>(uvwArray.strides(1) / uvwArray.itemsize()));
  }

  DatasetCreator creator_;
};

struct NufftSynthesisDispatcher {
  NufftSynthesisDispatcher(Context& ctx, NufftSynthesisOptions opt, std::size_t nImages,
                           const py::array& lmnX, const py::array& lmnY, const py::array& lmnZ,
                           const std::string& precision)
      : nImages_(nImages), nPixel_(lmnX.shape(0)) {
    if (precision == "single" || precision == "SINGLE") {
      py::array_t<float, pybind11::array::f_style | py::array::forcecast> lmnXArray(lmnX);
      py::array_t<float, pybind11::array::f_style | py::array::forcecast> lmnYArray(lmnY);
      py::array_t<float, pybind11::array::f_style | py::array::forcecast> lmnZArray(lmnZ);
      check_1d_array(lmnXArray);
      check_1d_array(lmnYArray, lmnXArray.shape(0));
      check_1d_array(lmnZArray, lmnXArray.shape(0));
      plan_ = NufftSynthesis<float>(ctx, std::move(opt), nImages, lmnXArray.shape(0),
                                    lmnXArray.data(0), lmnYArray.data(0), lmnZArray.data(0));
    } else if (precision == "double" || precision == "DOUBLE") {
      py::array_t<double, pybind11::array::f_style | py::array::forcecast> lmnXArray(lmnX);
      py::array_t<double, pybind11::array::f_style | py::array::forcecast> lmnYArray(lmnY);
      py::array_t<double, pybind11::array::f_style | py::array::forcecast> lmnZArray(lmnZ);
      check_1d_array(lmnXArray);
      check_1d_array(lmnYArray, lmnXArray.shape(0));
      check_1d_array(lmnZArray, lmnXArray.shape(0));
      plan_ = NufftSynthesis<double>(ctx, std::move(opt), nImages, lmnXArray.shape(0),
                                     lmnXArray.data(0), lmnYArray.data(0), lmnZArray.data(0));
    } else {
      throw InvalidParameterError();
    }
  }

  NufftSynthesisDispatcher(NufftSynthesisDispatcher&&) = default;

  NufftSynthesisDispatcher(const NufftSynthesisDispatcher&) = delete;

  NufftSynthesisDispatcher& operator=(NufftSynthesisDispatcher&&) = default;

  NufftSynthesisDispatcher& operator=(const NufftSynthesisDispatcher&) = delete;

  auto collect(double wl,
               const std::function<pybind11::array(std::size_t, pybind11::array)>& eigMaskFunc,
               pybind11::array s, pybind11::array w, pybind11::array xyz, pybind11::array uvw)
      -> void {
    std::visit(
        [&](auto&& arg) -> void {
          using variantType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<variantType, NufftSynthesis<float>> ||
                        std::is_same_v<variantType, NufftSynthesis<double>>) {
            using T = typename variantType::valueType;
            py::array_t<std::complex<T>, py::array::f_style | py::array::forcecast> wArray(w);
            check_2d_array(wArray);
            auto nAntenna = wArray.shape(0);
            auto nBeam = wArray.shape(1);
            py::array_t<T, py::array::f_style | py::array::forcecast> xyzArray(xyz);
            check_2d_array(xyzArray, {nAntenna, 3});
            py::array_t<T, py::array::f_style | py::array::forcecast> uvwArray(uvw);
            check_2d_array(uvwArray, {nAntenna * nAntenna, 3});

            auto sArray =
                py::array_t<std::complex<T>, py::array::f_style | py::array::forcecast>(s);
            check_2d_array(sArray, {nBeam, nBeam});

            auto eigMaskFuncLambda = [&](std::size_t idxBin, std::size_t nEig, T* d) {
              py::array_t<T> dArray(nEig);
              std::memcpy(dArray.mutable_data(0), d, nEig * sizeof(T));

              py::array_t<T, py::array::f_style | py::array::forcecast> dNew(
                  eigMaskFunc(idxBin, dArray));
              check_1d_array(dNew, nEig);

              std::memcpy(d, dNew.data(0), nEig * sizeof(T));
            };

            std::get<NufftSynthesis<T>>(plan_).collect(
                nAntenna, nBeam, wl, eigMaskFuncLambda, sArray.data(0),
                safe_cast<std::size_t>(sArray.strides(1) / sArray.itemsize()), wArray.data(0),
                safe_cast<std::size_t>(wArray.strides(1) / wArray.itemsize()), xyzArray.data(0),
                safe_cast<std::size_t>(xyzArray.strides(1) / xyzArray.itemsize()), uvwArray.data(0),
                safe_cast<std::size_t>(uvwArray.strides(1) / uvwArray.itemsize()));

          } else {
            throw InternalError();
          }
        },
        plan_);
  }

  auto get() -> py::array {
    return std::visit(
        [&](auto&& arg) -> pybind11::array {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, NufftSynthesis<double>>) {
            py::array_t<double> out({nImages_, nPixel_});
            std::get<NufftSynthesis<double>>(plan_).get(
                out.mutable_data(0), safe_cast<std::size_t>(out.strides(0) / out.itemsize()));
            return out;
          } else if constexpr (std::is_same_v<T, NufftSynthesis<float>>) {
            py::array_t<float> out({nImages_, nPixel_});
            std::get<NufftSynthesis<float>>(plan_).get(
                out.mutable_data(0), safe_cast<std::size_t>(out.strides(0) / out.itemsize()));
            return out;
          } else {
            throw InternalError();
            return py::array_t<double, py::array::f_style>();
          }
        },
        plan_);
  }

  std::variant<std::monostate, NufftSynthesis<float>, NufftSynthesis<double>> plan_;
  std::size_t nImages_, nPixel_;
};

struct CompileConfig {
#ifdef BIPP_CUDA
  const bool cuda = true;
#else
  const bool cuda = false;
#endif
#ifdef BIPP_ROCM
  const bool rocm = true;
#else
  const bool rocm = false;
#endif
#ifdef BIPP_UMPIRE
  const bool umpire = true;
#else
  const bool umpire = false;
#endif
#ifdef BIPP_OMP
  const bool omp = true;
#else
  const bool omp = false;
#endif
#ifdef BIPP_MPI
  const bool mpi = true;
#else
  const bool mpi = false;
#endif
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

  pybind11::class_<CompileConfig>(m, "CompileConfig")
      .def_readonly("cuda", &CompileConfig::cuda)
      .def_readonly("rocm", &CompileConfig::rocm)
      .def_readonly("umpire", &CompileConfig::umpire)
      .def_readonly("omp", &CompileConfig::omp)
      .def_readonly("mpi", &CompileConfig::mpi);

  m.attr("config") = CompileConfig();

  pybind11::class_<Context>(m, "Context")
      .def(py::init(&create_context), pybind11::arg("pu"))
      .def(py::init(&create_distributed_context), pybind11::arg("pu"), pybind11::arg("comm"))
      .def_property_readonly("processing_unit",
           [](const Context& ctx) { return processing_unit_to_string(ctx.processing_unit()); })
      .def("attach_non_root", &Context::attach_non_root);

  pybind11::class_<Communicator>(m, "communicator")
      .def_static("world", &Communicator::world)
      .def_static("local", &Communicator::local)
      .def_property_readonly("is_root", &Communicator::is_root)
      .def_property_readonly("size", &Communicator::size)
      .def_property_readonly("rank", &Communicator::rank);

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
      .def("set_local_uvw_partition", &NufftSynthesisOptions::set_local_uvw_partition)
      .def_readwrite("normalizeImage", &NufftSynthesisOptions::normalizeImage)
      .def("set_normalize_image", &NufftSynthesisOptions::set_normalize_image)
      .def_readwrite("normalizeImageNvis", &NufftSynthesisOptions::normalizeImageNvis)
      .def("set_normalize_image_by_nvis", &NufftSynthesisOptions::set_normalize_image_by_nvis);

  pybind11::class_<StandardSynthesisOptions>(m, "StandardSynthesisOptions")
      .def(py::init())
      .def_readwrite("collect_group_size", &StandardSynthesisOptions::collectGroupSize)
      .def("set_collect_group_size", &StandardSynthesisOptions::set_collect_group_size)
      .def_readwrite("normalizeImage", &StandardSynthesisOptions::normalizeImage)
      .def("set_normalize_image", &StandardSynthesisOptions::set_normalize_image)
      .def_readwrite("normalizeImageNvis", &StandardSynthesisOptions::normalizeImageNvis)
      .def("set_normalize_image_by_nvis", &StandardSynthesisOptions::set_normalize_image_by_nvis);

  pybind11::class_<NufftSynthesisDispatcher>(m, "NufftSynthesis")
      .def(pybind11::init<Context&, NufftSynthesisOptions, std::size_t, const py::array&,
                          const py::array&, const py::array&, const std::string&>(),
           pybind11::arg("ctx"), pybind11::arg("opt"), pybind11::arg("n_level"),
           pybind11::arg("lmn_x"), pybind11::arg("lmn_y"), pybind11::arg("lmn_y"),
           pybind11::arg("precision"))
      .def("collect", &NufftSynthesisDispatcher::collect, pybind11::arg("wl"),
           pybind11::arg("mask"), pybind11::arg("s"), pybind11::arg("w"), pybind11::arg("xyz"),
           pybind11::arg("uvw"))
      .def("get", &NufftSynthesisDispatcher::get);

  pybind11::class_<StandardSynthesisDispatcher>(m, "StandardSynthesis")
      .def(pybind11::init<Context&, StandardSynthesisOptions, std::size_t, const py::array&,
                          const py::array&, const py::array&, const std::string&>(),
           pybind11::arg("ctx"), pybind11::arg("opt"), pybind11::arg("n_level"),
           pybind11::arg("lmn_x"), pybind11::arg("lmn_y"), pybind11::arg("lmn_y"),
           pybind11::arg("precision"))
      .def("collect", &StandardSynthesisDispatcher::collect, pybind11::arg("wl"),
           pybind11::arg("mask"), pybind11::arg("s"), pybind11::arg("w"), pybind11::arg("xyz"))
      .def("get", &StandardSynthesisDispatcher::get);

  pybind11::class_<DatasetCreatorDispatcher>(m, "DatasetCreator")
      .def(pybind11::init<const std::string&, const std::string&, std::size_t, std::size_t>(),
           pybind11::arg("file_name"), pybind11::arg("description"), pybind11::arg("n_antenna"),
           pybind11::arg("n_beam"))
      .def("process_and_write", &DatasetCreatorDispatcher::process_and_write,
           pybind11::arg("precision"), pybind11::arg("wl"), pybind11::arg("s"), pybind11::arg("w"),
           pybind11::arg("xyz"), pybind11::arg("uvw"));

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
}
