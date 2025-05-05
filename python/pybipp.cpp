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
#include <string>

#include <unordered_map>
#include <map>

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

auto precision_to_string(BippPrecision prec) -> std::string {
  if (prec == BIPP_PRECISION_SINGLE) return "single";
  if (prec == BIPP_PRECISION_DOUBLE) return "double";

  throw InvalidParameterError();
}

auto string_to_precision(const std::string& prec) -> BippPrecision {
  if (prec == "SINGLE" || prec == "single") return BIPP_PRECISION_SINGLE;
  if (prec == "DOUBLE" || prec == "double") return BIPP_PRECISION_DOUBLE;

  throw InvalidParameterError();
}

struct DatasetFileDispatcher {
  DatasetFileDispatcher(DatasetFile file) : file_(std::move(file)) {}

  static DatasetFileDispatcher open(const std::string& fileName) {
    return DatasetFileDispatcher(DatasetFile::open(fileName));
  }

  static DatasetFileDispatcher create(const std::string& fileName, const std::string& description,
                        std::size_t nAntenna, std::size_t nBeam) {
    return DatasetFileDispatcher(DatasetFile::create(fileName, description, nAntenna, nBeam));
  }

  const std::string& description() const { return file_.description(); }

  std::size_t num_samples() const {return file_.num_samples();}

  std::size_t num_antenna() const { return file_.num_antenna(); }

  std::size_t num_beam() const { return file_.num_beam(); }

  py::array eig_vec(std::size_t index) {
    py::array_t<std::complex<float>, py::array::f_style | py::array::forcecast> out(
        {file_.num_antenna(), file_.num_beam()});

    file_.eig_vec(index, out.mutable_data(0), out.strides(1) / out.itemsize());
    return out;
  }

  py::array eig_val(std::size_t index) {
    py::array_t<float> out(file_.num_beam());

    file_.eig_val(index, out.mutable_data(0));
    return out;
  }

  py::array uvw(std::size_t index) {
    py::array_t<float, py::array::f_style | py::array::forcecast> out(
        {file_.num_antenna() * file_.num_antenna(), std::size_t(3)});

    file_.uvw(index, out.mutable_data(0), out.strides(1) / out.itemsize());
    return out;
  }

  float wl(std::size_t index) { return file_.wl(index); }

  float scale(std::size_t index) { return file_.scale(index); }

  void close() { file_.close(); }

  bool is_open() const noexcept { return file_.is_open(); }

  auto write(float timeStamp, float wl, float scale,
             const py::array_t<std::complex<float>, py::array::f_style | py::array::forcecast>& v,
             const py::array_t<float, py::array::f_style | py::array::forcecast>& d,
             const py::array_t<float, py::array::f_style | py::array::forcecast>& uvw) -> void {
    long nAntenna = file_.num_antenna();
    long nBeam = file_.num_beam();
    check_2d_array(v, {nAntenna, nBeam});
    check_2d_array(uvw, {nAntenna * nAntenna, 3});
    check_1d_array(d, nBeam);

    file_.write(timeStamp, wl, scale, v.data(0),
                safe_cast<std::size_t>(v.strides(1) / v.itemsize()), d.data(0), uvw.data(0),
                safe_cast<std::size_t>(uvw.strides(1) / uvw.itemsize()));
  }

  DatasetFile file_;
};

void image_synthesis_dispatch(
    Context& ctx, const NufftSynthesisOptions& opt, DatasetFileDispatcher& dataset,
    std::unordered_map<std::string, std::unordered_map<std::size_t, std::vector<float>>>
        pySelection,
    ImagePropFile& imagePropFile, const std::string& imageFileName) {
  // check selection sizes and convert
  std::unordered_map<std::string, std::vector<std::pair<std::size_t, const float*>>> selection;
  {
    const auto nBeam = dataset.file_.num_beam();
    for(const auto& [tag, sel] : pySelection) {
      auto& list = selection[tag];
      for(const auto&[id, eigVals] : sel) {
        if (eigVals.size() != nBeam)
          throw InvalidParameterError(
              "Number of eigenvalues in selection does not match number of beams.");

        list.emplace_back(id, eigVals.data());
      }
    }

    image_synthesis(ctx, opt, dataset.file_, selection, imagePropFile, imageFileName);
  }
}

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

auto create_context(const std::string& pu) -> Context {
  return Context(string_to_processing_unit(pu));
}

auto create_distributed_context(const std::string& pu, Communicator comm) -> Context {
  return Context(string_to_processing_unit(pu), std::move(comm));
}

template <typename T>
auto call_eigh(T wl, const py::array_t<std::complex<T>, py::array::f_style>& s,
               const py::array_t<std::complex<T>, py::array::f_style>& w)
    -> std::tuple<py::array_t<std::complex<T>, py::array::f_style>,
                  py::array_t<T, py::array::f_style>, T> {
  check_2d_array(w);
  auto nAntenna = w.shape(0);
  auto nBeam = w.shape(1);
  check_2d_array(s, {nBeam, nBeam});

  auto d = py::array_t<T, py::array::f_style>({py::ssize_t(nBeam)});
  auto v = py::array_t<std::complex<T>, py::array::f_style>({py::ssize_t(nAntenna), py::ssize_t(nBeam)});
  auto pev =
      eigh<T>(wl, nAntenna, nBeam, s.data(0), safe_cast<std::size_t>(s.strides(1) / s.itemsize()),
              w.data(0), safe_cast<std::size_t>(w.strides(1) / w.itemsize()), d.mutable_data(0),
              v.mutable_data(0), safe_cast<std::size_t>(v.strides(1) / v.itemsize()));

  return {std::move(v), std::move(d), pev.second};
}

template <typename T>
auto call_eigh_gram(T wl, const py::array_t<std::complex<T>, py::array::f_style>& s,
                    const py::array_t<std::complex<T>, py::array::f_style>& w,
                    const py::array_t<T, py::array::f_style>& xyz)
    -> std::tuple<py::array_t<std::complex<T>, py::array::f_style>,
                  py::array_t<T, py::array::f_style>, T> {
  check_2d_array(w);
  auto nAntenna = w.shape(0);
  auto nBeam = w.shape(1);
  check_2d_array(xyz, {nAntenna, 3});
  check_2d_array(s, {nBeam, nBeam});

  auto d = py::array_t<T, py::array::f_style>({py::ssize_t(nBeam)});
  auto v = py::array_t<std::complex<T>, py::array::f_style>({py::ssize_t(nAntenna), py::ssize_t(nBeam)});
  auto pev =
      eigh_gram<T>(wl, nAntenna, nBeam, s.data(0), safe_cast<std::size_t>(s.strides(1) / s.itemsize()),
              w.data(0), safe_cast<std::size_t>(w.strides(1) / w.itemsize()), xyz.data(0),
              safe_cast<std::size_t>(xyz.strides(1) / xyz.itemsize()), d.mutable_data(0),
              v.mutable_data(0), safe_cast<std::size_t>(v.strides(1) / v.itemsize()));

  return {std::move(v), std::move(d), pev.second};
}

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

  pybind11::class_<Communicator>(m, "communicator")
      .def_static("world", &Communicator::world)
      .def_static("local", &Communicator::local)
      .def_property_readonly("size", &Communicator::size)
      .def_property_readonly("rank", &Communicator::rank);

  pybind11::class_<Context>(m, "Context")
      .def(py::init(&create_context), pybind11::arg("pu"))
      .def(py::init(&create_distributed_context), pybind11::arg("pu"), pybind11::arg("comm"))
      .def_property_readonly("processing_unit", [](const Context& ctx) {
        return processing_unit_to_string(ctx.processing_unit());
      });

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
      .def_readwrite("sample_batch_size", &NufftSynthesisOptions::sampleBatchSize)
      .def("set_sample_batch_size", &NufftSynthesisOptions::set_sample_batch_size)
      .def_readwrite("local_uvw_partition", &NufftSynthesisOptions::localUVWPartition)
      .def("set_local_uvw_partition", &NufftSynthesisOptions::set_local_uvw_partition)
      .def_readwrite("normalize_image", &NufftSynthesisOptions::normalizeImage)
      .def("set_normalize_image", &NufftSynthesisOptions::set_normalize_image)
      .def_readwrite("apply_scaling", &NufftSynthesisOptions::apply_scaling)
      .def("set_apply_scaling", &NufftSynthesisOptions::set_apply_scaling)
      .def_property_readonly(
          "precision",
          [](const NufftSynthesisOptions& opt) { return precision_to_string(opt.precision); })
      .def("set_precision", [](NufftSynthesisOptions& opt, const std::string& prec) {
        opt.set_precision(string_to_precision(prec));
      });

  pybind11::class_<DatasetFileDispatcher>(m, "DatasetFile")
      .def_static("open", &DatasetFileDispatcher::open, pybind11::arg("file_name"))
      .def_static("create", &DatasetFileDispatcher::create, pybind11::arg("file_name"),
                  pybind11::arg("description"), pybind11::arg("n_antenna"), pybind11::arg("n_beam"))
      .def("close", &DatasetFileDispatcher::close)
      .def("is_open", &DatasetFileDispatcher::is_open)
      .def("num_samples", &DatasetFileDispatcher::num_samples)
      .def("num_beam", &DatasetFileDispatcher::num_beam)
      .def("num_antenna", &DatasetFileDispatcher::num_antenna)
      .def("eig_vec", &DatasetFileDispatcher::eig_vec, pybind11::arg("index"))
      .def("eig_val", &DatasetFileDispatcher::eig_val, pybind11::arg("index"))
      .def("uvw", &DatasetFileDispatcher::uvw, pybind11::arg("index"))
      .def("wl", &DatasetFileDispatcher::wl, pybind11::arg("index"))
      .def("scale", &DatasetFileDispatcher::scale, pybind11::arg("index"))
      .def("write", &DatasetFileDispatcher::write, pybind11::arg("time_stamp"), pybind11::arg("wl"),
           pybind11::arg("scale"), pybind11::arg("v"), pybind11::arg("d"), pybind11::arg("uvw"))
      .def("__enter__", [](DatasetFileDispatcher& d) -> DatasetFileDispatcher& { return d; })
      .def(
          "__exit__",
          [](DatasetFileDispatcher& d, const std::optional<pybind11::type>&,
             const std::optional<pybind11::object>&,
             const std::optional<pybind11::object>&) { d.close(); },
          "Close dataset file");

  pybind11::class_<ImagePropFile>(m, "ImagePropFile")
      .def_static("open", &ImagePropFile::open, pybind11::arg("file_name"))
      .def_static(
          "create",
          [](const std::string& fileName,
             const py::array_t<float, py::array::f_style | py::array::forcecast>& lmn)
              -> ImagePropFile {
            auto numPixel = lmn.shape(0);
            check_2d_array(lmn, {numPixel, 3});

            return ImagePropFile::create(fileName, numPixel, lmn.data(0),
                                     safe_cast<std::size_t>(lmn.strides(1) / lmn.itemsize()));
          },
          pybind11::arg("file_name"), pybind11::arg("lmn"))
      .def("close", &ImagePropFile::close)
      .def("is_open", &ImagePropFile::is_open)
      .def("num_pixel", &ImagePropFile::num_pixel)
      .def("meta_data", &ImagePropFile::meta_data)
      .def("set_meta", &ImagePropFile::set_meta, pybind11::arg("name"), pybind11::arg("value"))
      .def(
          "pixel_lmn",
          [](ImagePropFile& f) -> py::array_t<float, py::array::f_style> {
            auto lmn = py::array_t<float, py::array::f_style>(
                {py::ssize_t(f.num_pixel()), py::ssize_t(3)});
            f.pixel_lmn(lmn.mutable_data(0),
                        safe_cast<std::size_t>(lmn.strides(1) / lmn.itemsize()));
            return lmn;
          })
      .def("__enter__", [](ImagePropFile& f) -> ImagePropFile& { return f; })
      .def(
          "__exit__",
          [](ImagePropFile& f, const std::optional<pybind11::type>&,
             const std::optional<pybind11::object>&,
             const std::optional<pybind11::object>&) { f.close(); },
          "Close image file");

  pybind11::class_<ImageDataFile>(m, "ImageDataFile")
      .def_static("open", &ImageDataFile::open, pybind11::arg("file_name"))
      .def_static(
          "create",
          [](const std::string& fileName, std::size_t numPixel) -> ImageDataFile {
            return ImageDataFile::create(fileName, numPixel);
          },
          pybind11::arg("file_name"), pybind11::arg("lmn"))
      .def("close", &ImageDataFile::close)
      .def("is_open", &ImageDataFile::is_open)
      .def("tags", &ImageDataFile::tags)
      .def("num_tags", &ImageDataFile::num_tags)
      .def("num_pixel", &ImageDataFile::num_pixel)
      .def(
          "get",
          [](ImageDataFile& f, const std::string& tag) -> py::array_t<float, py::array::f_style> {
            auto image = py::array_t<float, py::array::f_style>(py::ssize_t(f.num_pixel()));
            f.get(tag, image.mutable_data(0));
            return image;
          },
          pybind11::arg("tag"))
      .def(
          "set",
          [](ImageDataFile& f, const std::string& tag,
             const py::array_t<float, py::array::f_style | py::array::forcecast>& image)
              -> py::array_t<float, py::array::f_style> {
            check_1d_array(image, f.num_pixel());
            f.set(tag, image.data(0));
            return image;
          },
          pybind11::arg("tag"), pybind11::arg("image"))
      .def("__enter__", [](ImageDataFile& f) -> ImageDataFile& { return f; })
      .def(
          "__exit__",
          [](ImageDataFile& f, const std::optional<pybind11::type>&,
             const std::optional<pybind11::object>&,
             const std::optional<pybind11::object>&) { f.close(); },
          "Close image file");

  m.def("image_synthesis", &image_synthesis_dispatch, pybind11::arg("ctx"), pybind11::arg("opt"),
        pybind11::arg("dataset"), pybind11::arg("selection"), pybind11::arg("image_prop"),
        pybind11::arg("image_file_name"));

  m.def(
       "eigh_gram",
       [](float wl, const py::array_t<std::complex<float>, py::array::f_style>& s,
          const py::array_t<std::complex<float>, py::array::f_style>& w,
          const py::array_t<float, py::array::f_style>& xyz) {
         return call_eigh_gram(wl, s, w, xyz);
       },
       pybind11::arg("wl"), pybind11::arg("s"), pybind11::arg("w"), pybind11::arg("xyz"))
      .def(
          "eigh_gram",
          [](double wl, const py::array_t<std::complex<double>, py::array::f_style>& s,
             const py::array_t<std::complex<double>, py::array::f_style>& w,
             const py::array_t<double, py::array::f_style>& xyz) {
            return call_eigh_gram(wl, s, w, xyz);
          },
          pybind11::arg("wl"), pybind11::arg("s"), pybind11::arg("w"), pybind11::arg("xyz"));

  m.def(
       "eigh",
       [](float wl, const py::array_t<std::complex<float>, py::array::f_style>& s,
          const py::array_t<std::complex<float>, py::array::f_style>& w) {
         return call_eigh(wl, s, w);
       },
       pybind11::arg("wl"), pybind11::arg("s"), pybind11::arg("w"))
      .def(
          "eigh",
          [](double wl, const py::array_t<std::complex<double>, py::array::f_style>& s,
             const py::array_t<std::complex<double>, py::array::f_style>& w) {
            return call_eigh(wl, s, w);
          },
          pybind11::arg("wl"), pybind11::arg("s"), pybind11::arg("w"));
}
