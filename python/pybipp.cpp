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

// void center_array(py::array_t<double> input_array) {
//   py::buffer_info buf_info = input_array.request();
//   double *ptr = static_cast<double *>(buf_info.ptr);
//   const auto N = buf_info.shape[0];
//   const auto M = buf_info.shape[1];
//   for (auto j=0; j<M; j++) {
//     double mean = 0.0;
//     for (auto i=j*N; i<(j+1)*N; i++) {
//       mean += ptr[i];
//     }
//     mean /= N;
//     for (auto i=j*N; i<(j+1)*N; i++) {
//       ptr[i] = ptr[i] - mean;
//     }
//   }
// }

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


  auto close() -> void {
    creator_.close();
  }

  auto is_open() -> bool {
    return creator_.is_open();
  }

  DatasetCreator creator_;
};

struct DatasetReaderDispatcher {
  explicit DatasetReaderDispatcher(const std::string& fileName) : reader_(fileName) {}

  const std::string& description() const { return reader_.description(); }

  std::size_t num_samples() const {return reader_.num_samples();}

  std::size_t num_antenna() const { return reader_.num_antenna(); }

  std::size_t num_beam() const { return reader_.num_beam(); }

  py::array read_eig_vec(std::size_t index) {
    py::array_t<std::complex<float>, py::array::f_style> out(
        {reader_.num_antenna(), reader_.num_beam()});

    reader_.read_eig_vec(index, out.mutable_data(0), out.strides(1) / out.itemsize());
    return out;
  }

  py::array read_eig_val(std::size_t index) {
    py::array_t<float> out(reader_.num_beam());

    reader_.read_eig_val(index, out.mutable_data(0));
    return out;
  }

  py::array read_uvw(std::size_t index) {
    py::array_t<float, py::array::f_style> out(
        {reader_.num_antenna() * reader_.num_antenna(), std::size_t(3)});

    reader_.read_uvw(index, out.mutable_data(0), out.strides(1) / out.itemsize());
    return out;
  }

  py::array read_xyz(std::size_t index) {
    py::array_t<float, py::array::f_style> out({reader_.num_antenna(), std::size_t(3)});

    reader_.read_xyz(index, out.mutable_data(0), out.strides(1) / out.itemsize());
    return out;
  }

  float read_wl(std::size_t index) { return reader_.read_wl(index); }

  std::size_t read_n_vis(std::size_t index) { return reader_.read_n_vis(index); }

  void close() { reader_.close(); }

  bool is_open() const noexcept { return reader_.is_open(); }

  DatasetReader reader_;
};

void image_synthesis_dispatch(
    Communicator& comm, const std::string& puName,
    const std::variant<NufftSynthesisOptions, StandardSynthesisOptions>& opt,
    const std::string& datasetFileName,
    std::unordered_map<std::string, std::unordered_map<std::size_t, std::vector<float>>>
        pySelection,
    const std::vector<float> pixelX, const std::vector<float> pixelY, const std::vector<float> pixelZ,
    const std::string& outputFileName) {

  const auto pu = string_to_processing_unit(puName);
  // check selection sizes and convert
  std::unordered_map<std::string, std::vector<std::pair<std::size_t, const float*>>> selection;
  {
    DatasetReader reader(datasetFileName);
    const auto nBeam = reader.num_beam();
    for(const auto& [tag, sel] : pySelection) {
      auto& list = selection[tag];
      for(const auto&[id, eigVals] : sel) {
        if (eigVals.size() != nBeam)
          throw InvalidParameterError(
              "Number of eigenvalues in selection does not match number of beams.");

        list.emplace_back(id, eigVals.data());
      }
    }

    image_synthesis(comm, pu, opt, datasetFileName, selection, pixelX.size(), pixelX.data(),
                    pixelY.data(), pixelZ.data(), outputFileName);
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

  pybind11::class_<DatasetCreatorDispatcher>(m, "DatasetCreator")
      .def(pybind11::init<const std::string&, const std::string&, std::size_t, std::size_t>(),
           pybind11::arg("file_name"), pybind11::arg("description"), pybind11::arg("n_antenna"),
           pybind11::arg("n_beam"))
      .def("process_and_write", &DatasetCreatorDispatcher::process_and_write,
           pybind11::arg("precision"), pybind11::arg("wl"), pybind11::arg("s"), pybind11::arg("w"),
           pybind11::arg("xyz"), pybind11::arg("uvw"))
      .def("close", &DatasetCreatorDispatcher::close)
      .def("is_open", &DatasetCreatorDispatcher::is_open)
      .def("__enter__", [](DatasetCreatorDispatcher& d) -> DatasetCreatorDispatcher& { return d; })
      .def(
          "__exit__",
          [](DatasetCreatorDispatcher& d, const std::optional<pybind11::type>&,
             const std::optional<pybind11::object>&,
             const std::optional<pybind11::object>&) { d.close(); },
          "Close dataset file");

  pybind11::class_<DatasetReaderDispatcher>(m, "DatasetReader")
      .def(pybind11::init<const std::string&>(), pybind11::arg("file_name"))
      .def("close", &DatasetReaderDispatcher::close)
      .def("is_open", &DatasetReaderDispatcher::is_open)
      .def("num_samples", &DatasetReaderDispatcher::num_samples)
      .def("num_beam", &DatasetReaderDispatcher::num_beam)
      .def("num_antenna", &DatasetReaderDispatcher::num_antenna)
      .def("read_eig_vec", &DatasetReaderDispatcher::read_eig_vec, pybind11::arg("index"))
      .def("read_eig_val", &DatasetReaderDispatcher::read_eig_val, pybind11::arg("index"))
      .def("read_uvw", &DatasetReaderDispatcher::read_uvw, pybind11::arg("index"))
      .def("read_xyz", &DatasetReaderDispatcher::read_xyz, pybind11::arg("index"))
      .def("read_wl", &DatasetReaderDispatcher::read_wl, pybind11::arg("index"))
      .def("read_n_vis", &DatasetReaderDispatcher::read_n_vis, pybind11::arg("index"))
      .def("__enter__", [](DatasetReaderDispatcher& d) -> DatasetReaderDispatcher& { return d; })
      .def(
          "__exit__",
          [](DatasetReaderDispatcher& d, const std::optional<pybind11::type>&,
             const std::optional<pybind11::object>&,
             const std::optional<pybind11::object>&) { d.close(); },
          "Close dataset file");

  m.def("image_synthesis", &image_synthesis_dispatch);
}
