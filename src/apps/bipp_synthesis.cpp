#include <CLI/CLI.hpp>
#include <deque>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include <cstddef>
#include <optional>

#include "bipp/bipp.hpp"
#include "bipp/communicator.hpp"

int main(int argc, char** argv) {
  std::string procName, datasetFileName, selectionFileName, imageFileName, precision;
  std::optional<std::array<std::size_t, 3>> uvwPartition, imagePartition;
  float tolerance = 1e-3;

  CLI::App app{"bipp image synthesis"};
  app.add_option("-d,--datset", datasetFileName, "Dataset file name")->required();
  app.add_option("-s,--selection", selectionFileName, "Selection file name");
  app.add_option("-i,--image", imageFileName, "Output image file name")->required();
  app.add_option("-t,--tol", tolerance, "NUFFT tolerance")->default_val(1e-3);
  app.add_option("-p,--proc", procName, "Processing unit")
      ->check(CLI::IsMember({"auto", "cpu", "gpu"}))
      ->default_val("auto");
  app.add_option("-f,--float_precision", procName, "Floating point precision")
      ->check(CLI::IsMember({"single", "double"}))
      ->default_val("single");
  app.add_option("--uvw_part", uvwPartition, "UVW partition size for lower memory usage");
  app.add_option("--image_part", imagePartition, "Image partition size for lower memory usage");
  CLI11_PARSE(app, argc, argv);

  auto dataset = bipp::DatasetFile::open(datasetFileName);
  auto image = bipp::ImageFile::open(imageFileName);

  const auto nEig = dataset.num_beam();

  std::unordered_map<std::string, std::vector<std::pair<std::size_t, const float*>>> selection;
  std::deque<std::vector<float>> selectionData;

  if(selectionFileName.empty()) {
    std::vector<std::pair<std::size_t, const float*>> currentSelection;
    for(std::size_t i = 0; i < dataset.num_samples();++i) {
      selectionData.emplace_back(nEig);
      dataset.eig_val(i, selectionData.back().data());
      currentSelection.emplace_back(i, selectionData.back().data());
    }
    selection.emplace("full_image", currentSelection);
  } else {
    std::ifstream jsonFile(selectionFileName);
    nlohmann::json j;
    jsonFile >> j;

    for(const auto& [tag, values] : j.items()) {
      std::vector<std::pair<std::size_t, const float*>> currentSelection;

      for (const auto& [strIndex, d] : values.items()) {
        if (!d.is_array() || nEig != d.size()) {
          throw bipp::FileError(
              "JSON input error: expected array of size equal to number of beams in dataset");
        }
        std::size_t index = std::stoull(strIndex);
        if (index >= dataset.num_samples()) {
          throw bipp::FileError("JSON input error: sample id exceeds number of samples in dataset");
        }

        selectionData.emplace_back(nEig);
        d.get_to(selectionData.back());
        currentSelection.emplace_back(index, selectionData.back().data());
      }

      selection.emplace(tag, std::move(currentSelection));
    }
  }

  auto pu = BIPP_PU_AUTO;
  if (procName == "cpu") {
    pu = BIPP_PU_CPU;
  } else if (procName == "gpu") {
    pu = BIPP_PU_GPU;
  }

  auto comm = bipp::Communicator::world();

  bipp::Context ctx(pu, comm);

  bipp::NufftSynthesisOptions opt;
  opt.tolerance = tolerance;
  if(uvwPartition.has_value()) {
    auto& p = uvwPartition.value();
    opt.localUVWPartition =
        bipp::Partition::Grid{p[0], p[1], p[2]};
  }

  if(imagePartition.has_value()) {
    auto& p = imagePartition.value();
    opt.localImagePartition = bipp::Partition::Grid{p[0], p[1], p[2]};
  }

  if (precision == "double") {
    opt.precision = BIPP_PRECISION_DOUBLE;
  } else {
    opt.precision = BIPP_PRECISION_SINGLE;
  }

  bipp::image_synthesis(ctx, opt, dataset, selection, image);

  return 0;
}
