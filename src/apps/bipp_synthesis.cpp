#include <CLI/CLI.hpp>
#include <deque>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "bipp/bipp.hpp"
#include "bipp/communicator.hpp"

int main(int argc, char** argv) {
  std::string procName, datasetFileName, selectionFileName, imageFileName;

  CLI::App app{"bipp image synthesis"};
  app.add_option("-i", datasetFileName, "Input dataset file name")->required();
  app.add_option("-s", selectionFileName, "Selection file name");
  app.add_option("-o", imageFileName, "Output image file name")->required();
  app.add_option("-p", procName, "Processing unit")
      ->check(CLI::IsMember({"auto", "cpu", "gpu"}))
      ->default_val("auto");
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
  if(procName == "cpu") {
    pu = BIPP_PU_CPU;
  } else if (procName == "gpu") {
    pu = BIPP_PU_GPU;
  }

  auto comm = bipp::Communicator::world();

  bipp::Context ctx(pu, comm);

  bipp::image_synthesis(ctx, bipp::NufftSynthesisOptions(), dataset, selection, image);

  return 0;
}
