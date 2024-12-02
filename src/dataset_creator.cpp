#include "bipp/dataset_creator.hpp"

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "context_internal.hpp"
#include "host/eigensolver.hpp"
#include "io/dataset_file_writer.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"
#include "memory/copy.hpp"

namespace bipp {
class DatasetCreator::DatasetCreatorImpl {
public:
  DatasetCreatorImpl(const std::string& fileName, const std::string& description,
                     std::size_t nAntenna, std::size_t nBeam, BippProcessingUnit pu = BIPP_PU_AUTO)
      : nAntenna_(nAntenna),
        nBeam_(nBeam),
        ctx_(pu),
        writer_(fileName, description, nAntenna, nBeam) {}

  void process_and_write(float wl, const std::complex<float>* s, std::size_t lds,
                         const std::complex<float>* w, std::size_t ldw, const float* xyz,
                         std::size_t ldxyz, const float* uvw, std::size_t lduvw) {
    if (ctx_.processing_unit() == BIPP_PU_GPU) {
      throw NotImplementedError();
    } else {
      HostArray<std::complex<float>, 2> vArray(ctx_.host_alloc(), {nAntenna_, nBeam_});
      HostArray<float, 1> dArray(ctx_.host_alloc(), nBeam_);

      auto sView = ConstHostView<std::complex<float>, 2>(s, {nBeam_, nBeam_}, {1, lds});
      auto wView = ConstHostView<std::complex<float>, 2>(w, {nAntenna_, nBeam_}, {1, ldw});
      auto xyzView = ConstHostView<float, 2>(xyz, {nAntenna_, 3}, {1, ldxyz});
      auto uvwView = ConstHostView<float, 2>(uvw, {nAntenna_ * nAntenna_, 3}, {1, lduvw});

      const auto pev = host::eigh<float>(ctx_, wl, sView, wView, xyzView, dArray, vArray);
      const auto nVis = pev.second;

      // input to writer must be contigious
      HostArray<float, 2> uvwArray;
      if (!uvwView.is_contiguous()) {
        uvwArray = HostArray<float, 2>(ctx_.host_alloc(), uvwView.shape());
        copy(uvwView, uvwArray);
        uvwView = uvwArray;
      }

      HostArray<float, 2> xyzArray;
      if (!xyzView.is_contiguous()) {
        xyzArray = HostArray<float, 2>(ctx_.host_alloc(), xyzView.shape());
        copy(xyzView, xyzArray);
        xyzView = xyzArray;
      }

      writer_.write(wl, nVis, vArray, dArray, uvwView, xyzView);
    }
  }

  void process_and_write(double wl, const std::complex<double>* s, std::size_t lds,
                         const std::complex<double>* w, std::size_t ldw, const double* xyz,
                         std::size_t ldxyz, const double* uvw, std::size_t lduvw) {
    if (ctx_.processing_unit() == BIPP_PU_GPU) {
      throw NotImplementedError();
    } else {
      HostArray<std::complex<double>, 2> vArray(ctx_.host_alloc(), {nAntenna_, nBeam_});
      HostArray<double, 1> dArray(ctx_.host_alloc(), nBeam_);

      auto sView = ConstHostView<std::complex<double>, 2>(s, {nBeam_, nBeam_}, {1, lds});
      auto wView = ConstHostView<std::complex<double>, 2>(w, {nAntenna_, nBeam_}, {1, ldw});
      auto xyzView = ConstHostView<double, 2>(xyz, {nAntenna_, 3}, {1, ldxyz});
      auto uvwView = ConstHostView<double, 2>(uvw, {nAntenna_ * nAntenna_, 3}, {1, lduvw});

      const auto pev = host::eigh<double>(ctx_, wl, sView, wView, xyzView, dArray, vArray);
      const auto nVis = pev.second;

      // convert to float
      HostArray<std::complex<float>, 2> vArrayFloat(ctx_.host_alloc(), {nAntenna_, nBeam_});
      HostArray<float, 1> dArrayFloat(ctx_.host_alloc(), nBeam_);
      HostArray<float, 2> uvwArrayFloat(ctx_.host_alloc(), uvwView.shape());
      HostArray<float, 2> xyzArrayFloat(ctx_.host_alloc(), xyzView.shape());

      for (std::size_t row = 0; row < vArray.shape(0); ++row) {
        for (std::size_t col = 0; col < vArray.shape(1); ++col) {
          vArrayFloat[{row, col}] = vArray[{row, col}];
        }
      }

      for (std::size_t row = 0; row < uvwView.shape(0); ++row) {
        for (std::size_t col = 0; col < uvwView.shape(1); ++col) {
          uvwArrayFloat[{row, col}] = uvwView[{row, col}];
        }
      }

      for (std::size_t row = 0; row < xyzView.shape(0); ++row) {
        for (std::size_t col = 0; col < xyzView.shape(1); ++col) {
          xyzArrayFloat[{row, col}] = xyzView[{row, col}];
        }
      }

      for (std::size_t row = 0; row < dArray.shape(0); ++row) {
        dArrayFloat[row] = dArray[row];
      }

      writer_.write(wl, nVis, vArrayFloat, dArrayFloat, uvwArrayFloat, xyzArrayFloat);
    }
  }

  void write(float wl, std::size_t nVis, const std::complex<float>* v, std::size_t ldv,
             const float* d, const float* xyz, std::size_t ldxyz, const float* uvw,
             std::size_t lduvw);

  std::size_t num_antenna() const { return nAntenna_; }

  std::size_t num_beam() const { return nBeam_; }

private:
  std::size_t nAntenna_, nBeam_;
  ContextInternal ctx_;
  DatasetFileWriter writer_;
};

void DatasetCreator::DatasetCreatorImplDeleter::operator()(DatasetCreatorImpl* p) { delete p; }

DatasetCreator::DatasetCreator(const std::string& fileName, const std::string& description,
                               std::size_t nAntenna, std::size_t nBeam)
    : impl_(new DatasetCreatorImpl(fileName, description, nAntenna, nBeam)) {}

void DatasetCreator::process_and_write(float wl, const std::complex<float>* s, std::size_t lds,
                                       const std::complex<float>* w, std::size_t ldw,
                                       const float* xyz, std::size_t ldxyz, const float* uvw,
                                       std::size_t lduvw) {
  if (impl_)
    impl_->process_and_write(wl, s, lds, w, ldw, xyz, ldxyz, uvw, lduvw);
  else
    throw GenericError("DatasetCreator: write after close");
}

void DatasetCreator::process_and_write(double wl, const std::complex<double>* s, std::size_t lds,
                                       const std::complex<double>* w, std::size_t ldw,
                                       const double* xyz, std::size_t ldxyz, const double* uvw,
                                       std::size_t lduvw) {
  if (impl_)
    impl_->process_and_write(wl, s, lds, w, ldw, xyz, ldxyz, uvw, lduvw);
  else
    throw GenericError("DatasetCreator: write after close");
}

void DatasetCreator::close() { return impl_.reset(); }

bool DatasetCreator::is_open() const { return bool(impl_); }

std::size_t DatasetCreator::num_antenna() const { return impl_->num_antenna(); }

std::size_t DatasetCreator::num_beam() const { return impl_->num_beam(); }

}  // namespace bipp
