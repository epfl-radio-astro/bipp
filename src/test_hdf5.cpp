#include <complex>
#include <iostream>

#include "context_internal.hpp"
#include "io/dataset_writer.hpp"
#include "io/dataset_reader.hpp"
#include "memory/array.hpp"

using namespace bipp;

int main() {
  std::size_t nAntenna = 10;
  std::size_t nBeam = 4;
  std::size_t nSamples = 5;

  // write
  {
    DatasetWriter writer("data.h5", "this_file", nAntenna, nBeam);

    ContextInternal ctx(BIPP_PU_CPU);

    HostArray<std::complex<float>, 2> v(ctx.host_alloc(), {nAntenna, nBeam});
    HostArray<float, 1> d(ctx.host_alloc(), nBeam);
    HostArray<float, 2> uvw(ctx.host_alloc(), {nAntenna * nAntenna, 3});
    HostArray<float, 2> xyz(ctx.host_alloc(), {nAntenna, 3});

    for (std::size_t i = 0; i < nBeam; ++i) {
      d[i] = i;
    }

    for (std::size_t i = 0; i < nAntenna * nAntenna; ++i) {
      uvw[{i, 0}] = i;
      uvw[{i, 1}] = i;
      uvw[{i, 2}] = i;
    }

    for (std::size_t i = 0; i < nBeam; ++i) {
      for (std::size_t j = 0; j < nAntenna; ++j) {
        v[{j, i}] = i * nAntenna + j;
      }
    }

    for (std::size_t r = 0; r < nSamples; ++r) {
      writer.write(1.0f, nAntenna * nBeam, v, d, uvw, xyz);
    }
  }


  // read
  {
    DatasetReader reader("data.h5");

    ContextInternal ctx(BIPP_PU_CPU);

    HostArray<std::complex<float>, 2> v(ctx.host_alloc(), {nAntenna, nBeam});
    HostArray<float, 1> d(ctx.host_alloc(), nBeam);
    HostArray<float, 2> uvw(ctx.host_alloc(), {nAntenna * nAntenna, 3});

    std::cout << "num_samples = " << reader.num_samples() << std::endl;
    std::cout << "description = " << reader.description() << std::endl;

    for (std::size_t index = 0; index < nSamples; ++index) {
      reader.read_eig_val(index, d);
      reader.read_eig_vec(index, v);
      reader.read_uvw(index, uvw);
      const auto wl = reader.read_wl(index);
      const auto nVis = reader.read_n_vis(index);
    }

    // std::cout << "d: " << std::endl;
    // for (std::size_t i = 0; i < nBeam; ++i) {
    //   std::cout << d[i] <<", ";
    // }
    // std::cout << std::endl;

    // std::cout << "v: " << std::endl;
    // for (std::size_t i = 0; i < nBeam; ++i) {
    //   for (std::size_t j = 0; j < nAntenna; ++j) {
    //     std::cout << v[{j, i}] << ", ";
    //   }
    //   std::cout << std::endl;
    // }

    // std::cout << "w: " << std::endl;
    // for (std::size_t i = 0; i < nAntenna * nAntenna; ++i) {
    //   std::cout << uvw[{i, 2}] <<", ";
    // }
    // std::cout << std::endl;

  }


  return 0;
}
