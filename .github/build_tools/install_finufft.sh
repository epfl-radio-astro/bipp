#!/bin/bash
yum install -y openblas-openmp fftw-devel wget
cd ${HOME}
wget --quiet https://github.com/flatironinstitute/finufft/archive/refs/tags/v2.2.0.tar.gz
tar -xzvf v2.2.0.tar.gz
cd finufft-2.2.0 && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DFINUFFT_ARCH_FLAGS=""
make install -j

