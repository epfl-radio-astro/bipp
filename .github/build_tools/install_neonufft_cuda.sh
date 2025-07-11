#!/bin/bash
export CUDACXX=/usr/local/cuda/bin/nvcc
cd ${HOME}
git clone https://github.com/epfl-radio-astro/neonufft.git
cd neonufft && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DNEONUFFT_THREADING=OPENMP -DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS} -DNEONUFFT_GPU=CUDA -DBUILD_SHARED_LIBS=OFF -DHWY_ENABLE_INSTALL=ON
make install -j4

