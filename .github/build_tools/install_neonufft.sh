#!/bin/bash
cd ${HOME}
git clone https://github.com/epfl-radio-astro/neonufft.git
cd neonufft && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DNEONUFFT_THREADING=OPENMP -DBUILD_SHARED_LIBS=OFF -DHWY_ENABLE_INSTALL=ON
make install -j4

