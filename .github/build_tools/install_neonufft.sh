#!/bin/bash
cd ${HOME}
git clone https://github.com/epfl-radio-astro/neonufft.git
cd neonufft && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DNEONUFFT_THREADING=NATIVE -DBUILD_SHARED_LIBS=OFF
make install -j

