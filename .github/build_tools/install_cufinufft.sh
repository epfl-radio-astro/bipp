#!/bin/bash
git clone https://github.com/AdhocMan/cufinufft.git
cd ${HOME}/cufinufft && git checkout t3_d3
export CFLAGS="-fPIC -O3 -funroll-loops"
export CUDACXX=/usr/local/cuda/bin/nvcc
export NVCC=$CUDACXX

# iterate through ";" separated list of CUDAARCHS
IFS=';' read -ra ADDR <<< "$CUDAARCHS"
NVARCH=""
for i in "${ADDR[@]}"; do
    NVARCH="$NVARCH -gencode=arch=compute_${i},code=sm_${i}"
done
export NVARCH=$NVARCH

make lib -j

