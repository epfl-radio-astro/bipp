ARG BASE_IMAGE
FROM $BASE_IMAGE
COPY . /src

WORKDIR /src
RUN mkdir -p /src/build_cpu
RUN mkdir -p /src/build_cuda

RUN source /etc/profile.d/z10_spack_environment.sh \
 && source /etc/profile.d/z11_paths.sh \
 && cd /src/build_cpu \
 && cmake .. -DBIPP_GPU=OFF -DBIPP_BUILD_TESTS=ON -DCMAKE_CXX_FLAGS="-march=${cpu_arch}" -DBIPP_PYTHON=OFF -DBIPP_OMP=OFF\
 && make -j 8

# build without NDEBUG macro defined to enable assertions,
# so overwrite release flags
RUN source /etc/profile.d/z10_spack_environment.sh \
 && source /etc/profile.d/z11_paths.sh \
 && cd /src/build_cuda \
 && cmake .. -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch} -DBIPP_GPU=CUDA -DBIPP_BUILD_TESTS=ON -DCMAKE_CXX_FLAGS="-march=${cpu_arch}" -DCMAKE_CXX_FLAGS_RELEASE="-O3" -DCMAKE_CUDA_FLAGS_RELEASE="-O3" -DBIPP_PYTHON=OFF\
 && make -j 8

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l", "-c"]
