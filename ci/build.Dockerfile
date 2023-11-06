ARG BASE_IMAGE
FROM $BASE_IMAGE
COPY . /src

WORKDIR /src
RUN mkdir -p /src/build_cpu
RUN mkdir -p /src/build_cuda

RUN source /etc/profile.d/z10_spack_environment.sh \
 && source /etc/profile.d/z11_paths.sh \
 && cd /src/build_cpu \
 && cmake .. -DBIPP_GPU=OFF -DBIPP_BUILD_TESTS=ON -DCMAKE_CXX_FLAGS="-march=${cpu_arch}"\
 && make -j 8

RUN source /etc/profile.d/z10_spack_environment.sh \
 && source /etc/profile.d/z11_paths.sh \
 && cd /src/build_cuda \
 && cmake .. -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch} -DBIPP_GPU=CUDA -DBIPP_BUILD_TESTS=ON -DCMAKE_CXX_FLAGS="-march=${cpu_arch}"\
 && make -j 8

RUN source /etc/profile.d/z10_spack_environment.sh \
 && source /etc/profile.d/z11_paths.sh \
 && cd /src \
 && BIPP_GPU=CUDA BIPP_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${cuda_arch}" python3 -m pip install .

# RUN echo "export PYTHONPATH=\$PYTHONPATH:/src/build/python" >> /etc/profile.d/z11_paths.sh

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l", "-c"]
