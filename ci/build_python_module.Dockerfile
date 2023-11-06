ARG BASE_IMAGE
FROM $BASE_IMAGE
COPY . /src

RUN source /etc/profile.d/z10_spack_environment.sh \
 && source /etc/profile.d/z11_paths.sh \
 && cd /src \
 && BIPP_GPU=CUDA BIPP_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${cuda_arch} -DCMAKE_CXX_FLAGS=\"-march=${cpu_arch}\"" python3 -m pip install .

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l", "-c"]
