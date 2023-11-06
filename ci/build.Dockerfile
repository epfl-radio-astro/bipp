ARG BASE_IMAGE
FROM $BASE_IMAGE
COPY . /src

WORKDIR /src
RUN mkdir -p /src/build

RUN source /etc/profile.d/z10_spack_environment.sh \
 && source /etc/profile.d/z11_paths.sh \
 && echo "${PATH}" \
 && echo "${PYTHON_PATH}" \
 && echo "${CMAKE_PREFIX_PATH}" \
 && cd /src/build \
 && cmake .. -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch} -DBIPP_GPU=CUDA -DBIPP_BUILD_TESTS=ON -DCMAKE_CXX_FLAGS="-march=${cpu_arch}"\
 && make -j 8 \

RUN echo "export PYTHONPATH=\$PYTHONPATH:/src/build/python" >> /etc/profile.d/z11_paths.sh

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l", "-c"]
