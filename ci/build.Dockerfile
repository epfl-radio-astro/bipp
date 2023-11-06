ARG BASE_IMAGE
FROM $BASE_IMAGE
COPY . /src

WORKDIR /src
RUN mkdir -p /src/build

RUN echo "${PATH}"
RUN echo "${PYTHON_PATH}"
RUN echo "${CMAKE_PREFIX_PATH}"
RUN which cmake

WORKDIR /src/build
RUN cmake .. -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch} -DBIPP_GPU=CUDA -DBIPP_BUILD_TESTS=ON -DCMAKE_CXX_FLAGS="-march=${cpu_arch}"\
&& make -j 8 \

RUN echo "export PYTHONPATH=\$PYTHONPATH:/src/build/python" >> /etc/profile.d/z11_paths.sh

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l", "-c"]
