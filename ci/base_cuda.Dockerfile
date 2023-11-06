# Build stage with Spack pre-installed and ready to be used
FROM docker.io/spack/ubuntu-focal:v0.19.0 as builder
# Cuda architecture list with semicolon separation, see https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
ARG cuda_arch="60;61;70;75;80;86"
# GCC cpu architecture flag, see https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
ARG cpu_arch=haswell
# spack cpu architecture flag, see https://spack.readthedocs.io/en/latest/basic_usage.html#support-for-specific-microarchitectures
ARG spack_arch=${cpu_arch}

# What we want to install and how we want to install it
# is specified in a manifest file (spack.yaml)
RUN mkdir /opt/spack-environment \
&&  (echo "spack:" \
&&   echo "  specs:" \
&&   echo "  - py-numpy@1.21.6 ^python@3.10.6 ^openblas threads=openmp" \
&&   echo "  - py-astropy" \
&&   echo "  - py-matplotlib" \
&&   echo "  - py-tqdm" \
&&   echo "  - py-pyproj" \
&&   echo "  - py-healpy ^libsharp ~mpi" \
&&   echo "  - py-scikit-learn" \
&&   echo "  - py-pandas" \
&&   echo "  - py-scipy" \
&&   echo "  - cmake" \
&&   echo "  - fftw ~mpi +openmp" \
&&   echo "  - vc" \
&&   echo "  - doxygen" \
&&   echo "  - py-breathe" \
&&   echo "  - py-sphinx" \
&&   echo "  - py-pip" \
&&   echo "  - casacore +python" \
&&   echo "  concretizer:" \
&&   echo "    unify: true" \
&&   echo "    targets:" \
&&   echo "      host_compatible: false" \
&&   echo "  config:" \
&&   echo "    install_tree: /opt/software" \
&&   echo "  view: /opt/view" \
&&   echo "  packages:" \
&&   echo "    all:" \
&&   echo "      target: [${spack_arch}]") > /opt/spack-environment/spack.yaml

# Install the software, remove unnecessary deps
RUN cd /opt/spack-environment && \
    spack env activate . && \
    spack install --fail-fast && \
    spack gc -y

# Strip all the binaries
RUN find -L /opt/view/* -type f -exec readlink -f '{}' \; | \
    xargs file -i | \
    grep 'charset=binary' | \
    grep 'x-executable\|x-archive\|x-sharedlib' | \
    awk -F: '{print $1}' | xargs strip -s

# Modifications to the environment that are necessary to run
RUN cd /opt/spack-environment && \
    spack env activate --sh -d . >> /etc/profile.d/z10_spack_environment.sh

# Bare OS image to run the installed executables
FROM docker.io/nvidia/cuda:11.1.1-devel-ubuntu20.04
ARG cuda_arch="60;61;70;75;80;86"
ARG cpu_arch=haswell
ARG spack_arch=${cpu_arch}

ENV cuda_arch=${cuda_arch}
ENV cpu_arch=${cpu_arch}
ENV spack_arch=${spack_arch}

COPY --from=builder /opt/spack-environment /opt/spack-environment
COPY --from=builder /opt/software /opt/software
COPY --from=builder /opt/view /opt/view
COPY --from=builder /etc/profile.d/z10_spack_environment.sh /etc/profile.d/z10_spack_environment.sh

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y gcc g++ gfortran git

ENV LIBRARY_PATH=${LIBRARY_PATH}:/opt/view/lib:/opt/view/lib64
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/view/lib:/opt/view/lib64

WORKDIR /project

RUN . /etc/profile \
RUN export CPATH=$CPATH:/opt/view/include \
&& git clone https://github.com/flatironinstitute/finufft.git \
&& export CFLAGS="-O3 -funroll-loops -march=${cpu_arch} -fcx-limited-range -fPIC  -I/opt/view/include" \
&& cd finufft \
&& git checkout v2.1.0 \
&& (echo "diff --git a/makefile b/makefile" \
&& echo "index 37c289a..169869c 100644" \
&& echo "--- a/makefile" \
&& echo "+++ b/makefile" \
&& echo "@@ -26,9 +26,9 @@ PYTHON = python3" \
&& echo " # Notes: 1) -Ofast breaks isfinite() & isnan(), so use -O3 which now is as fast" \
&& echo " #        2) -fcx-limited-range for fortran-speed complex arith in C++" \
&& echo " #        3) we use simply-expanded (:=) makefile variables, otherwise confusing" \
&& echo "-CFLAGS := -O3 -funroll-loops -march=native -fcx-limited-range \$(CFLAGS)" \
&& echo "-FFLAGS := \$(CFLAGS) \$(FFLAGS)" \
&& echo "-CXXFLAGS := \$(CFLAGS) \$(CXXFLAGS)" \
&& echo "+CFLAGS ?= -O3 -funroll-loops -march=native -fcx-limited-range" \
&& echo "+FFLAGS ?= \$(CFLAGS)" \
&& echo "+CXXFLAGS ?= \$(CFLAGS)" \
&& echo " # put this in your make.inc if you have FFTW>=3.3.5 and want thread-safe use..." \
&& echo " #CXXFLAGS += -DFFTW_PLAN_SAFE" \
&& echo " # FFTW base name, and math linking...") > patch_flags.patch \
&& git apply --reject patch_flags.patch \
&& make lib VERBOSE=1 -j4 \
&& echo "export PYTHONPATH=\$PYTHONPATH:/project/finufft/python" >> /etc/profile.d/z11_paths.sh \
&& echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/project/finufft/lib" >> /etc/profile.d/z11_paths.sh \
&& echo "export CMAKE_PREFIX_PATH=\"\$CMAKE_PREFIX_PATH:/project/finufft\"" >> /etc/profile.d/z11_paths.sh


WORKDIR /project
RUN . /etc/profile \
&& export NVARCH="" \
&& for i in ${cuda_arch//;/ }; do export NVARCH="-gencode=arch=compute_${i},code=sm_${i} ${NVARCH}"; done \
&& export NVCC=/usr/local/cuda/bin/nvcc \
# && export CFLAGS="-O3 -funroll-loops -march=${cpu_arch} -fcx-limited-range -fPIC" \
&& git clone https://github.com/AdhocMan/cufinufft.git \
&& cd cufinufft \
&& git checkout t3_d3 \
&& make lib -j4 VERBOSE=1 \
&& echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/project/cufinufft/lib" >> /etc/profile.d/z11_paths.sh \
&& echo "export CMAKE_PREFIX_PATH=\"\$CMAKE_PREFIX_PATH:/project/cufinufft\"" >> /etc/profile.d/z11_paths.sh


# install furo sphinx theme for documentation
WORKDIR /project
RUN /opt/view/bin/python3.10 -m pip install furo

# make default python commands available. Spack seems to set wrong symbolic links
RUN cd /opt/view/bin \
&& rm python python3 python-config python3-config \
&& ln -s /opt/view/bin/python3.10 python \
&& ln -s /opt/view/bin/python3.10 python3 \
&& ln -s /opt/view/bin/python3.10-config python-config \
&& ln -s /opt/view/bin/python3.10-config python3-config


WORKDIR /project
RUN . /etc/profile \
&& git clone https://github.com/casacore/python-casacore.git \
&& cd python-casacore \
&& git checkout v3.4.0 \
&& python3 -m pip install .

RUN echo "export PATH=\$PATH:/usr/local/cuda/bin" >> /etc/profile.d/z11_paths.sh

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l", "-c"]
