FROM nvcr.io/nvidia/physicsnemo/physicsnemo:22.09

ARG MYUSER=user
ARG MYUID=1000
ARG MYGID=1000
ARG PROCS=8

ARG BUILD_TYPE=Release
ENV DEBIAN_FRONTEND noninteractive

# Install cmake, add user
RUN cd /workspace && \
    if [ -f /usr/bin/yum ]; then yum install -y make wget; fi && \
    if [ -f /usr/bin/apt-get ]; then apt-get update && apt-get install -y apt-utils make wget vim; fi && \
    wget -q https://github.com/Kitware/CMake/releases/download/v3.23.0/cmake-3.23.0-linux-x86_64.tar.gz && \
    tar xf cmake-3.23.0-linux-x86_64.tar.gz && rm cmake-3.23.0-linux-x86_64.tar.gz && \
    groupadd -f -g ${MYGID} ${MYUSER} && \
    useradd -rm -u $MYUID -g $MYUSER -p "" $MYUSER && \
    chown ${MYUSER}:${MYGID} /workspace

RUN cd /workspace && git clone https://github.com/ergosimulation/mpslib.git MPSlib && cd MPSlib && make -j ${PROCS}

# USER $MYUSER
ENV PATH /workspace/cmake-3.23.0-linux-x86_64/bin:$PATH

# Create script to install AMGX
# https://github.com/NVIDIA/AMGX/releases
RUN echo "#!/bin/bash" >> /workspace/install_amgx.sh && \
    echo "cd /workspace/project/" >> /workspace/install_amgx.sh && \
    echo "wget -q https://github.com/NVIDIA/AMGX/archive/refs/tags/v2.2.0.tar.gz" >> /workspace/install_amgx.sh && \
    echo "tar xf v2.2.0.tar.gz && rm v2.2.0.tar.gz" >> /workspace/install_amgx.sh && \
    echo "cd /workspace/project/AMGX-2.2.0/" >> /workspace/install_amgx.sh && \
    echo "rm -rf build/" >> /workspace/install_amgx.sh && \
    echo "mkdir -p build/" >> /workspace/install_amgx.sh && \
    echo "cd build/" >> /workspace/install_amgx.sh && \
    echo "cmake ../ -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCUDA_ARCH=70;86 -DCMAKE_NO_MPI=true" >> /workspace/install_amgx.sh && \
    echo "make -j ${PROCS} all" >> /workspace/install_amgx.sh && \
    chmod 755 /workspace/install_amgx.sh

# Create script to install PyAMGX
RUN echo "#!/bin/bash" >> /workspace/install_pyamgx.sh && \
    echo "cd /workspace/project/" >> /workspace/install_pyamgx.sh && \
    echo "git clone https://github.com/shwina/pyamgx.git" >> /workspace/install_pyamgx.sh && \
    echo "export AMGX_DIR=/workspace/project/AMGX-2.2.0" >> /workspace/install_pyamgx.sh && \
    echo "cd pyamgx" >> /workspace/install_pyamgx.sh && \
    echo "pip install ." >> /workspace/install_pyamgx.sh && \
    chmod 755 /workspace/install_pyamgx.sh

RUN pip3 install tensorflow==2.9.1 scikit-mps kneed pyDOE FyeldGenerator py-cpuinfo gdown gstools scikit-image ema-pytorch accelerate


WORKDIR /workspace/project
# CMD /workspace/compile_amgx.sh

