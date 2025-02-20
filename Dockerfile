# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:25.01-py3
FROM $BASE_CONTAINER as builder

ARG TARGETPLATFORM

# Update pip and setuptools
RUN pip install "pip>=23.2.1" "setuptools>=70.0.0"  

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 && \
    git lfs install

# install vtk
ARG VTK_ARM64_WHEEL
ENV VTK_ARM64_WHEEL=${VTK_ARM64_WHEEL:-unknown}

COPY . /physicsnemo-sym/
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$VTK_ARM64_WHEEL" != "unknown" ]; then \
        echo "VTK wheel $VTK_ARM64_WHEEL for $TARGETPLATFORM exists, installing!" && \
        pip install --no-cache-dir /physicsnemo-sym/deps/${VTK_ARM64_WHEEL}; \
    elif [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        echo "Installing vtk for: $TARGETPLATFORM" && \
        pip install --no-cache-dir "vtk>=9.2.6"; \
    else \
        echo "No custom wheel or wheel on PyPi found. Installing vtk for: $TARGETPLATFORM from source" && \
        apt-get update && apt-get install -y libgl1-mesa-dev && \
        git clone https://gitlab.kitware.com/vtk/vtk.git && cd vtk && git checkout tags/v9.4.1 && git submodule update --init --recursive && \
        mkdir build && cd build && cmake -GNinja -DVTK_WHEEL_BUILD=ON -DVTK_WRAP_PYTHON=ON /workspace/vtk/ && ninja && \
        python setup.py bdist_wheel && \
        pip install --no-cache-dir dist/vtk-*.whl && \
        cd ../../ && rm -r vtk; \
    fi
    
# Install physicsnemo sym dependencies
RUN pip install --no-cache-dir "hydra-core>=1.2.0" "termcolor>=2.1.1" "chaospy>=4.3.7" "Cython>=0.29.28" "numpy-stl>=2.16.3" "opencv-python>=4.8.1.78" \
    "scikit-learn>=1.0.2" "symengine>=0.10.0" "sympy>=1.12" "timm>=1.0.3" "torch-optimizer>=0.3.0" "transforms3d>=0.3.1" \
    "typing>=3.7.4.3" "pillow==10.3.0" "notebook>=7.2.2" "mistune>=2.0.3" "pint>=0.19.2" "tensorboard>=2.8.0" "numpy<2.0"

# Install warp-lang
RUN pip install --no-cache-dir warp-lang

# Install tiny-cuda-nn
ARG TCNN_CUDA_AMD64_WHEEL
ENV TCNN_CUDA_AMD64_WHEEL=${TCNN_CUDA_AMD64_WHEEL:-unknown}

ARG TCNN_CUDA_ARM64_WHEEL
ENV TCNN_CUDA_ARM64_WHEEL=${TCNN_CUDA_ARM64_WHEEL:-unknown}

ENV TCNN_CUDA_ARCHITECTURES="60;70;75;80;86;90"
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] && [ "$TCNN_CUDA_AMD64_WHEEL" != "unknown" ]; then \
        echo "Tiny CUDA NN wheel $TCNN_CUDA_AMD64_WHEEL for $TARGETPLATFORM exists, installing!" && \
        pip install --force-reinstall --no-cache-dir /physicsnemo-sym/deps/$TCNN_CUDA_AMD64_WHEEL; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$TCNN_CUDA_ARM64_WHEEL" != "unknown" ]; then \
        echo "Tiny CUDA NN wheel $TCNN_CUDA_ARM64_WHEEL for $TARGETPLATFORM exists, installing!" && \
        pip install --force-reinstall --no-cache-dir /physicsnemo-sym/deps/$TCNN_CUDA_ARM64_WHEEL; \
    else \
        echo "No Tiny CUDA NN wheel present, building from source" && \
	pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn/@master#subdirectory=bindings/torch; \	
    fi

# CI Image
FROM builder as ci

# Install PhysicsNeMo
RUN pip install --upgrade --no-cache-dir git+https://github.com/NVIDIA/modulus.git@main

RUN pip install --no-cache-dir "black==22.10.0" "interrogate==1.5.0" "coverage==6.5.0"

# Deployment image
FROM builder as deploy

# Install physicsnemo sym
COPY . /physicsnemo-sym/
RUN cd /physicsnemo-sym/ && pip install --no-cache-dir . --no-deps --no-build-isolation
RUN rm -rf /physicsnemo-sym/

# Set Git Hash as a environment variable
ARG PHYSICSNEMO_SYM_GIT_HASH
ENV PHYSICSNEMO_SYM_GIT_HASH=${PHYSICSNEMO_SYM_GIT_HASH:-unknown}

# Docs image
FROM deploy as docs
# Install packages for Sphinx build
RUN pip install --no-cache-dir "recommonmark==0.7.1" "sphinx==5.1.1" "sphinx-rtd-theme==1.0.0" "pydocstyle==6.1.1" "nbsphinx==0.8.9" "nbconvert==6.4.3" "jinja2==3.0.3"
RUN wget https://github.com/jgm/pandoc/releases/download/3.1.2/pandoc-3.1.2-linux-amd64.tar.gz && tar xvzf pandoc-3.1.2-linux-amd64.tar.gz --strip-components 1 -C /usr/local/ 
