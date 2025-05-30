# PhysicsNeMo Symbolic

<!-- markdownlint-disable -->

[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/physicsnemo)](https://github.com/NVIDIA/physicsnemo/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- markdownlint-enable -->
[**PhysicsNeMo Sym**](#What-is-PhysicsNeMo-Symbolic)
| [**Getting started**](#Getting-started)
| [**Documentation**](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html)
| [**Contributing Guidelines**](#contributing-to-physicsnemo)
| [**Communication**](#communication)

## What is PhysicsNeMo Symbolic?

PhysicsNeMo Symbolic (PhysicsNeMo Sym) is sub-module of PhysicsNeMo framework that provides
algorithms and utilities to explicitly physics inform the
training of AI models. 

Please refer to the [PhysicsNeMo framework](https://github.com/NVIDIA/physicsnemo/blob/main/README.md)
to learn more about the full stack.

This includes utilities for explicitly integrating symbolic PDEs,
domain sampling and computing PDE-based residuals using various gradient computing schemes.

Please refer to the
[Physics informing surrogate model for Darcy flow](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/examples/cfd/darcy_physics_informed/readme.html)
that illustrates the concept.

It also provides an abstraction layer for developers that want to compose a training loop
from specification
of the geometry, PDEs and constraints like boundary conditions using simple symbolic APIs.
Please refer to the
[Lid Driven cavity](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/basics/lid_driven_cavity_flow.html)
that illustrates the concept.

Additional information can be found in the
[PhysicsNeMo documentation](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html).

<!-- markdownlint-enable -->

## Getting started

Please use the getting started guide here for [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/blob/main/README.md#getting-started)

Please refer [Introductory Example](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/darcy_physics_informed)
for usage of the physics utils in custom training loops and
[Lid Driven cavity](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/basics/lid_driven_cavity_flow.html)
for an end-to-end PINN workflow.

## Installation

Please ensure you have installed PhysicsNeMo using the steps [here](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html).

You can then install this package following the steps outlined below:

### PyPi

The recommended method for installing the latest version of PhysicsNeMo Symbolic is
using PyPi:

```bash
pip install "Cython"
pip install nvidia-physicsnemo.sym --no-build-isolation
```

Note, the above method only works for x86/amd64 based architectures. For installing
PhysicsNeMo Sym on Arm based systems using pip,
Install VTK from source as shown
[here](https://gitlab.kitware.com/vtk/vtk/-/blob/v9.2.6/Documentation/dev/build.md?ref_type=tags#python-wheels)
and then install PhysicsNeMo-Sym and other dependencies.

```bash
pip install nvidia-physicsnemo.sym --no-deps
pip install "hydra-core>=1.2.0" "termcolor>=2.1.1" "chaospy>=4.3.7" "Cython==0.29.28" \
    "numpy-stl==2.16.3" "opencv-python==4.5.5.64" "scikit-learn==1.0.2" \
    "symengine>=0.10.0" "sympy==1.12" "timm>=1.0.3" "torch-optimizer==0.3.0" \
    "transforms3d==0.3.1" "typing==3.7.4.3" "pillow==10.0.1" "notebook==6.4.12" \
    "mistune==2.0.3" "pint==0.19.2" "tensorboard>=2.8.0"
```

### Container

The recommended PhysicsNeMo docker image can be pulled from the
[NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/containers/physicsnemo):

```bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:<tag>
```

### From Source

### Package

For a local build of the PhysicsNeMo Symbolic Python package from source use:

```Bash
git clone git@github.com:NVIDIA/physicsnemo-sym.git && cd physicsnemo-sym

pip install --upgrade pip
pip install .
```

### Source Container

To build release image insert next tag and run below:

```bash
docker build -t physicsnemo-sym:deploy \
    --build-arg TARGETPLATFORM=linux/amd64 --target deploy -f Dockerfile .
```

Currently only `linux/amd64` and `linux/arm64` platforms are supported.

## Contributing to PhysicsNeMo

PhysicsNeMo is an open source collaboration and its success is rooted in community
contribution to further the field of Physics-ML. Thank you for contributing to the
project so others can build on top of your contribution.

For guidance on contributing to PhysicsNeMo, please refer to the
[contributing guidelines](CONTRIBUTING.md).

## Cite PhysicsNeMo

If PhysicsNeMo helped your research and you would like to cite it, please refer to the
[guidelines](https://github.com/NVIDIA/physicsnemo/blob/main/CITATION.cff)

## Communication

- Github Discussions: Discuss new architectures, implementations, Physics-ML research, etc.
- GitHub Issues: Bug reports, feature requests, install issues, etc.
- PhysicsNeMo Forum: The [PhysicsNeMo Forum](https://forums.developer.nvidia.com/t/welcome-to-the-physicsnemo-ml-model-framework-forum/178556)
hosts an audience of new to moderate-level users and developers for general chat, online
discussions, collaboration, etc.

## Feedback

Want to suggest some improvements to PhysicsNeMo? Use our [feedback form](https://docs.google.com/forms/d/e/1FAIpQLSfX4zZ0Lp7MMxzi3xqvzX4IQDdWbkNh5H_a_clzIhclE2oSBQ/viewform?usp=sf_link).

## License

PhysicsNeMo is provided under the Apache License 2.0, please see [LICENSE.txt](./LICENSE.txt)
for full license text.
