<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.4.0a0] - 2026-XX-YY

### Added

### Changed

### Deprecated

### Removed

### Fixed

- Fixed HTTP Error 308 when running `examples/super_resolution.py`
- Fixed deprecated documentation links to the Johns Hopkins Turbulence Database

### Security

### Dependencies

## [2.3.0] - 2025-11-17

### Added

- Added Blackwell support

## [2.2.0] - 2025-08-26

### Added

- Improved performance of connectivity calculation. Refactor of Least squares
  gradients
- Added Incompressible Navier Stokes PDE as a special formulation
- Added ability for `make_nodes` function to return a dict

### Changed

### Deprecated

### Removed

### Fixed

### Security

### Dependencies

## [2.1.0] - 2025-06-10

### Fixed

- Fix the zero equation formulation to correctly reflect turbulent kinematic viscosity

- Made `tiny-cuda-nn` optional in deployment image

## [2.0.0] - 2025-03-18

### Changed

- Update `inspect.getargspec` usage

### Fixed

- Fix optimizer checkpoint loading errors

### Dependencies

- Remove upper bound for Cython
- Add upper bound for numpoly (transitive dependency of chaospy)

## [1.8.0] - 2024-12-04

### Added

- Added devcontainer support.

### Changed

- Speeds up CUDA extension build by about 3x.

### Fixed

- Fix the area calculated for STL meshes.

### Dependencies

- Relaxes versions for several dependencies.

## [1.7.0] - 2024-09-24

### Added

- AMP for derivatives.
- DALI based dataloader for Geometry module.
- Generalized PDE residual computing utility.
- Support for spatial gradients calculations using finite difference, meshless finite
  difference, spectral and least squares methods.
- Add docs for SDF and relevant features for geometry module.

### Dependencies

- Upgrade Sympy and Scikit-Learn versions.

### Security

- Upgrade notebook, opencv-python and setuptools versions to fix CVEs GHSA-9q39-rmj3-p4r2,
  GHSA-qr4w-53vh-m672, and GHSA-cx63-2mw6-8hw5 respectively.

## [1.6.0] - 2024-07-23

### Added

- Added inference notebook for Turbulence Super-resolution example.

### Changed

- Warp based backed for STL geometry handling

### Dependencies

- Update `timm` dependency
- Update minimum python version to 3.10

## [1.5.0] - 2024-04-17

### Added

- Added reservoir examples using GenAI and CCUS workflows.

### Security

- Update OpenCV and Pillow versions to fix security

## [1.4.0] - 2024-01-25

### Fixed

- Fix bug for `ConvFullyConnectedArch`.
- Updating `Activation.SILU` test to conform with updated nvFuser kernel generation.

## [1.3.0] - 2023-11-20

### Added

- Added instructions for docker build on ARM architecture.
- Added domain decomposition examples using X-PINN and FB-PINN style.

### Changed

- Integrated the network architecture layers into PhysicsNeMo-Core.

### Fixed

- Fixed Gradient Aggregation bug.

### Security

- Upgrade Pillow and Sympy to their latest versions.
- Upgrade Scikit-Learn version.

### Dependencies

- Updated base PyTorch container to 23.10 and Optix version to 7.3.0

## [1.2.0] - 2023-09-21

### Added

- Example for reservoir modeling using PINO and FNO

## [1.1.0] - 2023-08-10

### Added

- Added a CHANGELOG.md

### Removed

- Accompanying licenses (will provide in the PhysicsNeMo docker image).

### Fixed

- Arch `from_config` bug for literal params.
- Fixed fused SiLU activation test.
- Update `np.bool` to `np.bool_`.
- Added a workaround fix for the CUDA graphs error in multi-node runs

### Dependencies

- Updated the base container to latest PyTorch base container which is based on torch 2.0
- Container now supports CUDA 12, Python 3.10
- Updated symengine to >=0.10.0 and vtk to >=9.2.6

## [1.0.0] - 2023-05-08

### Added

- Initial public release.
