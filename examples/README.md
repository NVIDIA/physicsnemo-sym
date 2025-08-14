<!-- markdownlint-disable MD043 -->
# NVIDIA PhysicsNeMo Sym Examples

## Introduction

This repository provides sample applications that demonstrate the use of specific Physics-ML
model architectures that are easy to train and deploy. These examples illustrate how such models can help solve real-world problems.

## Introductory

|Use case|Model|Level|Attributes|
| --- | --- |  --- | --- |
|[Lid Driven Cavity Flow](./ldc/)| Fully Connected MLP PINN |Introductory|Steady state, Multi-GPU|
|[Anti-derivative](./anti_derivative/)| Data and Physics informed DeepONet |Introductory|Steady state, Multi-GPU|
|[Darcy Flow](./darcy/)| FNO, AFNO, PINO |Introductory|Steady state, Multi-GPU|
|[Spring-mass system ODE](./ode_spring_mass/)| Fully Connected MLP PINN |Introductory|Steady state, Multi-GPU|
|[Surface PDE](./surface_pde/)| Fully Connected MLP PINN |Introductory|Steady state, Multi-GPU|

## Fluid Dynamics

| Use case                                          | Model                             | Level        | Attributes         |
|---------------------------------------------------|-----------------------------------|--------------|--------------------|
| [Taylor-Green](./taylor_green/)                   | Fully Connected MLP PINN          | Intermediate | Steady state, Multi-GPU |
| [Turbulent channel](./turbulent_channel/)         | Fourier Feature MLP PINN          | Intermediate | Steady state, Multi-GPU |
| [Turbulent super-resolution](./super_resolution/) | Super Resolution Network, Pix2Pix | Intermediate | Steady state, Multi-GPU | 
| [Airfoil Aerodynamics](./airfoil_pinn/)           | Fully Connected MLP PINN          | Intermediate | Steady state       |

## Electromagnetics

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|[Waveguide](./waveguide/)| Fourier Feature MLP PINN |Intermediate|Steady state, Multi-GPU|

## Solid Mechanics

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|[Plane displacement](./plane_displacement/)| Fully Connected MLP PINN, VPINN |Intermediate|Steady state, Multi-GPU|

## Design Optimization

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|[2D Chip](./chip_2d/)| Fully Connected MLP PINN |Advanced|Steady state, Multi-GPU|
|[3D Three fin Heatsink](./three_fin_3d/)| Fully Connected MLP PINN | Advanced |Steady state, Multi-Node|
|[FPGA Heatsink](./fpga/)| Multiple Models (including Fourier Feature MLP PINN, SIRENS, etc.) |Advanced|Steady state, Multi-Node|
|[Limerock Industrial Heatsink](./limerock/)| Fourier Feature MLP PINN |Advanced|Steady state, Multi-Node|

## Geophysics

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|[Reservoir simulation](./reservoir_simulation/)| FNO, PINO | Advanced | Steady state, Multi-Node|
|[Seismic wave](./seismic_wave/)| Fully Connected MLP PINN |Intermediate|Steady state, Multi-Node|
|[Wave equation](./wave_equation/)| Fully Connected MLP PINN |Intermediate|Steady state, Multi-Node|

## Healthcare

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|[Aneurysm modeling using STL geometry](./aneurysm/)| Fully Connected MLP PINN |Intermediate|Steady state, Multi-Node|

## Additional examples

In addition to the examples in this repo, more Physics-ML use cases and examples
can be referenced from the [PhysicsNeMo examples](https://github.com/NVIDIA/physicsnemo/blob/main/examples/README.md).

## NVIDIA support

Each README indicates the level of support provided.
Some examples are under active development and may change rapidly.
For stable examples, refer to the tagged versions.

## Feedback / Contributions

We're posting these examples on GitHub to better support the community, facilitate
feedback, as well as collect and implement contributions using
[GitHub issues](https://github.com/NVIDIA/physicsnemo-sym/issues) and pull requests.
We welcome all contributions!
