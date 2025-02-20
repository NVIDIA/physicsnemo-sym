# Airfoil Aerodynamics using Physics Informed Neural Networks

This example demonstrates how to set up a purely physics-driven model for solving for the flow field around an airfoil using Physics Informed Neural Networks (PINNs).

This example extends PINN-related concepts demonstrated in the simpler Lid Driven Cavity flow example (in `/ldc_pinns/`), with the goal of presenting them in a more applied engineering context. We recommend reading through this Lid Driven Cavity example first.

Specific tools and techniques demonstrated in this example include:
- Using [PhysicsNeMo-Sym](https://github.com/NVIDIA/Modulus-sym) to define a physics-based learning task for a PINN
- Representing custom geometry and interfacing this with PhysicsNeMo-Sym, as is often required in engineering applications
- Implementing custom [NVIDIA Warp](https://github.com/NVIDIA/warp) kernels for efficient point sampling of these custom geometries

## Problem Overview

Here, we examine the 2-dimensional steady, incompressible, viscous external aerodynamics flow around an airfoil. The flow is governed by the Navier-Stokes equations, and a chord-normalized Reynolds number of 50,000 is assumed. The airfoil its parameterized by a NACA 4-digit series, and both the camber and thickness parameters are encoded into the model inputs.

## Getting Started

### Prerequisites

If you are running this example outside the PhysicsNeMo container, install PhysicsNeMo Sym using
the instructions from [here](https://github.com/NVIDIA/Modulus-sym?tab=readme-ov-file#pypi).

### Training

To train the model on a single local GPU, run:

```bash
python airfoil.py
```

To start a multi-GPU training run, first check that multiple GPUs are available using `nvidia-smi`. Then, use the following command:

```bash
mpirun -np 2 python airfoil.py
```

Since this is training in a purely physics-based fashion, there is no dataset required. Instead, we generate the geometry using the PhysicsNeMo Sym's geometry module and a custom sampler, which is defined in `custom_airfoil_geometry.py`.

### Post-Processing

The results are saved in the `./outputs/` directory. Of particular interest are the point cloud files produced at:
    - `./outputs/inferencers/near_field_i.vtp`
    - `./outputs/inferencers/far_field_i.vtp`
    - `./outputs/inferencers/boundary_i.vtp`
as these can be visualized in ParaView to see the flow field around the airfoil. In these files, the suffix `i` refers to one of several different inference cases that are defined in the `airfoil.py` script. By modifying the script, you can explore the model's ability to generalize to different flow conditions.

In ParaView, you can visualize the flow field for a given case using the following steps:
    - Open all corresponding `.vtp` files for the case
    - Merge the point clouds using the following filters: "Group Datasets" -> "Merge Blocks"
    - Optionally, triangulate the point cloud using the "Delaunay 2D" filter to allow shading
    - Select the field variable you wish to visualize, and use either "Surface" or "Point" view to visualize the flow field

Examples:
![airfoil-cambered-pressure](../../docs/images/user_guide/airfoil-cambered-pressure.png)
![airfoil-cambered-velocity](../../docs/images/user_guide/airfoil-cambered-velocity.png)

## Additional Reading and Future Work

Notably, the default Reynolds number in this example is relatively low. This is for a few reasons:
- As the Reynolds number increases, boundary layer thicknesses decrease. As this occurs, gradients in the field solution become sharper, requiring more point density and representational capacity in the model to capture this accurately. Both factors increase the computational cost of training, and the goal of this example is to present a simple case where one can quickly iterate on and explore the problem setup.
- As the Reynolds number increases to the point where turbulent transition is expected, a turbulence model is required to accurately capture the flow physics in a steady model. Without this, the low-frequency bias of a PINN representation and the assumption of steady flow will lead to relaminarizing effects that are not physically accurate. Incorporating a turbulence model into PINNs is an active area of research (e.g., [this paper](https://arxiv.org/html/2412.01954)) and is not currently covered in this example.
