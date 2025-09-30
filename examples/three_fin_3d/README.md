# PINNs as parameterized surrogate model for Heat Sink design optimization

This example uses PINNs to create a parameterized surrogate model that can be used to explore the design space of certain design parameters to identify an optimal design.

## Problem overview

This sample illustrates the capabilities in PhysicsNeMo Sym to specify a paramterized geometry of a 3-fin heat sink whose fin height, fin thickness, and fin length are variable. It illustrates use of the CSG module to construct a parameterized geometry and then to use the trained surrogate to explore the design space for a range of values of the fin height, fin thickness, and fin length.
You can get more details on this sample from the [documentation](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/advanced/parametrized_simulations.html)

## Dataset

This example does not require any dataset as it solves of the Navier Stokes flow using the equations, the physical geometry and the boundary conditions.

## Model overview and architecture

This is a multi-physics problem where we have to emulate both the flow and heat tranfer. This example has three neural networks - one for emulating the flow governed by the Navier-Stokes equations, one for heat transfer in the fluid for Advection diffusion and one for heat transfer in the solid due to diffusion. We use a simple fully connected MLP for all three networks.

## Getting Started

To train, simply run

```bash
python three_fin_flow.py
python three_fin_thermal.py
```

To infer in a design exploration loop

```bash
bash design_optimization.sh
```

Note, if you run out of memory when running the design optimization loop, please
set `export CUDA_VISIBLE_DEVICES=""` in the `design_optimization.sh` script.

Additionally, the design optimization workflow assumes the model is trained with
default configs present in the `conf/` directory. If you plan to change any configs
via commandline, this changes the directory structure of the outputs and the
`design_optimization.sh` and `three_fin_design.py` will have to be modified appropriately.

## References

- [PhysicsNeMo Documentation](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/advanced/parametrized_simulations.html)
