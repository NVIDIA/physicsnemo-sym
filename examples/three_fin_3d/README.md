# PINNs as a parameterized surrogate model for heat-sink design optimization

This example uses PINNs to create a parameterized surrogate model that explores the design space of key parameters and identifies an optimal design.

## Problem overview

This sample demonstrates how PhysicsNeMo Sym can specify a parameterized geometry of a three-fin heat sink whose fin height, thickness, and length are variable. It shows how to use the CSG module to construct the geometry and, after training, use the surrogate to explore the design space across those parameters.
For more details, see the [documentation](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/advanced/parametrized_simulations.html).

## Dataset

This example does not require an external dataset because it directly solves the Navier–Stokes equations given the geometry and boundary conditions.

## Model overview and architecture

This is a multiphysics problem that must emulate both fluid flow and heat transfer. We deploy three neural networks: one for the flow (Navier–Stokes), one for heat transfer in the fluid (advection–diffusion), and one for heat transfer in the solid (diffusion). Each network is a fully connected MLP.

## Getting Started

To train the surrogate, run:

```bash
python three_fin_flow.py
python three_fin_thermal.py
```

To perform inference within a design-exploration loop, run:

```bash
bash design_optimization.sh
```

Note, if you run out of memory when running the design optimization loop, please
set `export CUDA_VISIBLE_DEVICES=""` in the `design_optimization.sh` script.
If you still run out of memory, lower the `batch_size` for `MonitorPlane` and
`MonitorHeatSource` in [`conf/conf_flow.yaml`](conf/conf_flow.yaml) and
[`conf/conf_thermal.yaml`](conf/conf_thermal.yaml) respectively.

Additionally, the design optimization workflow assumes the model is trained with
default configs present in the `conf/` directory. If you plan to change any configs
via commandline, this changes the directory structure of the outputs and the
`design_optimization.sh` and `three_fin_design.py` will have to be modified appropriately.

## References

- [PhysicsNeMo Documentation](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/advanced/parametrized_simulations.html)
