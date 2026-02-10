# PINNs for simulating 2D seismic-wave propagation

This example employs PINNs to emulate time-dependent 2D seismic-wave propagation in a simple domain. 

## Problem overview
This sample illustrates how to solve the acoustic wave equation.
For additional details, see the [documentation](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/foundational/2d_wave_equation.html).

## Dataset

This example does not require an external dataset because it directly solves the acoustic wave equation using the prescribed geometry and boundary conditions.

## Model overview and architecture

We use a fully connected MLP to approximate the solution of the time-dependent 2-D wave equation given the boundary conditions. The network takes (x, y, t) as input and outputs the pressure response u and wave velocity c.

## Getting Started

To run the example, execute:

```bash
python wave_2d.py
```

## References

- [PhysicsNeMo Documentation](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/foundational/2d_wave_equation.html)
