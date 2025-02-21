
Table of Contents
==================

* Getting Started

    * :ref:`Table of Contents`

* Learn the Basics

    * :ref:`PhysicsNeMo Sym Overview`
    * :ref:`Introductory Example`

* Theory

    * :ref:`Physics Informed Neural Networks in PhysicsNeMo Sym`
    * :ref:`Architectures In PhysicsNeMo Sym`
    * :ref:`Advanced Schemes and Tools`
    * :ref:`Recommended Practices in PhysicsNeMo Sym`
    * :ref:`Miscellaneous Concepts`

* PhysicsNeMo Sym Features

    * :ref:`PhysicsNeMo Sym Configuration`
    * :ref:`Post Processing in PhysicsNeMo Sym`
    * :ref:`Performance`

* PhysicsNeMo Sym API

    * :ref:`Core API`
    * :ref:`Models`
    * :ref:`Utilities`

Example Index
----------------

Physics-Informed Foundations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`1D Wave Equation`:  This example solves a transient 1D wave equation and demonstrates coding a custom PDE in PhysicsNeMo Sym. The time-dependent problem is solved using the continuous time, time marching and temporal loss weighting schemes.

* :ref:`2D Seismic Wave Propagation`: This example applies the concepts of continuous time for a 2D wave propagation problem encountered in seismic surveys.

* :ref:`Coupled Spring Mass ODE System`: This example shows the use of PhysicsNeMo Sym for solving a system of ordinary differential equations.

* :ref:`Turbulent physics: Zero Equation Turbulence Model`: This example extends the lid driven cavity flow by including a turbulence model in the governing equations.

* :ref:`Scalar Transport: 2D Advection Diffusion`: This example simulates an advection-diffusion problem to model a scalar transport phenomenon.

* :ref:`Linear Elasticity`: This example demonstrates how to use PhysicsNeMo Sym for solving 3D and 2D stress-strain problems.

* :ref:`Inverse Problem: Finding Unknown Coefficients of a PDE`: This example provides a guide on using PINNs to assimilate the known quantities to infer/invert data which would be otherwise impossible for traditional methods.

Neural Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`darcy_fno`: This example develops a data-driven model for a 2D Darcy flow using the Fourier Neural Operator.

* :ref:`darcy_afno`: This example develops a data-driven model for a 2D Darcy flow using the Adaptive Fourier Neural Operator.

* :ref:`darcy_pino`: This example develops a physics-informed data-driven model for a 2D Darcy flow using the Physics-Informed Neural Operator.

* :ref:`deeponet`: This example uses PhysicsNeMo Sym to solve anti-derivative problems with data-driven and physics informed DeepONet.

* :ref:`fourcastnet_example`: This example recreates the example from FourCastNet paper in PhysicsNeMo Sym.

Intermediate Case Studies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`Interface Problem by Variational Method`: In this example we show how to solve the PDEs in their variational form (weak solutions) using PhysicsNeMo Sym. Such formulation helps to solve the PDEs for which obtaining the solution in classical sense is very complex (e.g. problems with interface, singularities, etc.).

* :ref:`STL Geometry: Blood Flow in Intracranial Aneurysm`: This example demonstrates import of an STL geometry (that can be exported from a CAD program) in PhysicsNeMo Sym. In this tutorial, PhysicsNeMo Sym uses its native SDF (Signed Distance Function) library to calculate the SDF for the points in the point cloud and determine if they are on, outside or inside the surface.

* :ref:`Moving Time Window: Taylor Green Vortex Decay`: This example introduces PhysicsNeMo Sym' sequential solver and solves the canonical Taylor-Green vortex decay problem using the moving time window approach

* :ref:`Electromagnetics: Frequency Domain Maxwell's Equation`: This example covers the electromagnetic simulations using PINNs, solving the frequency domain Maxwell's equations.

* :ref:`two_equation_turbulent_channel`: This example shows the use of PINNs to solve a canonical turbulent flow in a 2D channel using two equation turbulence models and wall functions.

* :ref:`turbulence_super_res`: This example develops a super resolution surrogate model for predicting high-fidelity forced isotropic turbulence fields from filtered low-resolution observations.

Advanced Case Studies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`Conjugate Heat Transfer`: This example demonstrates the use of PhysicsNeMo Sym to study the conjugate heat transfer between a 3D heat sink and the surrounding fluid. 

* :ref:`Parameterized 3D Heat Sink`: This example showcases parameterization and the major computational advantage of PINNs in solving industrial scale design optimization problems.

* :ref:`2d_heat`: This example demonstrates PhysicsNeMo Sym for solving conjugate heat transfer problems with higher thermal conductivities that represent more realistic materials. 

* :ref:`FPGA Heat Sink with Laminar Flow`: This example showcases the various features and architectures in PhysicsNeMo Sym for more complex geometry.

* :ref:`Industrial Heat Sink`: This example shows an even more complicated geometry with real physics. Such problems present a new class of complexities for the PINNs and algorithms like hFTB (heat transfer coefficient forward temperature backward), gradient aggregation and surrogate modeling through gPC (generalized polynomial chaos) are presented that help to tackle them.
