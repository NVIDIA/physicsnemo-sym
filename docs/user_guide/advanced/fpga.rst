.. _fpga:

FPGA Heat Sink with Laminar Flow
================================

Introduction
------------

This tutorial shows how some of the features in PhysicsNeMo Sym apply for a
complicated FPGA heat sink design and solve the conjugate heat transfer.
In this tutorial you will learn:

#. How to use Fourier Networks for complicated geometries with sharp
   gradients

#. How to solve problem with symmetry using symmetry boundary conditions

#. How to formulate velocity field as a vector potential (Exact
   continuity feature)

#. How different features and architectures in PhysicsNeMo Sym perform on a
   problem with complicated geometry

.. note::
   This tutorial is very similar to the conjugate heat transfer problem
   shown in the :ref:`cht` and
   :ref:`ParameterizedSim` tutorials. You should review
   review these tutorials (especially the :ref:`cht` tutorial) for the
   details on the geometry generation, constraints, etc. 
   This tutorial skips the description for these
   processes and instead focuses on the
   implementation of different features and the case study.

Problem Description
-------------------

The geometry of the FPGA heatsink is shown in 
:numref:`fig-laminar_fpga_geom_2`. This particular geometry is
challenging to simulate due to the thin closely spaced fins that causes
sharp gradients which are particularly difficult to learn using regular
fully connected neural network (slow convergence).



.. _fig-laminar_fpga_geom_2:

.. figure:: /images/user_guide/fpga_geom.PNG
   :alt: The FPGA heat sink geometry
   :name: fig:laminar_fpga_geom_2
   :width: 40.0%
   :align: center

   FPGA heat sink geometry

This section solves the conjugate heat transfer problem for the
geometry above at :math:`Re=50`. The dimensions of the geometry, as modeled in PhysicsNeMo Sym,
are summarized here: 

.. table:: FPGA Dimensions 
   :align: center

   ============================ ======================
   Dimension                    Value
   Heat Sink Base `(l x b x h)`  0.65 x 0.875 x 0.05
   Fin dimension `(l x b x h)`   0.65 x 0.0075 x 0.8625
   Heat Source `(l x b)`         0.25 x 0.25
   Channel `(l x b x h)``        5.0 x 1.125 x 1.0
   ============================ ======================

All the dimensions are scaled such that the channel height is 1
:math:`m`. The temperature is scaled according to
:math:`\theta=T / 273.15-1.0`. The channel walls are treated as
adiabatic and the interface boundary conditions are applied at the
fluid-solid interface. Other flow and thermal parameters are described
in the table below. 

.. table:: Fluid and Solid Properties
   :align: center

   ============================================== ======= =====
   Property                                       Fluid   Solid
   Inlet Velocity :math:`(m/s)`                   1.0     NA
   Density :math:`(kg/m^3)`                       1.0     1.0
   Kinematic Viscosity :math:`(m^2/s)`            0.02    NA
   Thermal Diffusivity :math:`(m^2/s)`            0.02    0.1
   Thermal Conductivity :math:`(W/m.K)`           1.0     1.0
   Inlet Temperature :math:`(K)`                  273.15  NA
   Heat Source Temperature Gradient :math:`(K/m)` 409.725 NA
   ============================================== ======= =====

Case Setup
----------

The case setup for this problem is very similar to
the problem described in the :ref:`cht` tutorial. Like the :ref:`cht` 
tutorial, you have 3 separate scripts for this problem for
the geometry definition, flow constraints and solver and heat constraints and solver. 

.. note:: All the relevant domain, flow and heat solver files for this problem using various versions of features can be found at ``examples/fpga/``.

Solver using Fourier Network Architecture
-----------------------------------------

As described in the :ref:`theory`, in PhysicsNeMo Sym, the spectral bias of the neural networks can be overcome by
using the Fourier Networks. These networks have shown a significant
improvement in results over the regular fully connected neural networks
due to their ability to capture sharp gradients.

You do not need to make any special changes to the way the geometry and
constraints definition while making changes to the neural network
architectures. This also means the architecture is independent of the
physics or parameterization being solved and can be applied to any other
class of problems covered in the User Guide. More details about architecture configuration
can be found in the Hydra configs section (:ref:`config`).

**A note on frequencies:** One of the main parameters of these networks are 
the frequencies. In PhysicsNeMo Sym, you can choose frequencies from the spectrum you
want to sample (full/axis/gaussian/diagonal) and the number of frequencies in the
spectrum. The optimal number of frequencies depends on every problem and one 
often needs to balance the accuracy benefits and the
computational expense added due to use of extra Fourier features. For the
FPGA problem, choosing the default works for the laminar problem, while 
for the turbulent case, you increase the number of frequencies to 35.

The solver file for the parameterized FPGA flow field
simulation for laminar case is shown below. The different architectures can be chosen by setting the 
appropriate keyword in the custom arguments defined in the config file. 

.. literalinclude:: ../../../examples/fpga/laminar/fpga_flow.py
   :lines: 64-108 

Leveraging Symmetry of the Problem
----------------------------------

Whenever there is a symmetric geometry and the variable fields are expected
to be symmetric, you can use symmetry boundary conditions about the plane
or axis symmetry to minimize the computational expense of modeling the
entire geometry. For the FPGA heat sink, you have such a plane of
symmetry in the z-plane (:numref:`fig-fpga_symm`). The symmetry
boundary conditions are discussed in the :ref:`theory-symmetry` section. 
Simulating the FPGA problem using symmetry, you can
achieve about 33% reduction in training time, compared to a training on
the full domain.

For the FPGA problem where the plane of symmetry is z-plane, the
boundary conditions stated in Section
:ref:`theory-symmetry` can be translated to the
following:

#. Variables which are odd functions w.r.t. ``z`` coordinate axis:
   ``'w'``. Hence on symmetry plane ``'w'=0``.

#. Variables which are even functions w.r.t. ``z`` coordinate axis:
   ``'u', 'v'`` components of velocity vector and scalar quantities like
   ``'p', 'theta_s' , 'theta_f'``. On a symmetry plane, set their
   normal derivative to :math:`0`. Eg. ``'u__z'=0``.


.. _fig-fpga_symm:

.. figure:: /images/user_guide/symmetry_plane_viz.png
   :alt: The FPGA heat sink with plane of symmetry
   :name: fig:fpga_symm
   :width: 50.0%
   :align: center

   FPGA plane of symmetry

Only the symmetry boundary conditions in the flow and heat
training domains are shown here. The rest of the training domain remains the same.
(Full files can be accessed at ``examples/fpga/laminar_symmetry/``)

.. literalinclude:: ../../../examples/fpga/laminar_symmetry/fpga_flow.py
   :lines: 115-123

.. literalinclude:: ../../../examples/fpga/laminar_symmetry/fpga_heat.py
   :lines: 139-157 

Imposing Exact Continuity
-------------------------

You can define the velocity field as a vector potential such
that it is divergence free and satisfies continuity automatically. You
can use this formulation for any class of flow problems covered in this
guide regardless of the network architecture. However, 
it is most effective when using fully connected networks.

The code below shows how the exact continuity is implemented by modifying the 
output nodes of the problem. Caution should be taken when using the exact continuity
as it is memory intensive and you might have to modify the batch sizes 
to fit the problem in GPU memory. 

.. literalinclude:: ../../../examples/fpga/laminar/fpga_flow.py
   :lines: 64-71

Results, Comparisons, and Summary
---------------------------------

The :ref:`table-feature-summary` table
summarizes the features discussed in this chapter and their applications. 
The :ref:`table-results-fpga` table summarizes the important results
of these features on this FPGA problem. Also, :numref:`fig-fpga-loss-plots`, provides a
comparison for the loss values from different runs.

.. _table-feature-summary:

.. table:: Summary of features introduced in this tutorial
   :align: center

   +----------------------+----------------------+-----------------------+
   | **Feature**          | **Applicability to** | **Comments**          |
   |                      | **other problems**   |                       |
   +----------------------+----------------------+-----------------------+
   | Fourier Networks     | Applicable to all    | Shown to be highly    |
   |                      | class of problems    | effective for         |
   |                      |                      | problems involving    |
   |                      |                      | sharp gradients.      |
   |                      |                      | Modified Fourier      |
   |                      |                      | network found to      |
   |                      |                      | improve the           |
   |                      |                      | performance one step  |
   |                      |                      | further.              |
   +----------------------+----------------------+-----------------------+
   | Symmetry             | Applicable to all    | Reduces the           |
   |                      | problems with a      | computational domain  |
   |                      | plane/axis of        | to half leading to    |
   |                      | symmetry             | significant speed-up  |
   |                      |                      | (33% reduction in     |
   |                      |                      | training time         |
   |                      |                      | compared to full      |
   |                      |                      | domain).              |
   +----------------------+----------------------+-----------------------+
   | Exact Continuity     | Applicable to        | Gives better          |
   |                      | incompressible flow  | satisfaction of the   |
   |                      | problems requiring   | continuity equation   |
   |                      | solution to Navier   | than                  |
   |                      | Stokes equation      | Velocity-pressure     |
   |                      |                      | formulation. Found    |
   |                      |                      | to work best with     |
   |                      |                      | standard fully        |
   |                      |                      | connected networks.   |
   |                      |                      | Also improves the     |
   |                      |                      | accuracy of results   |
   |                      |                      | in Fourier networks.  |
   +----------------------+----------------------+-----------------------+
   | SiReNs               | Applicable to all    | Shown to be           |
   |                      | class of problems    | effective for         |
   |                      |                      | problems with sharp   |
   |                      |                      | gradients. However,   |
   |                      |                      | it does not           |
   |                      |                      | outperform the        |
   |                      |                      | Fourier Networks in   |
   |                      |                      | terms of accuracy.    |
   +----------------------+----------------------+-----------------------+
   | DGM Networks         | Applicable to all    | Improves accuracy     |
   | Global Adaptive,     | class of problems    | compared to regular   |
   | Activations and      |                      | fully connected       |
   | Halton Sequences,    |                      | networks.             |
   | etc.                 |                      |                       |
   +----------------------+----------------------+-----------------------+

.. _table-results-fpga: 


.. table:: Comparison of pressure drop and peak temperatures from various runs
   :align: center

   +---------------------------+----------------------+----------------------+
   | **Case Description**      | :math:`P_{drop}`     | :math:`T_{peak}`     |
   |                           | :math:`(Pa)`         | :math:`(^{\circ} C)` |
   +---------------------------+----------------------+----------------------+
   | **PhysicsNeMo Sym:** Fully    | 29.24                | 77.90                |
   | Connected Networks        |                      |                      |
   +---------------------------+----------------------+----------------------+
   | **PhysicsNeMo Sym:** Fully    | 28.92                | 90.63                |
   | Connected Networks        |                      |                      |
   | with Exact                |                      |                      |
   | Continuity                |                      |                      |
   +---------------------------+----------------------+----------------------+
   | **PhysicsNeMo Sym:** Fourier  | 29.19                | 77.08                |
   | Networks                  |                      |                      |
   +---------------------------+----------------------+----------------------+
   | **PhysicsNeMo Sym:** Modified | 29.23                | 80.96                |
   | Fourier Networks          |                      |                      |
   +---------------------------+----------------------+----------------------+
   | **PhysicsNeMo Sym:** SiReNs   | 29.21                | 76.54                |
   +---------------------------+----------------------+----------------------+
   | **PhysicsNeMo Sym:** Fourier  | 29.14                | 78.56                |
   | Networks with             |                      |                      |
   | Symmetry                  |                      |                      |
   +---------------------------+----------------------+----------------------+
   | **PhysicsNeMo Sym:** DGM      | 29.10                | 76.86                |
   | Networks with Global      |                      |                      |
   | LR annealing, Global      |                      |                      |
   | Adaptive                  |                      |                      |
   | Activations, and          |                      |                      |
   | Halton Sequences          |                      |                      |
   +---------------------------+----------------------+----------------------+
   | **OpenFOAM Solver**       | 28.03                | 76.67                |
   +---------------------------+----------------------+----------------------+
   | **Commercial Solver**     | 28.38                | 84.93                |
   |                           |                      |                      |
   +---------------------------+----------------------+----------------------+


.. _fig-fpga-loss-plots:

.. figure:: /images/user_guide/loss_fpga_laminar_flow.png
   :alt: flow
   :align: center

   Flow comparisons

.. figure:: /images/user_guide/loss_fpga_laminar_heat.png
   :alt: heat
   :align: center

   Heat comparisons

