.. _cht:

Conjugate Heat Transfer
=======================

Introduction
------------

This tutorial uses PhysicsNeMo Sym to study the conjugate heat
transfer between the heat sink and the surrounding fluid. The
temperature variations inside solid and fluid would be solved in a
coupled manner with appropriate interface boundary conditions. In this
tutorial, you will learn:

#. How to generate a 3D geometry using the geometry module in PhysicsNeMo Sym.

#. How to set up a conjugate heat transfer problem using the interface
   boundary conditions in PhysicsNeMo Sym.

#. How to use the Multi-Phase training approach in PhysicsNeMo Sym for one way
   coupled problems.

.. note::
   This tutorial assumes that you have completed tutorial :ref:`Introductory Example`
   on and have familiarized yourself with the basics
   of the PhysicsNeMo Sym APIs. Also, you should review the 
   :ref:`advection-diffusion` tutorial for additional details
   on writing some of the thermal boundary conditions.

.. note::
   The scripts used in this problem can be found at ``examples/three_fin_3d/``. The scripts are made configurable such that the same three fin problem can be solved for higher reynolds numbers, 
   parameterized geometry, etc. very easily by only configuring the Hydra configs. This example focuses on the laminar variant of this as the purpose of this tutorial is to cover
   the ideas related to multi-phase training and conjugate heat transfer. The tutorial :ref:`ParameterizedSim` covers the parameterization aspect of this problem. 

   Therefore in this problem, the ``custom`` configs for ``turbulent`` and ``parameterized`` are both set to ``false``.  

Problem Description
-------------------

The geometry for a 3-fin heat sink placed inside a channel is shown in
:numref:`fig-threeFin_heatsink`. The inlet to the channel is at 1
:math:`m/s`. The pressure at the outlet is specified as 0 :math:`Pa`.
All the other surfaces of the geometry are treated as no-slip walls.
 
.. _fig-threeFin_heatsink:

.. figure:: ../../images/user_guide/threeFin_geom.png
   :alt: Three fin heat sink geometry (All dimensions in :math:`m`)
   :name: fig:threeFin_heatsink
   :width: 100.0%
   :align: center

   Three fin heat sink geometry (All dimensions in :math:`m`)

The inlet is at 273.15 :math:`K`. The channel walls are adiabatic. The
heat sink has a heat source of :math:`0.2 \times 0.4` :math:`m` at the bottom of the
heat sink situated centrally on the bottom surface. The heat source
generates heat such that the temperature gradient on the source surface
is 360 :math:`K/m` in the normal direction. Conjugate heat transfer
takes place between the fluid-solid contact surface.

The properties fluid and thermal properties of the fluid and the solid
are as follows:

.. table:: Fluid and Solid Properties
   :align: center

   ==================================== ===== ======
   Property                             Fluid Solid
   Kinematic Viscosity :math:`(m^2/s)`  0.02  NA
   Thermal Diffusivity :math:`(m^2/s)`  0.02  0.0625
   Thermal Conductivity :math:`(W/m.K)` 1.0   5.0
   ==================================== ===== ======

Case Setup
----------

In this tutorial, since you are dealing with only incompressible flow,
there is a one way coupling between the heat and flow equations. This
means that it is possible to train the temperature field after the flow
field is trained and converged. Such an approach is useful while
training the multiphysics problems which are one way coupled as it is
possible to achieve significant speed-up, as well as simulate cases with
same flow boundary conditions but different thermal boundary conditions.
One can easily use the same flow field as in input to train for
different thermal boundary conditions.

Therefore, for this problem you have three separate files for the geometry,
flow solver, and heat solver. The ``three_fin_geometry.py`` will contain
all the definitions of geometry. ``three_fin_flow.py`` and ``three_fin_thermal.py``
would then use this geometry to setup the relevant flow and heat constraints and 
solve them individually. The basic idea would be to train the flow model to convergence 
and then start the heat training after by initializing from the trained flow model to solve 
for the temperature distributions in fluid and solid simultaneously.

In this problem you will nondimensionalize the temperature according to
the following equation:

.. math:: \theta= T/273.15 - 1.0

Creating Geometry
~~~~~~~~~~~~~~~~~

The ``three_fin_geometry.py`` script contains all the details relevant to the geometry generation. 
We will use the ``Box`` primitive to create the
3-fin geometry and ``Channel`` primitive to generate the channel.
Similar to 2D, ``Channel`` and ``Box`` are defined by using the two
corner points. Like 2D, the ``Channel`` geometry has no bounding planes
in the x-direction. You will also make use of the ``repeat`` method to
create the fins. This speeds up the generation of repetitive
structures in comparison to constructing the same geometry separately and doing boolean operations to assemble them.

Use the ``Plane`` geometry to create the planes at the inlet and
outlet. The code for generating the required geometries is shown below.
Please note the normal directions for the inlet and outlet planes.

Additionally, the parameters required for solving the heat part as also
defined upfront, ex. dimensions and locations of source etc.

.. note:: The script contains a few extra definitions that are only relevant for parameterized geometry. These are not relevant for this tutorial. 

.. literalinclude:: ../../../examples/three_fin_3d/three_fin_geometry.py
   :language: python
   :lines: 29-

Neural network, Nodes and Multi-Phase training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's have a look at the networks and nodes required to solve the flow and heat for this problem. The architectures and nodes for flow problem are very similar to previous tutorials. You will add the nodes for ``NavierStokes`` and ``NormalDotVec`` and create a single flow network that has the coordinates as inputs and the velocity components and the pressure as output. The code for the flow nodes can be found here:  

.. literalinclude:: ../../../examples/three_fin_3d/three_fin_flow.py
   :language: python
   :lines: 51-86


For the thermal nodes, start by adding nodes for relevant equations like ``AdvectionDiffusion``, ``Diffusion``, ``DiffusionInterface`` and ``GradNormal`` that will be used to define the various thermal boundary conditions relevant to this problem. Also, create 3 separate neural networks ``flow_net``, ``thermal_f_net`` and ``thermal_s_net``. The first one is the same flow network  defined in the flow scripts. This network architecture definition in heat script must exactly match to that of the flow script for successful initialization of the flow model during heat training. Set the ``optimize`` argument as ``False`` while making the nodes of flow network to avoid optimizing the flow network during the heat training. Finally, separate networks to predict the temperatures in fluid and solid are created. 

.. literalinclude:: ../../../examples/three_fin_3d/three_fin_thermal.py
   :language: python
   :lines: 51-95


Setting up Flow Domain and Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contents of ``three_fin_flow.py`` script are described below.

Inlet, Outlet and Channel and Heat Sink walls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For inlet boundary conditions, specify the velocity to be a
constant velocity of 1.0 :math:`m/s` in x-direction. Like in tutorial
:ref:`Introductory Example`, weight the velocity by the SDF of the channel
to avoid sharp discontinuity at the boundaries. For outlet, 
specify the pressure to be 0. All the channel walls and heat sink walls
are treated as no slip boundaries.

Interior
^^^^^^^^

The flow equations can be specified in the low resolution and high
resolution domains of the problem by using ``PointwiseInteriorConstraint``. This allows 
independent point densities in these two areas to be controlled easily.
 
Integral Continuity
^^^^^^^^^^^^^^^^^^^

The inlet volumetric flow is 1 :math:`m^3/s` so,
specify 1.0 as the value for ``integral_continuity`` key. 

The code for flow domain is shown below.

.. literalinclude:: ../../../examples/three_fin_3d/three_fin_flow.py
   :language: python
   :lines: 87-200

.. note::
   The addition of integral continuity planes and separate flow box for dense sampling are examples of adding more training data and user knowledge/bias of the problem to the training. This addition helps to improve the accuracy and convergence to a great extent and it is recommended wherever possible.


Setting up Thermal Multi-Phase Domain and Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contents of ``three_fin_thermal.py`` are described below.

Inlet, Outlet and Channel walls:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the heat part, specify temperature at the inlet. All the
outlet and the channel walls will have a zero gradient boundary
condition which will be enforced by setting
``'normal_gradient_theta_f'`` equal to 0. We will use ``'theta_f'`` for
defining the temperatures in fluid and ``'theta_s'`` for defining the
temperatures in solid.

Fluid and Solid Interior:
^^^^^^^^^^^^^^^^^^^^^^^^^

Just like the :ref:`advection-diffusion` tutorial, 
set ``'advection_diffusion'`` equal to 0 in both, low and high
resolution fluid domains. For solid interior, set
``'diffusion'`` equal to 0.

Fluid-Solid Interface:
^^^^^^^^^^^^^^^^^^^^^^

For the interface between fluid and solid, enforce both Neumann
and Dirichlet boundary condition by setting
``'diffusion_interface_dirichlet_theta_f_theta_s'`` and
``'diffusion_interface_neumann_theta_f_theta_s'`` both equal to 0.

.. note::
   The order in which you define ``'theta_f'`` and ``'theta_s'`` in the interface boundary condition must match the one in PDE definition ``DiffusionInterface`` to avoid any graph unroll errors. 
   The corresponding conductivities must be also specified in the same order in the PDE definition. 

Heat Source:
^^^^^^^^^^^^

Apply a :math:`tanh` smoothing while defining the heat source on
the bottom wall of the heat sink. Smoothing out the
sharp boundaries helps in training the Neural Network converge faster.
The ``'normal_gradient_theta_s'`` is set equal to ``grad_t`` in the area
of source and 0 everywhere else on the bottom surface of heat sink.

The code for heat domain is shown below.

.. literalinclude:: ../../../examples/three_fin_3d/three_fin_thermal.py
   :language: python
   :lines: 96-215


Adding Validators and Monitors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this tutorial you will monitors for pressure drops during the flow field
simulation and monitors for peak temperature reached in the source chip during the heat simulation.

Similarly, respective validation data are added to the flow and heat scripts. Only flow monitors and validators are shown for brevity. 

Monitors and validators in flow script:

.. literalinclude:: ../../../examples/three_fin_3d/three_fin_flow.py
   :language: python
   :lines: 201-373


Training the Model
------------------

Once both the flow and heat scripts are defined, run the
``three_fin_flow.py`` first to solve for the flow field. Once a
desired level of convergence is achieved you can run ``three_fin_thermal.py`` to solve for heat.

Results and Post-processing
---------------------------

The table and figures below show the results of Pressure drop and Peak
temperatures obtained from the PhysicsNeMo Sym and compare it with the results
from OpenFOAM solver.

.. table:: Comparisons of Results with OpenFOAM
   :align: center

   ===================================== =========== ========
   \                                     PhysicsNeMo Sym OpenFOAM
   Pressure Drop :math:`(Pa)`            7.51        7.49
   Peak Temperature :math:`(^{\circ} C)` 78.35       78.05
   ===================================== =========== ========


.. figure:: ../../images/user_guide/CHTXSlice.png
   :alt: Left: PhysicsNeMo Sym. Center: OpenFOAM. Right: Difference. Top: Temperature distribution in Fluid. Bottom: Temperature distribution in Solid (*Temperature scales in C*)
   :name: fig:3d_heat_sink_xslice_heat
   :width: 100.0%
   :align: center

   Left: PhysicsNeMo Sym. Center: OpenFOAM. Right: Difference. Top: Temperature
   distribution in Fluid. Bottom: Temperature distribution in Solid (*Temperature scales in C*)

.. figure:: ../../images/user_guide/CHTZSlice.png
   :alt: Left: PhysicsNeMo Sym. Center: OpenFOAM. Right: Difference. (*Temperature scales in C*)
   :name: fig:3d_heat_sink_zslice_heat
   :width: 100.0%
   :align: center

   Left: PhysicsNeMo Sym. Center: OpenFOAM. Right: Difference. (*Temperature scales in C*)

Plotting gradient quantities: Wall Velocity Gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a variety of applications, it is desirable to plot the gradients of
some quantities inside the domain. One such example relevant to fluid
flows is the wall velocity gradients and wall shear stresses. These
quantities are often plotted to compute frictional forces, etc. You can
visualize such quantities in PhysicsNeMo Sym by outputting the :math:`x`,
:math:`y` and :math:`z` derivatives of the desired variables using an 
``PointwiseInferencer``.
 
.. code:: python
    
   ...
    	# add inferencer
    	inferencer = PointwiseInferencer(
    	    geo.sample_boundary(4000, parameterization=pr),
    	    ["u__x", "u__y", "u__z", 
	     "v__x", "v__y", "v__z",
	     "w__x", "w__y", "w__z"],
    	    nodes=flow_nodes,
    	)
    	flow_domain.add_inferencer(inferencer, "inf_data")


You can then post-process these quantities based on your choice to
visualize the desired variables. `Paraview’s Calculator Filter
<https://www.paraview.org/paraview-docs/nightly/python/paraview.simple.Calculator.html>`_ was used for the 
plot shown below. The wall velocity gradients comparison between OpenFOAM and PhysicsNeMo Sym is
shown in :numref:`fig-3d_heat_sink_wall_velGrad`.

.. _fig-3d_heat_sink_wall_velGrad:

.. figure:: ../../images/user_guide/wall_shear_stress_3Fin.png
   :alt: Comparison of magnitude of wall velocity gradients. Left: PhysicsNeMo Sym. Right: OpenFOAM
   :width: 80.0%
   :align: center

   Comparison of magnitude of wall velocity gradients. Left: PhysicsNeMo Sym.
   Right: OpenFOAM
