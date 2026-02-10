.. _advection-diffusion:

Scalar Transport: 2D Advection Diffusion
========================================

Introduction
------------

In this tutorial, you will use an advection-diffusion transport equation
for temperature along with the Continuity and Navier-Stokes equation to
model the heat transfer in a 2D flow. In this tutorial you will learn:

#. How to implement advection-diffusion for a scalar quantity.

#. How to create custom profiles for boundary conditions and to set up
   gradient boundary conditions.

#. How to use additional constraints like ``IntegralBoundaryConstraint`` to
   speed up convergence.

.. note::
   This tutorial assumes that you have completed tutorial :ref:`Introductory Example` and have 
   familiarized yourself with the basics of the PhysicsNeMo Sym APIs.
 
Problem Description
-------------------

In this tutorial, you will solve the heat transfer from a 3-fin heat
sink. The problem describes a hypothetical scenario wherein a 2D slice
of the heat sink is simulated as shown in the figure. The heat sinks are
maintained at a constant temperature of 350 :math:`K` and the inlet is
at 293.498 :math:`K`. The channel walls are treated as adiabatic. The
inlet is assumed to be a parabolic velocity profile with 1.5 :math:`m/s`
as the peak velocity. The kinematic viscosity :math:`\nu` is set to 0.01
:math:`m^2/s` and the Prandtl number is 5. Although the flow is laminar,
the Zero Equation turbulence model is kept on.

.. figure:: ../../images/user_guide/threeFin_2d_geom.png
   :alt: 2D slice of three fin heat sink geometry (All dimensions in :math:`m`)
   :name: fig:threeFin_2d
   :width: 100.0%
   :align: center

   2D slice of three fin heat sink geometry (All dimensions in :math:`m`)

Case Setup
----------

.. note::
 The python script for this problem can be found at ``examples/three_fin_2d/heat_sink.py``.

Importing the required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this tutorial you will make use of ``Channel2D`` geometry to make the
duct. ``Line`` geometry can be used to make inlet, outlet and
intermediate planes for integral boundary conditions. The
``AdvectionDiffusion`` equation is imported from the ``PDES`` module.
The ``parabola`` and ``GradNormal`` are imported from appropriate
modules to generate the required boundary conditions.

.. literalinclude:: ../../../examples/three_fin_2d/heat_sink.py
   :language: python
   :lines: 15-42


Creating Geometry
~~~~~~~~~~~~~~~~~

To generate the geometry of this problem, use
``Channel2D`` for duct and ``Rectangle`` for generating the heat sink.
The way of defining ``Channel2D`` is same as ``Rectangle``. The
difference between a channel and a rectangle is that a channel is infinite and
composed of only two curves and a rectangle is composed of four curves
that form a closed boundary.

``Line`` is defined using the x and y coordinates of the two endpoints
and the normal direction of the curve. The ``Line`` requires
the x coordinates of both the points to be same. A line in arbitrary
orientation can then be created by rotating the ``Line`` object.

The following code generates the geometry for the 2D heat sink problem.

.. literalinclude:: ../../../examples/three_fin_2d/heat_sink.py
   :language: python
   :lines: 47-100


Defining the Equations, Networks and Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this problem, you will make two separate network architectures for
solving flow and heat variables to achieve increased accuracy.

Additional equations (compared to tutorial :ref:`ldc_zeroEq`) 
for imposing Integral continuity (``NormalDotVec``),
``AdvectionDiffusion`` and ``GradNormal`` are specified and the variable to
compute is defined for the ``GradNormal`` and ``AdvectionDiffusion``.

Also, it is possible to detach certain variables from the computation 
graph in a particular equation if you want to decouple it from 
rest of the equations. This uses the ``.detach()`` method in the backend 
to `stop the computation of gradient <https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html>`_ . 
In this
problem, you can stop the gradient calls on :math:`u` , :math:`v`. This
prevents the network from optimizing :math:`u` , and :math:`v` to
minimize the residual from the advection equation. In this way, you can
make the system one way coupled, where the heat does not influence the
flow, but the flow influences the heat. Decoupling the computations this way can help 
the convergence behavior. 

.. literalinclude:: ../../../examples/three_fin_2d/heat_sink.py
   :language: python
   :lines: 102-129 


Setting up the Domain and adding Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The boundary conditions described in the problem statement are
implemented in the code shown below. Key ``'normal_gradient_c'`` is
used to set the gradient boundary conditions. A new variable
:math:`c` is defined for the solving the advection diffusion equation.

.. math::
   :label: c_non_dim

   c=(T_{actual} - T_{inlet})/273.15

In addition to the continuity and Navier-Stokes equations in 2D, you will have to 
solve the advection diffusion equation :eq:`advection_diffusion_eqn` (with no source
term) in the interior. The thermal diffusivity :math:`D`
for this problem is 0.002 :math:`m^2/s`.

.. math::
   :label: advection_diffusion_eqn
   
   u c_{x} + v c_{y} = D (c_{xx} + c_{yy})

You can use integral continuity planes to specify the targeted mass flow rate through these planes. 
For a parabolic velocity of 1.5
:math:`m/s`, the integral mass flow is :math:`1` which is added as an
additional constraint to speed up the convergence. Similar to tutorial
:ref:`Introductory Example` you can define keys for ``'normal_dot_vel'`` on
the plane boundaries and set its value to ``1`` to specify the targeted
mass flow. Use the ``IntegralBoundaryConstraint`` constraint to define 
these integral constraints. Here the ``integral_line`` 
geometry was created with a symbolic variable for the line's x position, ``x_pos``. 
The ``IntegralBoundaryConstraint`` constraint will randomly generate samples 
for various ``x_pos`` values. The number of such samples can be controlled by ``batch_size`` argument,
while the points in each sample can be configured by ``integral_batch_size`` argument. 
The range for the symbolic variables (in this case ``x_pos``) can be set using the ``parameterization`` argument. 

These planes (lines for 2D geometry) would then be used when
the ``NormalDotVec`` PDE that will compute the dot product of normal components of the
geometry and the velocity components. The parabolic profile can be created by using the ``parabola`` function
by specifying the variable for sweep, the two intercepts and max height.

.. literalinclude:: ../../../examples/three_fin_2d/heat_sink.py
   :language: python
   :lines: 131-214


Adding Monitors and Validators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The validation data comes from a 2D simulation computed using OpenFOAM and the code 
to import it can be found below. 

.. literalinclude:: ../../../examples/three_fin_2d/heat_sink.py
   :language: python
   :lines: 216-289


Training the model 
------------------

Once the python file is setup, the training can be started by executing the python script.

.. code:: bash

   python heat_sink.py

Results and Post-processing
---------------------------

The results for the PhysicsNeMo Sym simulation are compared against the OpenFOAM
data in :numref:`fig-2d_heat_sink_results`.

.. _fig-2d_heat_sink_results:

.. figure:: ../../images/user_guide/heatSink_try4.png
   :alt: Left: PhysicsNeMo Sym. Center: OpenFOAM. Right: Difference
   :name: fig:2d_heat_sink_results
   :width: 100.0%
   :align: center

   Left: PhysicsNeMo Sym. Center: OpenFOAM. Right: Difference
