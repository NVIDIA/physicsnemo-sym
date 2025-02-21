.. _ldc_zeroEq:

Turbulent physics: Zero Equation Turbulence Model
=================================================

Introduction
------------

This tutorial walks you through the process of adding a
algebraic (zero equation) turbulence model to the PhysicsNeMo Sym simulations. 
In this tutorial you will learn the following:

#. How to use the Zero equation turbulence model in PhysicsNeMo Sym.

#. How to create nodes in the graph for arbitrary variables.

.. note::
   This tutorial assumes that you have completed the :ref:`Introductory Example`
   tutorial on Lid Driven Cavity Flow and have familiarized yourself with the basics
   of the PhysicsNeMo Sym APIs.

Problem Description
-------------------

In this tutorial you will add the zero equation turbulence for a Lid
Driven Cavity flow. The problem setup is very similar to the one found in the
:ref:`Introductory Example`. The Reynolds number is increased to 1000. The velocity
profile is kept the same as before. To increase the
Reynolds Number, the viscosity is reduced to 1 × 10\ :sup:`−4`
:math:`m^2/s`.
 
Case Setup
----------

The case set up in this tutorial is very similar to the example in :ref:`Introductory Example`. 
It only describes the additions that are made to the previous code.

.. note::
   The python script for this problem can be found at ``examples/ldc/ldc_2d_zeroEq.py``


Importing the required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import PhysicsNeMo Sym' ``ZeroEquation`` to help setup the problem. 
Other import are very similar to previous LDC. 

.. literalinclude:: ../../../examples/ldc/ldc_2d_zeroEq.py
   :language: python
   :lines: 15-37

Defining the Equations, Networks and Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the Navier-Stokes equation, the Zero Equation turbulence
model is included by instantiating the ``ZeroEquation`` equation class.
The kinematic viscosity :math:`\nu` in the Navier-Stokes equation is a now a sympy expression 
given by the ``ZeroEquation``. 
The ``ZeroEquation`` turbulence model provides the
effective viscosity :math:`(\nu+\nu_t)` to the Navier-Stokes equations.
The kinematic viscosity of the fluid calculated based on the Reynolds
number is given as an input to the ``ZeroEquation`` class.

The Zero Equation turbulence model is defined in the equations below. 
Note, :math:`\mu_t = \rho \nu_t`.

.. math::
   
   \mu_t=\rho l_m^2 \sqrt{G}

.. math::

   G=2(u_x)^2 + 2(v_y)^2 + 2(w_z)^2 + (u_y + v_x)^2 + (u_z + w_x)^2 + (v_z + w_y)^2

.. math::

   l_m=\min (0.419d, 0.09d_{max})

Where, :math:`l_m` is the mixing length, :math:`d` is the normal
distance from wall, :math:`d_{max}` is maximum normal distance and
:math:`\sqrt{G}` is the physicsnemo of mean rate of strain tensor.

The zero equation turbulence model requires normal distance
from no slip walls to compute the turbulent viscosity. For most examples, 
signed distance field (SDF) can act as a normal distance. When the geometry is generated 
using either the PhysicsNeMo Sym' geometry module/tesselation module you have access to the ``sdf``
variable similar to the other coordinate variables when used in interior sampling. Since 
zero equation also computes the derivatives of the viscosity, when using the ``PointwiseInteriorConstraint``, 
you can pass an argument that says ``compute_sdf_derivatives=True``. This will compute 
the required derivatives of the SDF like ``sdf__x``, ``sdf__y``, etc. 


.. literalinclude:: ../../../examples/ldc/ldc_2d_zeroEq.py
   :language: python
   :lines: 43-60

Setting up domain, adding constraints and running the solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section of the code is very similar to LDC tutorial, so 
the code and final results is presented here. 

.. literalinclude:: ../../../examples/ldc/ldc_2d_zeroEq.py
   :language: python
   :lines: 62-

The results of the turbulent lid driven cavity flow are shown below. 

.. figure:: /images/user_guide/try_decoupled_inference.png
   :alt: Visualizing variables from Inference domain
   :name: fig:zeroInference
   :width: 40.0%
   :align: center

   Visualizing variables from Inference domain

.. figure:: /images/user_guide/try_decoupled.png
   :alt: Comparison with OpenFOAM data. Left: PhysicsNeMo Sym Prediction. Center: OpenFOAM, Right: Difference
   :name: fig:zeroValidation
   :width: 80.0%
   :align: center

   Comparison with OpenFOAM data. Left: PhysicsNeMo Sym Prediction. Center:
   OpenFOAM, Right: Difference
