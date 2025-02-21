.. _ode:

Coupled Spring Mass ODE System
===========================================================

Introduction
------------

In this tutorial PhysicsNeMo Sym is used to solve a system of coupled
ordinary differential equations. Since the APIs used for this problem
have already been covered in a previous tutorial, only the
problem description is discussed without going into the details of the code.

.. note::
   This tutorial assumes that you have completed tutorial :ref:`Introductory Example`
   and have familiarized yourself with the basics of the PhysicsNeMo Sym APIs. 
   Also, refer to tutorial :ref:`transient` for information on defining new
   differential equations, and solving time dependent problems in PhysicsNeMo Sym.
   
Problem Description
-------------------

In this tutorial, a simple spring mass system as shown in
:numref:`fig-spring-mass` is solved. The systems shows three masses attached
to each other by four springs. The springs slide along a frictionless
horizontal surface. The masses are assumed to be point masses and the
springs are massless. The problem is solved so that the masses
(:math:`m's`) and the spring constants (:math:`k's`) are constants, but
they can later be parameterized if you intend to solve the parameterized
problem (Tutorial :ref:`ParameterizedSim`).

The model’s equations are given as below:

.. math::
   :label: ode_eqn

   \begin{split}
   m_1 x_1''(t) &= -k_1 x_1(t) + k_2(x_2(t) - x_1(t)),\\
   m_2 x_2''(t) &= -k_2 (x_2(t) - x_1(t))+ k_3(x_3(t) - x_2(t)),\\
   m_3 x_3''(t) &= -k_3 (x_3(t) - x_2(t)) - k_4 x_3(t). \end{split}

Where, :math:`x_1(t), x_2(t), \text{and } x_3(t)` denote the mass
positions along the horizontal surface measured from their equilibrium
position, plus right and minus left. As shown in the figure, first and
the last spring are fixed to the walls.

For this tutorial, assume the following conditions:

.. math::
   :label: ode_IC

   \begin{split}
   [m_1, m_2, m_3] &= [1, 1, 1],\\
   [k_1, k_2, k_3, k_4] &= [2, 1, 1, 2],\\
   [x_1(0), x_2(0), x_3(0)] &= [1, 0, 0],\\
   [x_1'(0), x_2'(0), x_3'(0)] &= [0, 0, 0].
   \end{split}

.. _fig-spring-mass:

.. figure:: /images/user_guide/spring_mass_drawing.png
   :alt: Three masses connected by four springs on a frictionless surface
   :width: 50.0%
   :align: center

   Three masses connected by four springs on a frictionless surface

Case Setup
----------

The case setup for this problem is very similar to the setup
for the tutorial :ref:`transient`. You can define the
differential equations in ``spring_mass_ode.py`` and then define the
domain and the solver in ``spring_mass_solver.py``.

.. note::
 The python scripts for this problem can be found at ``examples/ode_spring_mass/``.

Defining the Equations
~~~~~~~~~~~~~~~~~~~~~~

The equations of the system :eq:`ode_eqn` can be coded
using the sympy notation similar to tutorial :ref:`transient`.

.. literalinclude:: ../../../examples/ode_spring_mass/spring_mass_ode.py
   :language: python

Here, each parameter :math:`(k's \text{ and } m's)` is written as a function
and is substituted as a number if it's constant. This will allow you to
parameterize any of this constants by passing them as a string.

Solving the ODEs: Creating Geometry, defining ICs and making the Neural Network Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Once you have the ODEs defined, you can easily form the constraints needed for optimization as seen in earlier tutorials. 
This example, uses ``Point1D``
geometry to create the point mass. You also have to define the time range of
the solution and create symbol for time (:math:`t`) to define the
initial condition, etc. in the train domain. This code shows the
geometry definition for this problem. Note that this tutorial does not use the x-coordinate
(:math:`x`) information of the point, it is only used to sample a
point in space. The point is assigned different values for variable
(:math:`t`) only (initial conditions and ODEs over the time range). The code to 
generate the nodes and relevant constraints is below:

.. literalinclude:: ../../../examples/ode_spring_mass/spring_mass_solver.py
   :language: python
   :lines: 15-82

Next, you can define the validation data for this problem. The solution
of this problem can be obtained analytically and the expression can be
coded into dictionaries of numpy arrays for
:math:`x_1, x_2, \text{and } x_3`. This part of the code is similar to
the tutorial :ref:`transient`.

.. literalinclude:: ../../../examples/ode_spring_mass/spring_mass_solver.py
   :language: python
   :lines: 84-103

Now that you have the definitions for the various constraints and domains complete, 
you can form the solver and run the problem.
The code to do the same can be found below:

.. literalinclude:: ../../../examples/ode_spring_mass/spring_mass_solver.py
   :language: python
   :lines: 105-110

Once the python file is setup, you can solve the problem by executing the
solver script ``spring_mass_solver.py`` as seen in other tutorials.

Results and Post-processing
---------------------------

The results for the PhysicsNeMo Sym simulation are compared against the
analytical validation data. You can see that the solution converges to
the analytical result very quickly. The plots can be created
using the ``.npz`` files that are created in the ``validator/`` directory
in the network checkpoint.

.. figure:: /images/user_guide/comparison_ode.png
   :alt: Comparison of PhysicsNeMo Sym results with an analytical solution
   :width: 70.0%
   :align: center

   Comparison of PhysicsNeMo Sym results with an analytical solution.
