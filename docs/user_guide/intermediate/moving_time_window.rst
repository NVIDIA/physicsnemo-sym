.. _transient-navier-stokes:

Moving Time Window: Taylor Green Vortex Decay
==============================================

Introduction
------------

Some of the previous tutorials have shown solving the transient problems
with continuous time approach. This tutorial presents a moving time window
approach to solve a complex transient Navier-Stokes problem. In this
tutorial, you will learn:

#. How to solve sequences of problems/domains in PhysicsNeMo Sym.

#. How to set up periodic boundary conditions.

.. note::
   This tutorial assumes you have completed tutorial on :ref:`transient`.

Problem Description
-------------------

As mentioned in tutorial :ref:`transient`, solving transient
simulations with only the continuous time method can be difficult for long
time durations. This tutorial shows how this can be overcome
by using a moving time window. The example problem is the 3D
Taylor-Green vortex decay at Reynolds number 500. The Taylor-Green
vortex problem is often used as a benchmark to compare solvers and in
this case validation data is generated using a spectral solver. The
domain is a cube of length :math:`2\pi` with periodic boundary
conditions on all sides. The initial conditions you will use are,

.. math::

   \begin{aligned}
   u(x,y,z,0) &= sin(x)cos(y)cos(z) \\
   v(x,y,z,0) &= -cos(x)sin(y)cos(z) \\
   w(x,y,z,0) &= 0 \\
   p(x,y,z,0) &= \frac{1}{16} (cos(2x) + cos(2y))(cos(2z) + 2) \\\end{aligned}

This problem shows solving the time dependent incompressible Navier-Stokes
equations with a density :math:`1` and viscosity :math:`0.002`. Note
that because there are periodic boundaries on all sides you do not need
boundary conditions :numref:`fig-taylor_green_initial_conditions`.

.. _fig-taylor_green_initial_conditions:

.. figure:: ../../images/user_guide/taylor_green_initial_conditions.png
   :alt: Taylor-Green vortex initial conditions.
   :name: fig:tylor_green_initial_conditions
   :align: center
   :width: 50%

   Taylor-Green vortex initial conditions.

The moving time window approach works by iteratively solving for small
time windows to progress the simulation forward. The time windows use
the previous window as new initial conditions. The continuous time
method is used for solving inside a particular window. A figure of this
method can be found here :numref:`fig-moving_time_window` for a
hypothetical 1D problem. Learning rate decay is restarted after each
time window. A similar approach can be found here
[#wight2020solving]_.

.. _fig-moving_time_window:

.. figure:: ../../images/user_guide/moving_time_window_fig.png
   :alt: Moving Time Window Method
   :name: fig:moving_time_window
   :align: center

   Moving Time Window Method

Case Setup
----------

The case setup for this problem is similar to many of the previous
tutorials except for two key differences. This example shows
how to set up a sequence of domains to iteratively solve for. 

.. note:: The python script for this problem can be found at ``examples/taylor_green/``.

Sequence of Train Domains
~~~~~~~~~~~~~~~~~~~~~~~~~

First, construct your geometry similar to the previous problems.

Also, define values for how large the time window will be. In
this case solve to :math:`1` unit of time and then construct 
all needed nodes. 


.. literalinclude:: ../../../examples/taylor_green/taylor_green.py
   :language: python
   :lines: 58-68

The architecture can be created as below.

.. literalinclude:: ../../../examples/taylor_green/taylor_green.py
   :language: python
   :lines: 70-80


Note that the periodicity is set when creating out network architecture.
This will force the network by construction to give periodic solutions on the 
given bounds. Now two domains are defined, one for the initial conditions
and one to be used for all future time windows.

.. literalinclude:: ../../../examples/taylor_green/taylor_green.py
   :language: python
   :lines: 82-152

In the moving time window domain there is a new initial condition that will come from
the previous time window. Now that the domain files have been constructed for the initial conditions and future time windows You can put it all together and make your sequential solver.


Sequence Solver
~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/taylor_green/taylor_green.py
   :language: python
   :lines: 154-162

Instead of using the normal ``Solver`` class use the ``SequentialSolver`` class.
This class expects a list where each element is a tuple of the number of times
solving as well as the given domain. In this case, solve the initial condition domain once and then solve the time window domain multiple times. A ``custom_update_operation`` is provided which is called after solving each iteration and in this case is used to update network parameters.

The iterative domains are solved here in a very general way. 
In subsequent chapters this structure is used to
implement a complex iterative algorithm to solve conjugate heat transfer
problems.

Results and Post-processing
---------------------------

After solving this problem you can visualize the results. Look in
the network directory to see,

``current_step.txt initial_conditions window_0000 window_0001...``

If you look in any of these directories you see the typical files storing
network checkpoint and results. Plotting at time 15 gives the snapshot
:numref:`fig-taylor_green_initial_conditions`. To validate the simulation, 
the results are compared against a spectral solver's results. Comparing
the point-wise error of transient simulations can be misleading as this
is a chaotic dynamical system and any small errors will quickly cause
large differences. Instead, you can look at the average turbulent kinetic
energy decay (TKE) of the simulation. A figure of this can be found
here below.

.. figure:: ../../images/user_guide/taylor_green_re_500.png
   :alt: Taylor-Green vortex at time :math:`15.0`.
   :name: fig:taylor_green_re_500
   :align: center

   Taylor-Green vortex at time :math:`15.0`.

.. figure:: ../../images/user_guide/tke_plot.png
   :alt: Taylor-Green Turbulent kinetic energy decay.
   :name: fig:taylor_green_tke
   :align: center
   :width: 60%

   Taylor-Green Turbulent kinetic energy decay.

.. rubric:: References

.. [#wight2020solving] Colby L Wight and Jia Zhao. Solving allen-cahn and cahn-hilliard equations using the adaptive physics informed neural networks. arXiv preprint arXiv:2007.04542, 2020.
