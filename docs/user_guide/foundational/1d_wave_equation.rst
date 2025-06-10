.. _transient:

1D Wave Equation
================================

Introduction
------------

This tutorial, walks you through the process of setting up a
custom PDE in PhysicsNeMo Sym. It demonstrates the process on a
time-dependent, simple 1D wave equation problem. It
also shows how to solve transient physics in PhysicsNeMo Sym. In this tutorial you
will learn the following:

#. How to write your own Partial Differential Equation and boundary
   conditions in PhysicsNeMo Sym.

#. How to solve a time-dependent problem in PhysicsNeMo Sym.

#. How to impose initial conditions and boundary conditions for a transient problem.

#. How to generate validation data from analytical solutions.


.. note::
   This tutorial assumes that you have completed the 
   :ref:`Introductory Example` tutorial and have familiarized
   yourself with the basics of PhysicsNeMo Sym APIs.

Problem Description
-------------------

In this tutorial, you will solve a simple 1D wave equation . The wave is
described by the below equation.

.. math::

   \begin{aligned}
   \begin{split}\label{transient:eq1}
   u_{tt} & = c^2 u_{xx}\\
   u(0,t) & = 0, \\
   u(\pi, t) & = 0,\\
   u(x,0) & = \sin(x), \\
   u_t(x, 0) & = \sin(x). \\
   \end{split}\end{aligned}

Where, the wave speed :math:`c=1` and the analytical solution to the
above problem is given by :math:`\sin(x)(\sin(t) + \cos(t))`.

Writing custom PDEs and boundary/initial conditions
---------------------------------------------------

In this tutorial, you will write the `1D wave equation
<https://en.wikipedia.org/wiki/Wave_equation>`_ using PhysicsNeMo Sym APIs. You will also see how to
handle derivative type boundary conditions. The PDEs defined in the
source directory ``physicsnemo/eq/pdes/`` can be used for reference.

In this tutorial, you will defined the 1D wave equation in a ``wave_equation.py`` script. 
The ``PDES`` class allows you to write the equations
symbolically in Sympy. This allows you to quickly write your
equations in the most natural way possible. The Sympy equations are
converted to Pytorch expressions in the back end and can also be
printed to ensure correct implementation.

First create a class ``WaveEquation1D`` that inherits from
``PDES``.

.. literalinclude:: ../../../examples/wave_equation/wave_equation.py
   :language: python
   :lines: 15-23
 
Now create the initialization method for this class that defines
the equation(s) of interest. You will define the wave equation using
the wave speed (:math:`c` ). If :math:`c` is given as a string you will
convert it to functional form. This will allow you to solve problems with
spatially/temporally varying wave speed. This is also used in the
subsequent inverse example.

Below code block shows the definition of the PDEs. First, the input variables :math:`x` and :math:`t` are defined with Sympy
symbols. Then the functions for :math:`u` and :math:`c` that
are dependent on :math:`x` and :math:`t` are defined. Using these you can write 
the simple equation :math:`u_{tt} = (c^2 u_x)_x`. Store this equation
in the class by adding it to the dictionary of equations.


.. literalinclude:: ../../../examples/wave_equation/wave_equation.py
   :language: python
   :lines: 40-

Note the rearrangement of the equation for ``'wave_equation'``. You will have
to move all the terms of the PDE either to LHS or RHS and just have the
source term on one side. This way, while using the equations in the constraints, you can assign a custom source function to the ``'wave_equation'`` key instead of 0 to add the source to the PDE.

Once you have written your own PDE for the wave equation, you can verify the
implementation, by refering to the script ``physicsnemo/eq/pdes/wave_equation.py`` from PhysicsNeMo Sym source. 
Also, once you have understood the
process to code a simple PDE, you can easily extend the procedure for
different PDEs in multiple dimensions (2D, 3D, etc.) by making additional
input variables, constants, etc. You can also bundle multiple PDEs
together in a class definition by adding new keys to the equations dictionary.

Now you can write the solver file where you can make use of the newly
coded wave equation to solve the problem.

Case Setup
----------

This tutorial uses ``Line1D`` to sample points in a
single dimension. The time-dependent equation is solved by supplying
:math:`t` as a variable parameter to the ``parameterization`` argument , with the
ranges being the time domain of interest. ``parameterization`` argument is also used
when solving problems involving variation in geometric or variable PDE
constants.

.. note:: 

 *  This solves the problem by treating time as a continuous variable. The examples of discrete time stepping in the form of continuous time window approach that is presented in :ref:`transient-navier-stokes`.
 
 * The python script for this problem can be found at ``examples/wave_equation/wave_1d.py``. The PDE coded in ``wave_equation.py`` is also in the same directory for reference.

Importing the required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new packages/modules imported in this tutorial are ``geometry_1d``
for using the 1D geometry. Import ``WaveEquation1D`` from
the file you just created.

.. literalinclude:: ../../../examples/wave_equation/wave_1d.py
   :language: python
   :lines: 15-31
 
Creating Nodes and Domain
~~~~~~~~~~~~~~~~~~~~~~~~~

This part of of the problem is similar to the tutorial :ref:`Introductory Example`.
``WaveEquation`` class is used to compute the wave equation and the
wave speed is defined based on the problem statement. A neural network with ``x`` and ``t`` as input and ``u`` as output is also created. 

.. literalinclude:: ../../../examples/wave_equation/wave_1d.py
   :language: python
   :lines: 34-43,52-53
 
Creating Geometry and Adding Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For generating geometry of this problem, use the
``Line1D(pt1, pt2)``. The boundaries for ``Line1D`` are the end points
and the interior covers all the points in between the two endpoints.

As described earlier, use the ``parameterization`` argument to
solve for time. To define the initial conditions, set
``parameterization={t_symbol: 0.0}``. You will solve the wave equation for
:math:`t=(0, 2\pi)`. The derivative boundary condition can be handled by
specifying the key ``'u__t'``. The derivatives of the variables can be
specified by adding ``'__t'`` for time derivative and ``'__x'`` for
spatial derivative (``'u__x'`` for :math:`\partial u/\partial x`,
``'u__x__x'`` for :math:`\partial^2 u/\partial x^2`, etc.).

The below code uses these tools to generate the geometry, initial/boundary
conditions and the equations.

.. literalinclude:: ../../../examples/wave_equation/wave_1d.py
   :language: python
   :lines: 45-51,55-84
 
Adding Validation data from analytical solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this problem, the analytical solution can be solved
simultaneously instead of importing a .csv file. This code shows
the process define such a dataset:

.. literalinclude:: ../../../examples/wave_equation/wave_1d.py
   :language: python
   :lines: 86-100
 
Results
-------

The figure below shows the comparison of PhysicsNeMo Sym results with the analytical solution. You can see that the error in PhysicsNeMo Sym prediction increases as the time increases. Some advanced approaches to tackle transient problems are covered in :ref:`transient-navier-stokes`. 

.. figure:: /images/user_guide/try12.png
   :alt: Left: PhysicsNeMo Sym. Center: Analytical Solution. Right: Difference
   :name: fig:wave1
   :width: 100.0%
   :align: center

   Left: PhysicsNeMo Sym. Center: Analytical Solution. Right: Difference

Temporal loss weighting and time marching schedule
--------------------------------------------------

Two simple tricks, namely temporal loss weighting and time marching schedule, can improve the
performance of the continuous time approach for transient simulations. The idea behind the temporal loss weighting is
to weight the loss terms temporally such that the terms corresponding to earlier times have a larger weight compared to
those corresponding to later times in the time domain. For example, the temporal loss weighting can take the following
linear form

.. math::

   \lambda_T = C_T \left( 1 - \frac{t}{T} \right) + 1


Here, :math:`\lambda_T` is the temporal loss weight, :math:`C_T` is a constant that controls the weight scale, :math:`T` is the upper bound for the time
domain, and :math:`t` is time.


The idea behind the time marching schedule is to consider the time domain upper bound T to be variable and a function
of the training iterations. This variable can then change such that more training iterations are taken for the earlier times
compared to later times. Several schedules can be considered, for instance, you can use the following

.. math::

   T_v (s) = \min \left( 1, \frac{2s}{S} \right)


Where :math:`T_v (s)` is the variable time domain upper bound, :math:`s` is the training iteration number, and :math:`S` is the 
maximum number of training iterations. At each training iteration, PhysicsNeMo Sym will then sample continuously from the time domain in the range of :math:`[0, T_v (s)]`.

The below figures show the PhysicsNeMo Sym validation error for models trained with and without using temporal loss weighting and time 
marching for transient 1D, 2D wave examples and a 2D channel flow over a bump. It is evident that these two simple tricks
can improve the training accuracy. 

.. figure:: /images/user_guide/continuous_time_vs_temporal_marching_1.png
   :alt:  PhysicsNeMo Sym validation error for the 1D transient wave example: (a) standard continuous time approach; (b) continuous time approach with temporal loss weighting and time marching.
   :width: 60%
   :align: center

   PhysicsNeMo Sym validation error for the 1D transient wave example: (a) standard continuous time approach; (b) continuous time approach with temporal loss weighting and time marching.

.. figure:: /images/user_guide/continuous_time_vs_temporal_marching_2.png
   :alt: PhysicsNeMo Sym validation error for the 2D transient wave example: (a) standard continuous time approach; (b) continuous time approach with temporal loss weighting and time marching. 
   :width: 60%
   :align: center

   PhysicsNeMo Sym validation error for the 2D transient wave example: (a) standard continuous time approach; (b) continuous time approach with temporal loss weighting and time marching.


.. figure:: /images/user_guide/continuous_time_vs_temporal_marching_3.png
   :alt: PhysicsNeMo Sym validation error for a 2D transient channel flow over a bump: (a) standard continuous time approach; (b) continuous time approach with temporal loss weighting.
   :width: 80%
   :align: center

   PhysicsNeMo Sym validation error for a 2D transient channel flow over a bump: (a) standard continuous time approach; (b) continuous time approach with temporal loss weighting. 

