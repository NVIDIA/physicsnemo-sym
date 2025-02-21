.. _wave_propagation:

2D Seismic Wave Propagation
============================

Introduction
------------

This tutorial extends the previous 1D wave equation example and
solve a 2D seismic wave propagation problem commonly used in seismic
surveying. In this tutorial, you will learn the following:

#. How to solve a 2D time-dependent problem in PhysicsNeMo Sym

#. How to define an open boundary condition using custom equations

#. How to present a variable velocity model as an additional network

.. note:: 
   This tutorial assumes that you have completed the :ref:`Introductory Example` tutorial
   on Lid Driven Cavity flow and have familiarized yourself with the basics
   of PhysicsNeMo Sym. Also, see the 
   :ref:`transient` tutorial for more information on defining new
   differential equations, and solving time-dependent problems in PhysicsNeMo Sym.

Problem Description
-------------------

The acoustic wave equation for the square slowness, defined as
:math:`m=1/c^2` where :math:`c` is the speed of sound (velocity) of a
given physical medium with constant density, and a source :math:`q` is
given by:

.. math:: u_{tt} = c^2u_{xx} + c^2u_{yy} + q \quad \mbox{ in } \Omega

Where :math:`u(\mathbf{x},t)` represents the pressure response (known as
the "wavefield") at location vector :math:`\mathbf{x}` and time :math:`t` in
an acoustic medium. Despite its linearity, the wave equation is
notoriously challenging to solve in complex media, because the dynamics
of the wavefield at the interfaces of the media can be highly complex, with
multiple types of waves with large range of amplitudes and frequencies
interfering simultaneously.

In this tutorial, you will solve the 2D acoustic wave equation with a single
Ricker Source in a layered velocity model, 1.0 :math:`km/s` at the top
layer and 2.0 :math:`km/s` the bottom (:numref:`fig-seismic-wave-velocity`). Sources in
seismic surveys are positioned at a single or a few physical locations
where artificial pressure is injected into the domain you want to model.
In the case of land survey, it is usually dynamite blowing up at a given
location, or a vibroseis (a vibrating plate generating continuous sound
waves). For a marine survey, the source is an air gun sending a bubble
of compressed air into the water that will expand and generate a seismic
wave.

This problem, uses a domain size 2 :math:`km` x 2 :math:`km`, and a
single source is located at the center of the domain. The source
signature is modelled using a Ricker wavelet, illustrated in :numref:`fig-seismic-wave-source`, with a
peak wavelet frequency of :math:`f_0=15 Hz`.

.. _fig-seismic-wave-velocity:

.. figure:: /images/user_guide/velocity_model.png
   :alt: Velocity model of the medium
   :width: 40.0%
   :align: center

   Velocity model of the medium

.. _fig-seismic-wave-source:

.. figure:: /images/user_guide/ricker_source.png
   :alt: Ricker source signature
   :width: 40.0%
   :align: center

   Ricker source signature

The problem uses wavefield data at time steps (150 :math:`ms` - 300
:math:`ms`) generated from finite difference simulations, using `Devito <https://github.com/devitocodes/devito/tree/master../../../examples/seismic/tutorials>`_, as constraints for the temporal boundary conditions, and train
PhysicsNeMo Sym to produce wavefields at later time steps (300 :math:`ms` – 1000
:math:`ms`).

Problem Setup
-------------

The setup for this problem is similar to tutorial :ref:`transient`. So, only
the main highlights of this problem is discussed here.

.. note::
   The full python script for this problem can be found at ``examples/seismic_wave/wave_2d.py``

Defining Boundary Conditions
----------------------------

A second-order PDE requires strict BC on both the initial wavefield and
its derivatives for its solution to be unique. In the field, the seismic
wave propagates in every direction to an "infinite" distance. In the
finite computational domain, Absorbing BC (ABC) or Perfectly Matched
Layers (PML) are artificial boundary conditions that are typically applied to approximate an infinite media by
damping and absorbing the waves at the edges of the domain, in order to
avoid reflections from the boundary. However, a NN solver is meshless – it
is not suitable to implement ABC or PML. To enable a wave to leave the
computational domain and travel undisturbed through the boundaries,
you can apply an `open boundary condition <http://hplgit.github.io/wavebc/doc/pub/._wavebc_cyborg002.html>`_, also called a radiation
condition, which imposes the first-order PDEs at the boundaries:

.. math::

   \begin{split}
   \frac{\partial u}{\partial t} - c\frac{\partial u}{\partial x} &= 0,\quad \mbox{ at } x = 0 \\
   \frac{\partial u}{\partial t} + c\frac{\partial u}{\partial x} &= 0,\quad \mbox{ at } x = dLen \\
   \frac{\partial u}{\partial t} - c\frac{\partial u}{\partial y} &= 0,\quad \mbox{ at } y = 0 \\
   \frac{\partial u}{\partial t} + c\frac{\partial u}{\partial y} &= 0,\quad \mbox{ at } y = dLen \\
   \end{split}

Previous tutorials have described how to define custom PDEs. Similarly
here you create a class ``OpenBoundary`` that inherits from ``PDES``:

.. literalinclude:: ../../../examples/seismic_wave/wave_2d.py
   :language: python
   :lines: 63-133

Variable Velocity Model
------------------------

In the :ref:`transient` tutorial, the velocity (:math:`c`) of the
physical medium is constant. In this problem, you have a velocity model
that varies with locations :math:`x`. Use a :math:`tanh` function
form to represent the velocity :math:`c` as a function of :math:`x` and
:math:`y`:

.. note::
  A :math:`tanh` function was used to avoid the sharp discontinuity at the interface. Such smoothing of sharp boundaries helps the convergence of the Neural network solver.

.. literalinclude:: ../../../examples/seismic_wave/wave_2d.py
   :language: python
   :lines: 189-198

Creating PDE and Neural Network Nodes
--------------------------------------

First, define the PhysicsNeMo Sym PDE and NN nodes for this problem:

.. literalinclude:: ../../../examples/seismic_wave/wave_2d.py
   :language: python
   :lines: 154-176

The PDE nodes are created using both the ``OpenBoundary`` PDE defined above and the ``WaveEquation`` PDE defined within PhysicsNeMo Sym. Since this is a time-dependent problem, use the ``time=True`` arguments.

Here two neural network nodes are defined where ``wave_net`` is used to learn the solution to the wave equation and ``speed_net`` is used to learn the velocity model given the training data (``wave_speed_invar`` and ``wave_speed_outvar``) defined above.

Creating Geometry
-----------------

The 2D geometry of the computational domain as well as the time range of the solution is defined:

.. literalinclude:: ../../../examples/seismic_wave/wave_2d.py
   :language: python
   :lines: 178-180

Adding Constraints
------------------

A total of four different constraints is sufficient to define this problem. The first constraint is a simple supervised constraint which matches the velocity model to the training data above. The second is another supervised constraint, constraining the wavefield solution to match the numerical wavefield generated using Devito at 4 starting time steps. The third constraint ensures the ``WaveEquation`` PDE is honoured in the interior of the domain, whilst the fourth constraint ensures the ``OpenBoundary`` PDE is honoured along the boundaries of the domain:

.. literalinclude:: ../../../examples/seismic_wave/wave_2d.py
   :language: python
   :lines: 200-245

The supervised constraints are added using the ``PointwiseConstraint.from_numpy(...)`` constructor.

Validation
----------

Finally, you can validate your results using 13 later time steps from Devito, defined within a PhysicsNeMo Sym validator as shown below. 

.. literalinclude:: ../../../examples/seismic_wave/wave_2d.py
   :language: python
   :lines: 247-265

You can define a custom tensorboard plotter within ``examples/seismic_wave/wave_2d.py`` to add validation plots to TensorBoard, see :ref:`tensorboard` for more details on how to do this.

Now that you have defined the model, its constraints and validation data, you can form a solver to train your model. All of the hyperparameters used for the example can be found in the project's Hydra configuration file.

.. literalinclude:: ../../../examples/seismic_wave/wave_2d.py
   :language: python
   :lines: 255

Results
-------

The training results can be viewed in TensorBoard. It can be seen that the PhysicsNeMo Sym
results are noticeably better than Devito, predicting the wavefield with
much less boundary reflections, especially at later time steps.

.. figure:: /images/user_guide/simnet_vs_devito_combined.PNG
   :alt: Comparison of PhysicsNeMo Sym results with Devito solution
   :width: 65.0%
   :align: center

   Comparison of PhysicsNeMo Sym results with Devito solution


