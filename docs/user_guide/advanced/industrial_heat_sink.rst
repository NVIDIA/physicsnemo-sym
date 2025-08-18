.. _limerock:

Industrial Heat Sink
====================

Introduction
------------

This tutorial uses PhysicsNeMo Sym to conduct a thermal simulation of
NVIDIA’s NVSwitch heatsink. You will learn:

#. How to use hFTB algorithm to solve conjugate heat transfer problems

#. How to build a gPC based Surrogate via Transfer Learning

.. note::
   This tutorial assumes you have completed tutorial :ref:`transient-navier-stokes`
   as well as the tutorial :ref:`cht` on conjugate heat transfer.

Problem Description
-------------------

This tutorial solves the conjugate heat transfer problem of NVIDIA's NVSwitch
heat sink as shown in :numref:`fig-limerock_original`. Similar to the
previous FPGA problem, the heat sink is placed in a channel with inlet
velocity similar to its operating conditions. This case differs from the FPGA one, because you will be using the real heat properties for atmospheric air and copper as the heat sink
material. Unlike :ref:`2d_heat`, a hFTB algorithm will be used to handle the large conductivity differences. 

.. _fig-limerock_original:

.. figure:: ../../images/user_guide/limerock_original.png
   :alt: NVSwitch heat sink geometry
   :width: 25.0%
   :align: center
   :name: fig:limerock_original

   NVSwitch heat sink geometry

Using real heat properties causes an issue on the interface between the
solid and fluid because the conductivity is around 4 orders of magnitude
different (Air: 0.0261 :math:`W/m.K` and Copper: 385 :math:`W/m.K`). 
To remedy this, PhysicsNeMo Sym has a static
conjugate heat transfer approached referred to as heat transfer
coefficient forward temperature backward or hFTB
[#school2018stability]_. This method works by
iteratively solving for the heat transfer in the fluid and solid where
they are one way coupled. Using the hFTB method, assign Robin
boundary conditions on the solid interface and Dirichlet boundaries for
the fluid. The simulation starts by giving an initial guess for the
solid temperature and uses a hyper parameter :math:`h` for the Robin
boundary conditions. A description of the algorithm is shown in :numref:`fig-hFTB_algorithm`. A more 
complete description can be found here [#school2018stability]_.

.. _fig-hFTB_algorithm:

.. figure:: ../../images/user_guide/hftb_algorithm.png
   :alt: hFTB algorithm
   :width: 70.0%
   :align: center
   :name: fig:hFTB_algorithm

   hFTB algorithm

Case Setup
----------

The case setup for this problem is similar to the FPGA and three fin
examples (covered in tutorials :ref:`ParameterizedSim` and :ref:`fpga`)
however, this section shows construction of multiple train domains to implement the hFTB
method.

.. note:: The python script for this problem can be found at ``examples/limerock/limerock_hFTB``.

Defining Domain
~~~~~~~~~~~~~~~

This case setup skips over several sections of the code and
only focuses on the portions related to the hFTB algorithm. You should 
be familiar with how to set up the flow simulation from
previous tutorials. Geometry construction is not discussed in detail as well 
and all relevant information can be found in ``examples/limerock/limerock_hFTB/limerock_geometry.py``. 
The code description begins by defining the parameters of the
simulation and importing all needed modules.

.. literalinclude:: ../../../examples/limerock/limerock_hFTB/limerock_properties.py
   :language: python
   :lines: 1-

.. note:: We nondimensionalize all parameters so that the scales for velocity, temperature, and pressure are roughly in the range 0-1. Such nondimensionalization trains the Neural network more efficiently.

Sequence Solver
~~~~~~~~~~~~~~~

Now setup the solver. Similar to the moving time window
implementation in Tutorial :ref:`transient-navier-stokes`, construct
a separate neural network that stores the thermal solution from the
previous cycles fluid solution. We suggest that this problem is either run on
:math:`8` GPUs or gradient aggregation frequency is set to :math:`8`. Details
on running with multi-GPUs and multi-nodes can be found in tutorial
:ref:`performance` and the details on using gradient aggregation
can be found in tutorial :ref:`config`.

Next, set up a train domain to only solve for the temperature in
the fluid given a Dirichlet boundary condition on the solid. This will
be the first stage of the hFTB method. After getting this initial solution
for the temperature in the fluid solve for the main loop of the hFTB algorithm. 
Now you will solve for both the fluid and solid in a one way coupled manner. The Robin boundary
conditions for the solid are coming from the previous iteration of the
fluid solution.

.. note:: 
   Sometimes for visualization purposes it is beneficial to visualize the
   results on a mesh. Here, this is done using the ``VTKUniformGrid`` method.
   Note that the SDF was used as a mask function to filter out the temperature evaluations outside the
   solid.

.. warning::

   Multi-GPU training is currently not supported for this problem.

.. literalinclude:: ../../../examples/limerock/limerock_hFTB/limerock_thermal.py
   :language: python
   :lines: 1-


Results and Post-processing
---------------------------

To confirm the accuracy of the model, the results are compared for
pressure drop and peak temperature with the OpenFOAM and a commercial
solver results, and the results are reported in :numref:`table-limerock1`. The results
show good accuracy achieved by the hFTB method. :numref:`table-limerock2` 
demonstrates the impact of mesh refinement on
the solution of the commercial solver where with increasing mesh density
and mesh quality, the commercial solver results show convergence towards
the PhysicsNeMo Sym results. A visualization of the heat sink temperature
profile is shown in :numref:`fig-limerock_thermal`.

.. _table-limerock1:

.. table:: A comparison for the solver and PhysicsNeMo Sym results for NVSwitch pressure drop and peak temperature.
   :align: center

   +----------------------+---------------+---------------+-------------------+
   | Property             | OpenFOAM      | Commercial    | PhysicsNeMo Sym   |
   |                      |               | Solver        |                   |
   +----------------------+---------------+---------------+-------------------+
   | Pressure Drop        |               |               |                   |
   | :math:`(Pa)`         | :math:`133.96`| :math:`137.50`| :math:`150.25`    |
   +----------------------+---------------+---------------+-------------------+
   | Peak                 | :math:`93.41` | :math:`95.10` | :math:`97.35`     |
   | Temperature          |               |               |                   |
   | :math:`(^{\circ} C)` |               |               |                   |
   +----------------------+---------------+---------------+-------------------+


.. _table-limerock2:

.. table:: Commercial solver mesh refinement results for NVSwitch pressure drop and peak temperature.
   :align: center
   
   +--------------------+----------------------------------------------+----------------------------------------------+
   | Number of elements | Pressure drop (Pa)                           | Peak temperature :math:`(^{\circ} C)`        |
   |                    +-------------------+-----------------+--------+-------------------+-----------------+--------+
   |                    | Commercial solver | PhysicsNeMo Sym | % diff | Commercial solver | PhysicsNeMo Sym | % diff |
   +--------------------+-------------------+-----------------+--------+-------------------+-----------------+--------+
   | 22.4 M             | 81.27             | 150.25          | 84.88  | 97.40             | 97.35           | 0.05   |
   +--------------------+-------------------+-----------------+--------+-------------------+-----------------+--------+
   | 24.7 M             | 111.76            | 150.25          | 34.44  | 95.50             | 97.35           | 1.94   |
   +--------------------+-------------------+-----------------+--------+-------------------+-----------------+--------+
   | 26.9 M             | 122.90            | 150.25          | 22.25  | 95.10             | 97.35           | 2.36   |
   +--------------------+-------------------+-----------------+--------+-------------------+-----------------+--------+
   | 30.0 M             | 132.80            | 150.25          | 13.14  | -                 | -               | -      |
   +--------------------+-------------------+-----------------+--------+-------------------+-----------------+--------+
   | 32.0 M             | 137.50            | 150.25          | 9.27   | -                 | -               | -      |
   +--------------------+-------------------+-----------------+--------+-------------------+-----------------+--------+


.. _fig-limerock_thermal:

.. figure:: ../../images/user_guide/limerock_thermal.png
   :alt: NVSwitch Solid Temperature
   :width: 50.0%
   :align: center
   :name: fig:limerock_thermal

   NVSwitch Solid Temperature

.. _limerock_gPC_surrogate:

gPC Based Surrogate Modeling Accelerated via Transfer Learning
--------------------------------------------------------------

Previously, Chapter :ref:`ParameterizedSim` 
showed that by parameterizing the input of the neural network, you can
solve for multiple design parameters in a single run and use that
parameterized network for design optimization. This section
introduces another approach for parameterization and design optimization,
which is based on constructing a surrogate using the solution obtained
from a limited number of non-parameterized neural network models.
Compared to the parameterized network approach that is limited to the
CSG module, this approach can be used for parameterization of both
constructive solid and STL geometries, and additionally, can offer
improved accuracy specially for cases with a high-dimensional parameter
space and also in cases where some or all of the design parameters are
discrete. However, this approach requires training of multiple neural
networks and may require multi-node resources.

This section focuses on surrogates based on the generalized Polynomial
Chaos (gPC) expansions. The gPC is an efficient tool for uncertainty
quantification using limited data, and in introduced in Section
:ref:`generalized_polynomial_chaos`. It
starts off by generating the required number of realizations form the
parameter space using a low discrepancy sequence such as Halton or Sobol. 
Next, for each realization, a separate neural network
model is trained. Note that these trainings are independent from each other and
therefore, this training step is embarrassingly parallel and can be done
on multiple GPUs or nodes. Finally, a gPC surrogate is trained that maps
the parameter space to the quantities of interest (e.g., pressure drop
and peak temperature in the heat sink design optimization problem).

In order to reduce the computational cost of this approach associated
with training of multiple models, transfer learning is used, that is,
once a model is fully trained for a single realization, it is used for
initialization of the other models, and this can significantly reduce
the total time to convergence. Transfer learning has been previously
introduced in Chapter :ref:`stl`.

Here, to illustrate the gPC surrogate modeling accelerated via transfer
learning, consider the NVIDIA’s NVSwitch heat sink introduced above.
We introduce four geometry parameters related to fin cut angles, as shown in 
:numref:`fig-limerock_parameterized_geometry`. We then construct a
pressure drop surrogate. Similarly, one can also construct a surrogate
for the peak temperature and use these two surrogates for design
optimization of this heat sink.

.. _fig-limerock_parameterized_geometry:

.. figure:: ../../images/user_guide/limerock_parameterized_geometry.png
   :alt: NVSwitch heat sink geometry parameterization. Each parameter ranges between 0 and :math:`\pi/6`.
   :width: 30.0%
   :align: center
   :name: ig:limerock_parameterized_geometry

   NVSwitch heat sink geometry parameterization. Each parameter ranges
   between 0 and :math:`\pi/6`.

The scripts for this example are available at ``examples/limerock/limerock_transfer_learning``.
Following Section
:ref:`generalized_polynomial_chaos`, one can
generate 30 geometry realizations according to a Halton sequence by
running ``sample_generator.py``, as follows

.. literalinclude:: ../../../examples/limerock/limerock_transfer_learning/sample_generator.py
   :language: python
   :lines: 1-

Then train a separate flow network for each of these realizations
using transfer learning. To do this, update the configs for network checkpoint,
learning rate and decay rate, and the maximum training iterations in
``conf/config.py``. Also change the ``sample_id``
variable in ``limerock_geometry.py``, and then run ``limerock_flow.py``.
This is repeated until all of the geometry realizations are covered.
These flow models are initialized using the trained network for the base
geometry (as shown in :numref:`fig-limerock_original`), and are
trained for a fraction of the total training iterations for the base
geometry, with a smaller learning rate and a faster learning rate decay,
as specified in ``conf/config.yaml``.
This is because you only need to fine-tune these models as opposed to
training them from the scratch. Please note that, before you launch the
transfer learning runs, a flow network for the base geometry needs to be
fully trained.

:numref:`fig-limerock_pce_pressure` shows the front and back
pressure results for different runs. It is evident that the pressure has
converged faster in the transfer learning runs compared to the base
geometry full run, and that transfer learning has reduced the total time
to convergence by a factor of 5.

.. _fig-limerock_pce_pressure:

.. figure:: ../../images/user_guide/limerock_pce_pressure.png
   :alt: NVSwitch front and back pressure convergence results for different geometries using transfer learning.
   :name: fig:limerock_pce_pressure
   :align: center
   :width: 90.0%

   NVSwitch front and back pressure convergence results for different
   geometries using transfer learning.

Finally, randomly divide the pressure drop data obtained from these
models into training and test sets, and construct a gPC surrogate, as follows:

.. literalinclude:: ../../../examples/limerock/limerock_transfer_learning/limerock_pce_surrogate.py
   :language: python
   :lines: 1-

The code for constructing this surrogate is available at ``limerock_pce_surrogate.py``:
:numref:`fig-limerock_tests` shows the gPC surrogate
performance on the test set. The relative errors are below 1%, showing
the good accuracy of the constructed gPC pressure drop surrogate.

.. _fig-limerock_tests:

.. figure:: ../../images/user_guide/limerock_pce_test.png
   :alt: The gPC pressure drop surrogate accuracy tested on four geometries
   :align: center
   :width: 90.0%

   The gPC pressure drop surrogate accuracy tested on four geometries



.. rubric:: References

.. [#school2018stability] Sebastian Scholl, Bart Janssens, and Tom Verstraete. Stability of static conjugate heat transfer coupling approaches using robin interface conditions. Computers & Fluids, 172, 06 2018.
