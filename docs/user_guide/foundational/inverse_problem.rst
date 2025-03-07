.. _inverse:

Inverse Problem: Finding Unknown Coefficients of a PDE
======================================================

Introduction
------------

In this tutorial, you will use PhysicsNeMo Sym to solve an inverse problem by
assimilating data from observations. You will use the flow field computed
by OpenFOAM as an input to the PINNs whose job is to predict the
parameters characterizing the flow (eg. viscosity (:math:`\nu`) and
thermal diffusivity (:math:`\alpha`)). In this tutorial you will learn:

#. How to assimilate analytical/experimental/simulation data in PhysicsNeMo Sym.

#. How to use the ``PointwiseConstraint`` in PhysicsNeMo Sym to create constraints from 
   data that can be loaded in form of .csv files/numpy arrays.

#. How to use the assimilated data to make predictions of unknown
   quantities.

.. note:: 
   This tutorial assumes that you have completed tutorial :ref:`Introductory Example` and have familiarized yourself with the basics of the PhysicsNeMo Sym APIs. 

Problem Description
-------------------

In this tutorial, you will predict the fluid’s viscosity and thermal
diffusivity by providing the flow field information obtained from
OpenFOAM simulations as an input to the PINNs. You will use the same 2D
slice from a 3-fin flow field that was used as a validation data in
tutorial :ref:`advection-diffusion`.

To summarize, the training data for this problem is (:math:`u_i`,
:math:`v_i`, :math:`p_i`, :math:`T_i`) from OpenFOAM simulation and the
model is trained to predict (:math:`\nu`, :math:`\alpha`) with the
constraints of satisfying the governing equations of continuity,
Navier-Stokes and advection-diffusion.

The :math:`\nu` and :math:`\alpha` used for the OpenFOAM simulation are
:math:`0.01` and :math:`0.002` respectively.

:math:`T` is scaled to define a new transport variable :math:`c` for
the advection-diffusion equation as shown in equation
:eq:`temp_scaling`.

.. math::
   :label: temp_scaling
   
   \begin{split}
   c &=\frac{T_{actual}}{T_{base}} - 1.0\\
   T_{base} &=293.498 K
   \end{split}

As the majority of diffusion of temperature occurs in the wake of the
heat sink (:numref:`fig-inverse_point_sample`), you can
sample points only in the wake for training the PINNs. You also should also discard
the points close to the boundary as you will train the network to
minimize loss from the conservation laws alone.

.. _fig-inverse_point_sample:

.. figure:: /images/user_guide/inverse_problem.png
   :alt: Batch of training points sampled from OpenFOAM data
   :name: fig:inverse_point_sample
   :width: 50.0%
   :align: center

   Batch of training points sampled from OpenFOAM data

Case Setup
----------

In this tutorial you will use the ``PointwiseConstraint`` class for making 
the training data from .csv file. You will make three networks. The first network will memorize
the flow field by developing a mapping between (:math:`x`, :math:`y`)
and (:math:`u`, :math:`v`, :math:`p`). The second network will memorize the
temperature field by developing a mapping between (:math:`x`, :math:`y`)
and (:math:`c`). The third network will be trained to invert out the
desired quantities viz. (:math:`\nu`, :math:`\alpha`). For this problem,
you will be using ``NavierStokes`` and ``AdvectionDiffusion`` equations
from the PDES module.


.. note:: The python script for this problem can be found at ``examples/three_fin_2d/heat_sink_inverse.py``.

Importing the required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The list of packages/modules to be imported are shown below.

.. literalinclude:: ../../../examples/three_fin_2d/heat_sink_inverse.py
   :language: python
   :lines: 15-43


Defining the Equations, Networks and Nodes for a Inverse problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The process of creating a neural network for an inverse problem is
similar to most of the problems you have seen in previous tutorials.
However, the information for the flow variables, and in turn their
gradients is already present (from OpenFOAM data) for the network to
memorize. Hence, this example detachs these variables in computation graph of 
their respective equations. This means that only the networks predicting
``'nu'`` and ``'D'`` will be optimized to minimize the equation
residuals. The velocity, pressure and their gradients are treated as
ground truth data.

Also, note that the viscosity and diffusivity are passed in as
strings (``'nu'`` and ``'D'`` respectively) to the equations
as they are unknowns in this problem.

Similar to the tutorial :ref:`advection-diffusion`, you will
train two separate networks to memorize flow variables (:math:`u`,
:math:`v` , :math:`p` ), and scalar transport variable (:math:`c`). Also,
because ``'nu'`` and ``'D'`` are the custom variables that were defined, you
will have a separate network ``invert_net_nu`` and ``invert_net_D`` 
that produces :math:`\nu` and :math:`\alpha` at the output nodes.

The code to generate the nodes for the problem is shown here:

.. literalinclude:: ../../../examples/three_fin_2d/heat_sink_inverse.py
   :language: python
   :lines: 48-101


Assimilating data from CSV files/point clouds to create Training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PointwiseConstraint``: Here the tutorial does not solve the problem for the full
domain. Therefore, it does not use the geometry module to create
geometry for sampling points. Instead, you use the point cloud data
in the form of a .csv file. You can use the ``PointwiseConstraint`` class to handle such
input data. This class takes in separate dictionaries for input
variables and output variables. These dictionaries have a key for each
variable and a numpy array of values associated to the key. Also, this tutorial provides a
batch size for sampling this .csv point cloud. This
is done by specifying the required batch size to the ``batch_size``
argument.

Since part of the problem involves memorizing the given flow field, you
will have ``['x', 'y']`` as input keys and
``['u', 'v', 'p', 'c', 'continuity', 'momentum_x', 'momentum_y', 'advection_diffusion']``
as the output keys. Setting ``['u', 'v', 'p', 'c']`` as input values
from OpenFOAM data, you are making the network assimilate the
OpenFOAM distribution of these variables in the selected domain. Setting
``['continuity', 'momentum_x', 'momentum_y', 'advection_diffusion']``
equal to :math:`0`, you also instruct the network to satisfy the PDE losses
at those sampled points. Now, except the :math:`\nu` and :math:`\alpha`,
all the variables in these PDEs are known. Thus the network can use this
information to invert out the unknowns.

The code to generate such a boundary condition is shown here:

.. literalinclude:: ../../../examples/three_fin_2d/heat_sink_inverse.py
   :language: python
   :lines:  103-147


Adding Monitors
~~~~~~~~~~~~~~~

In this tutorial, you will create monitors for the convergence of 
average ``'nu'`` and ``'D'`` inside the domain as the
solution progresses. Once you find that the average value of these
quantities has reached a steady value, you can end the simulation. The
code to generate the ``PointwiseMonitor`` is shown here:

.. literalinclude:: ../../../examples/three_fin_2d/heat_sink_inverse.py
   :language: python
   :lines: 148-163


Training the model 
------------------

Once the python file is setup, the training can be simply started by executing the 
python script.

.. code:: bash

   python heat_sink_inverse.py

Results and Post-processing
---------------------------

You can monitor the Tensorboard plots to see the convergence of the
simulation. The Tensorboard graphs should look similar to the ones shown
in :numref:`fig-inverse-point-result`.


.. table:: Comparison of the inverted coefficients with the actual values
   :align: center

   +----------------------+----------------------+------------------------------+
   | Property             | OpenFOAM (True)      | PhysicsNeMo Sym (Predicted)  |
   +----------------------+----------------------+------------------------------+
   | Kinematic Viscosity  | 1.00 × 10\ :sup:`−2` | 9.87 × 10\ :sup:`−3`         |
   | :math:`(m^2/s)`      |                      |                              |
   +----------------------+----------------------+------------------------------+
   | Thermal Diffusivity  | 2.00 × 10\ :sup:`−3` | 2.53 × 10\ :sup:`−3`         |
   | :math:`(m^2/s)`      |                      |                              |
   +----------------------+----------------------+------------------------------+


.. _fig-inverse-point-result:

.. figure:: /images/user_guide/inverse_problem_results_try2.png
   :alt: Tensorboard plots for :math:`\nu` and :math:`\alpha`
   :width: 60.0%
   :align: center

   Tensorboard plots for :math:`\alpha` and :math:`\nu`
