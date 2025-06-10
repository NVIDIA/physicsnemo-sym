.. _deeponet:

Deep Operator Network
================================================

Introduction
--------------

This tutorial illustrates how to learn abstract operators using data-informed and physics-informed
Deep operator network (DeepONet) in PhysicsNeMo Sym. In this tutorial, you will learn

#. How to use DeepONet architecture in PhysicsNeMo Sym

#. How to set up data-informed and physics-informed DeepONet for learning operators

.. note::
   This tutorial assumes that you have completed the tutorial :ref:`Introductory Example` and are
   familiar with PhysicsNeMo Sym APIs.

Problem 1: Anti-derivative (data-informed)
------------------------------------------

Problem Description
~~~~~~~~~~~~~~~~~~~

Consider a 1D initial value problem

.. math::  \frac{du}{dx} = a(x), \quad x \in [0, 1],

subject to an initial condition :math:`u(0)=0`. The anti-derivative operator :math:`G` over :math:`[0,1]`
given by

.. math:: G:\quad a(x) \mapsto G(a)(x):= \int_0^x a(t) dt, \quad x \in [0,1].

You're going to setup a DeepONet to learn the operator :math:`G`. In this case, the :math:`a` will be the input of branch net and the
:math:`x` will be the input of trunk net. As the input of branch net, :math:`a` is discretized on a fixed uniform grid.
They are not necessary to be the same as the query coordinates :math:`x` at which a DeepONet model is evaluated.
For example, you may give the data of :math:`a` as
:math:`\{a(0),\ a(0.5),\ a(1)\}` but evaluate the output at :math:`\{G(a)(0.1), G(u)(0.8), G(u)(0.9)\}`. This is one of the advantages of
DeepONet compared with Fourier neural operator.

Data Preparation
~~~~~~~~~~~~~~~~

As data preparation, generate :math:`10,000` different input functions :math:`a` from a zero mean Gaussian random field (GRF)
with an exponential quadratic kernel of a length scale :math:`l=0.2`. Then obtain the corresponding :math:`10,000`
ODE solutions :math:`u` using an explicit Runge-Kutta method. For each input output pair of :math:`(a, u)`, it is worth noting
that only one observation of :math:`u(\cdot)` is selected. It highlights the flexibility of DeepONet in terms
of tackling various data structure. The training and validation data are provided in ``/examples/anti_derivative/data/``.
With this data, you can start the data informed DeepONet code.

.. note::

    The python script for this problem can be found at ``/examples/anti_derivative/data_informed.py``.


Case Setup
~~~~~~~~~~

Let us first import the necessary packages.

.. literalinclude:: ../../../examples/anti_derivative/data_informed.py
   :language: python
   :lines: 15-31


Initializing the Model
~~~~~~~~~~~~~~~~~~~~~~

In this case, you will use a fully-connected network as the branch net and a Fourier feature network as the trunk net.
In branch net, the ``Key("a", 100)`` and  ``Key("branch", 128)`` specify the input and the output shape corresponding to one input function :math:`a`.
Similarly, in trunk net, the ``Key("x", 1)`` and  ``Key("trunk", 128)`` specify the input and the output shape corresponding to one coordinate point :math:`x`.
In the config, these models are defined under the ``arch`` config group.

.. literalinclude:: ../../../examples/anti_derivative/conf/config.yaml
   :language: yaml

The models are initialized in the Python script using the following:

.. literalinclude:: ../../../examples/anti_derivative/data_informed.py
   :language: python
   :start-after: [init-model] 
   :end-before: [init-model]

.. note::
   The DeepONet architecture in PhysicsNeMo Sym is extremely flexible allowing users to use different branch and trunk nets.
   For example a convolutional model can be used in the branch network while a fully-connected is used in the trunk.


Loading Data
~~~~~~~~~~~~

Then import the data from the ``.npy`` file.

.. literalinclude:: ../../../examples/anti_derivative/data_informed.py
   :language: python
   :start-after: [datasets] 
   :end-before: [datasets]

Adding Data Constraints
~~~~~~~~~~~~~~~~~~~~~~~

To add the data constraint, use ``DeepONetConstraint``.

.. literalinclude:: ../../../examples/anti_derivative/data_informed.py
   :language: python
   :start-after: [constraint] 
   :end-before: [constraint]

Adding Data Validator
~~~~~~~~~~~~~~~~~~~~~

You can set validators to verify the results.

.. literalinclude:: ../../../examples/anti_derivative/data_informed.py
   :language: python
   :start-after: [validator] 
   :end-before: [validator]


Training the Model
~~~~~~~~~~~~~~~~~~

Start the training by executing the python script.

.. code:: bash

   python data_informed.py

Results
~~~~~~~

The validation results (ground truth, DeepONet prediction, and difference, respectively) are shown as below
(:numref:`fig-data-deeponet-0`, :numref:`fig-data-deeponet-1`, :numref:`fig-data-deeponet-2`).

.. _fig-data-deeponet-0:

.. figure:: /images/user_guide/data_deeponet_0.png
   :alt: DeepONet validation result
   :width: 50.0%
   :align: center

   Data informed DeepONet validation result, sample 1

.. _fig-data-deeponet-1:

.. figure:: /images/user_guide/data_deeponet_1.png
   :alt: DeepONet validation result
   :width: 50.0%
   :align: center

   Data informed DeepONet validation result, sample 2

.. _fig-data-deeponet-2:

.. figure:: /images/user_guide/data_deeponet_2.png
   :alt: DeepONet validation result
   :width: 50.0%
   :align: center

   Data informed DeepONet validation result, sample 3

Problem 2: Anti-derivative (physics-informed)
---------------------------------------------

This section uses the physics-informed DeepONet to learn the anti-derivative operator without any observations
except for the given initial condition of the ODE system. Although there is no need for the training data, you will need some data for validation.

.. note::

    The python script for this problem can be found at ``/examples/anti_derivative/physics_informed.py``.

Case Setup
~~~~~~~~~~

Most of the setup for physics-informed DeepONet is the same as the data informed version. First you import the needed packages.

.. literalinclude:: ../../../examples/anti_derivative/physics_informed.py
   :language: python
   :lines: 1-15

Initializing the Model
~~~~~~~~~~~~~~~~~~~~~~

In the run function, setup the branch and trunk nets, respectively. This part is the same as the data informed version.

.. literalinclude:: ../../../examples/anti_derivative/physics_informed.py
   :language: python
   :start-after: [init-model] 
   :end-before: [init-model]

Loading Data
~~~~~~~~~~~~

Then, import the data as the data informed version.

.. literalinclude:: ../../../examples/anti_derivative/physics_informed.py
   :language: python
   :start-after: [datasets] 
   :end-before: [datasets]

Adding Constraints
~~~~~~~~~~~~~~~~~~

Now the main difference of physics informed version compared with data informed is highlighted.
First, impose the initial value constraint that :math:`a(0)=0`. The way to
achieve this is to set the input of the trunk net as all zero data. Then the output function will be
evaluated at only :math:`0`.

.. literalinclude:: ../../../examples/anti_derivative/physics_informed.py
   :language: python
   :start-after: [constraint1] 
   :end-before: [constraint1]

Next, impose the derivative constraint that :math:`\frac{d}{dx}u(x) = a(x)`.
Note here that ``u__x`` is the derivative of ``u`` w.r.t ``x``.

.. literalinclude:: ../../../examples/anti_derivative/physics_informed.py
   :language: python
   :start-after: [constraint2] 
   :end-before: [constraint2]

Adding Data Validator
~~~~~~~~~~~~~~~~~~~~~

Finally, add the validator. This is the same as data informed version.

.. literalinclude:: ../../../examples/anti_derivative/physics_informed.py
   :language: python
   :start-after: [validator] 
   :end-before: [validator]

Training the Model
~~~~~~~~~~~~~~~~~~

Start the training by executing the python script.

.. code:: bash

   python physics_informed.py

Results
~~~~~~~

The validation results (ground truth, DeepONet prediction, and difference, respectively) are shown as below
(:numref:`fig-physics-deeponet-0`, :numref:`fig-physics-deeponet-1`, :numref:`fig-physics-deeponet-2`).

.. _fig-physics-deeponet-0:

.. figure:: /images/user_guide/physics_deeponet_0.png
   :alt: DeepONet validation result
   :width: 50.0%
   :align: center

   Physics informed DeepONet validation result, sample 1

.. _fig-physics-deeponet-1:

.. figure:: /images/user_guide/physics_deeponet_1.png
   :alt: DeepONet validation result
   :width: 50.0%
   :align: center

   Physics informed DeepONet validation result, sample 2

.. _fig-physics-deeponet-2:

.. figure:: /images/user_guide/physics_deeponet_2.png
   :alt: DeepONet validation result
   :width: 50.0%
   :align: center

   Physics informed DeepONet validation result, sample 3


Problem 3: Darcy flow (data-informed)
-------------------------------------

Case Setup
~~~~~~~~~~

In this section, you will set up a data-informed DeepONet for learning the solution operator of a 2D Darcy flow in
PhysicsNeMo Sym. The problem setup and training data are the same as in Fourier Neural Operators. Please see the tutorial
:ref:`darcy_fno` for more details. It is worth emphasising that one can employ any built-in PhysicsNeMo Sym
network architectures  in a DeepONet model.

.. note::

   The python script for this problem can be found at ``examples/darcy/darcy_DeepO.py``.


Loading Data
~~~~~~~~~~~~

Loading both the training and validation datasets into memory follows a similar process as the :ref:`darcy_fno` example.

.. literalinclude:: ../../../examples/darcy/darcy_DeepO.py
   :language: python
   :start-after: [datasets] 
   :end-before: [datasets]

Initializing the Model
~~~~~~~~~~~~~~~~~~~~~~

Initializing DeepONet and domain is similar to the anti-derivative example but this time we will use a convolutional model.
Similar to the FNO example the model can be configured entirely through the config file.
A pix2pix convolutional model will be used as the branch net a while a fully-connected will be used as the trunk.
The DeepONet architecture will automatically handle the dimensionality difference.


.. literalinclude:: ../../../examples/darcy/conf/config_DeepO.yaml
   :language: yaml

The models are initialized inside the Python script using the following:

.. literalinclude:: ../../../examples/darcy/darcy_DeepO.py
   :language: python
   :start-after: [init-model] 
   :end-before: [init-model]

Adding Data Constraints and Validators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then you can add data constraints as before

.. literalinclude:: ../../../examples/darcy/darcy_DeepO.py
   :language: python
   :start-after: [constraint] 
   :end-before: [constraint]

.. literalinclude:: ../../../examples/darcy/darcy_DeepO.py
   :language: python
   :start-after: [validator] 
   :end-before: [validator]

Training the Model
~~~~~~~~~~~~~~~~~~

The training can now be simply started by executing the python script.

.. code:: bash

   python darcy_DeepO.py


Results and Post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The validation results (ground truth, DeepONet prediction, and difference, respectively) are shown as below.

.. figure:: /images/user_guide/deeponet_darcy_1.png
   :alt: DeepONet validation result 1
   :width: 80.0%
   :align: center

   DeepONet validation result, sample 1

.. figure:: /images/user_guide/deeponet_darcy_2.png
   :alt: DeepONet validation result 2
   :width: 80.0%
   :align: center

   DeepONet validation result, sample 2

.. figure:: /images/user_guide/deeponet_darcy_3.png
   :alt: DeepONet validation result 3
   :width: 80.0%
   :align: center

   DeepONet validation result, sample 3
