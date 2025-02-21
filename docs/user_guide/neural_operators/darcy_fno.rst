.. _darcy_fno:

Darcy Flow with Fourier Neural Operator
=======================================

Introduction
------------

In this tutorial, you will use PhysicsNeMo Sym to set up a data-driven model for a 2D Darcy flow using 
the Fourier Neural Operator (FNO) architecture inside of PhysicsNeMo Sym. In this tutorial, you will learn the following:

#. How to load grid data and set up data-driven constraints

#. How to create a grid validator node

#. How to use Fourier Neural Operator architecture in PhysicsNeMo Sym

.. note::

   This tutorial assumes that you are familiar with the basic functionality of PhysicsNeMo Sym and understand the FNO architecture.
   Please see the :ref:`Introductory Example` and :ref:`fno` sections for additional information.

.. warning::

   The Python package `gdown <https://github.com/wkentaro/gdown>`_ is required for this example if you do not already have the example data downloaded and converted.
   Install using ``pip install gdown``.

Problem Description
-------------------

The Darcy PDE is a second order, elliptic PDE with the following form:

.. math::
   :label: darcy

      -\nabla \cdot \left(k(\textbf{x})\nabla u(\textbf{x})\right) = f(\textbf{x}), \quad \textbf{x} \in D,

in which :math:`u(\textbf{x})` is the flow pressure, :math:`k(\textbf{x})` is the permeability field and :math:`f(\cdot)` is the
forcing function. The Darcy flow can parameterize a variety of systems including flow through porous media, elastic materials 
and heat conduction.
Here you will define the domain as a 2D unit square  :math:`D=\left\{x,y \in (0,1)\right\}` with the boundary condition 
:math:`u(\textbf{x})=0, \textbf{x}\in\partial D`. Recall that FNO requires a structured Euclidean input s.t. :math:`D = \textbf{x}_{i}` where 
:math:`i \in \mathbb{N}_{N\times N}`. Thus both the permeability and flow fields are discretized into a 2D matrix 
:math:`\textbf{K}, \textbf{U} \in \mathbb{R}^{N \times N}`.

This problem develops a surrogate model that learns the mapping between a permeability field and the pressure field,
:math:`\textbf{K} \rightarrow \textbf{U}`, for a distribution of permeability fields :math:`\textbf{K} \sim p(\textbf{K})`.
This is a key distinction of this problem from other examples, you are `not` learning just a single solution but rather a distribution.

.. _fig-fno_darcy:

.. figure:: /images/user_guide/fno_darcy.png
   :alt: FNO surrogate model for 2D Darcy Flow
   :width: 80.0%
   :align: center

   FNO surrogate model for 2D Darcy flow

Case Setup
----------

This example is a data-driven problem. This means that before starting any coding you need to make sure you have both the training and validation data.
The training and validation datasets for this example can be found on the `Fourier Neural Operator Github page <https://github.com/zongyi-li/fourier_neural_operator>`_.
Here is an automated script for downloading and converting this dataset.
This requires the package `gdown <https://github.com/wkentaro/gdown>`_ which can easily installed through ``pip install gdown``.

.. note::

   The python script for this problem can be found at ``examples/darcy/darcy_FNO_lazy.py``.

Configuration
~~~~~~~~~~~~~~
The configuration for this problem is fairly standard within PhysicsNeMo Sym. 
Note that we have two architectures in the config: one is the pointwise decoder for FNO and the other is the FNO model which will eventually ingest the decoder.
The most important parameter for FNO models is ``dimension`` which tells PhysicsNeMo Sym to load a 1D, 2D or 3D FNO architecture.
``nr_fno_layers`` are the number of Fourier convolution layers in the model.
The size of the latent features in FNO are determined based on the decoders input key ``z``, in this case the embedded feature space is 32.

.. literalinclude:: ../../../examples/darcy/conf/config_FNO.yaml
   :language: yaml

.. note::

   PhysicsNeMo Sym configs can allow users to define keys inside the YAML file. 
   In this instance, ``input_keys: [z,32]`` will create a single key of size 32 and ``input_keys: coeff`` creates a single input key of size 1.

Loading Data
~~~~~~~~~~~~~

For this data-driven problem the first step is to get the training data into PhysicsNeMo Sym.
Prior to loading data, set any normalization value that you want to apply to the data.
For this dataset, calculate the `scale` and `shift` parameters for both the input permeability field and output pressure.
Then, set this normalization inside PhysicsNeMo Sym by providing a shift/scale to each key, ``Key(name, scale=(shift, scale))``.

.. literalinclude:: ../../../examples/darcy/darcy_FNO_lazy.py
   :language: python
   :start-after: [keys] 
   :end-before: [keys]


There are two approaches for loading data: First you have eager loading where you immediately read the entire dataset onto memory at one time.
Alternatively, you can use lazy loading where the data is loaded on a per example basis as the model needs it for training.
The former eliminates potential overhead from reading data from disk during training, however this cannot scale to large datasets.
Lazy loading is used in this example for the training dataset to demonstrate this utility for larger problems.

.. literalinclude:: ../../../examples/darcy/darcy_FNO_lazy.py
   :language: python
   :start-after: [datasets] 
   :end-before: [datasets]

This data is in HDF5 format which is ideal for lazy loading using the ``HDF5GridDataset`` object.

.. note::

   The key difference when setting up eager versus lazy loading is the object passed in the variable dictionaries `invar_train` and `outvar_train`.
   In eager loading these dictionaries should be of the type ``Dict[str: np.array]``, where each variable is a numpy array of data.
   Lazy loading uses dictionaries of the type ``Dict[str: DataFile]``, consisting of ``DataFile`` objects which are classes that are used to map between example index and the datafile.

Initializing the Model
~~~~~~~~~~~~~~~~~~~~~~

FNO initialization allows users to define their own pointwise decoder model.
Thus we first initialize the small fully-connected decoder network, which we then provide to the FNO model as a parameter.

.. literalinclude:: ../../../examples/darcy/darcy_FNO_lazy.py
   :language: python
   :start-after: [init-model] 
   :end-before: [init-model]


Adding Data Constraints
~~~~~~~~~~~~~~~~~~~~~~~

For the physics-informed problems in PhysicsNeMo Sym, you typically need to define a geometry and 
constraints based on boundary conditions and governing equations.
Here the only constraint is a ``SupervisedGridConstraint`` which performs standard supervised
training on grid data.
This constraint supports the use of multiple workers, which are particularly important when using lazy loading.

.. literalinclude:: ../../../examples/darcy/darcy_FNO_lazy.py
   :language: python
   :start-after: [constraint] 
   :end-before: [constraint]

.. note::

   Grid data refers to data that can be defined in a tensor like an image. Inside PhysicsNeMo Sym this grid of data typically represents a spatial domain 
   and should follow the standard dimensionality of ``[batch, channel, xdim, ydim, zdim]`` where channel is the dimensionality of your state variables.
   Both Fourier and convolutional models use grid-based data to efficiently learn and predict entire domains in one forward pass, which contrasts to the
   pointwise predictions of standard PINN approaches.

Adding Data Validator
~~~~~~~~~~~~~~~~~~~~~~

The validation data is then added to the domain using ``GridValidator`` which should be used  
when dealing with structured data. 
Recall that unlike the training constraint, you will use eager loading for the validator.
Thus, a dictionary of numpy arrays are passed to the constraint.

.. literalinclude:: ../../../examples/darcy/darcy_FNO_lazy.py
   :language: python
   :start-after: [validator] 
   :end-before: [validator]


Training the Model
------------------

Start the training by executing the python script. 

.. code:: bash

   python darcy_FNO_lazy.py


Results and Post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The checkpoint directory is saved based on the results recording frequency
specified in the ``rec_results_freq`` parameter of its derivatives. See :ref:`hydra_results` for more information.
The network directory folder (in this case ``'outputs/darcy_fno/validators'``) contains several plots of different 
validation predictions. Several are shown below, and you can see that the model is able to accurately predict the 
pressure field for permeability fields it had not seen previously.

.. figure:: /images/user_guide/fno_darcy_pred1.png
   :alt: FNO Darcy Prediction 1
   :width: 80.0%
   :align: center

   FNO validation prediction 1. (Left to right) Input permeability, true pressure, predicted pressure, error.

.. figure:: /images/user_guide/fno_darcy_pred2.png
   :alt: FNO Darcy Prediction 2
   :width: 80.0%
   :align: center

   FNO validation prediction 2. (Left to right) Input permeability, true pressure, predicted pressure, error.

.. figure:: /images/user_guide/fno_darcy_pred3.png
   :alt: FNO Darcy Prediction 3
   :width: 80.0%
   :align: center

   FNO validation prediction 3. (Left to right) Input permeability, true pressure, predicted pressure, error.
