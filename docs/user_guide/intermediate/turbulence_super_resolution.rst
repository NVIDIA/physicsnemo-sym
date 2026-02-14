.. _turbulence_super_res:

Turbulence Super Resolution
===========================

Introduction
------------

This example uses PhysicsNeMo Sym to train a super-resolution surrogate model for predicting high-fidelity homogeneous isotropic turbulence fields from filtered low-resolution observations provided by the `Johns Hopkins Turbulence Database <https://turbulence.idies.jhu.edu/home>`_.
This model will combine standard data-driven learning as well as how to define custom data-driven loss functions that are uniquely catered to a specific problem.
In this example you will learn the following:

#. How to use data-driven convolutional neural network models in PhysicsNeMo Sym

#. How to define custom data-driven loss and constraint

#. How to define custom data-driven validator

#. PhysicsNeMo Sym features for structured/grid data

#. Adding custom parameters to the problem configuration file

.. note:: 

   This tutorial assumes that you have completed the :ref:`Introductory Example` tutorial on Lid Driven Cavity flow and have familiarized yourself with the basics of PhysicsNeMo Sym. 
   This also assumes that you have a basic understanding of the convolution models in PhysicsNeMo Sym :ref:`pix2pix` and :ref:`super_res`.

.. warning::

   The Python package `giverny <https://github.com/sciserver/giverny>`_ is required for this example to download and process the training and validation datasets.
   Install using ``pip install giverny``.

Problem Description
-------------------

The objective of this problem is to learn the mapping between a low-resolution filtered 3D flow field to a high-fidelity solution.
The flow field will be samples of a forced `isotropic turbulence direct numerical simulation <https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence/isotropic>`_ originally simulated with a resolution of :math:`1024^{3}`.
This simulation solves the forced Navier-Stokes equations:

.. math::

    \frac{\partial \textbf{u}}{\partial t} + \textbf{u} \cdot \nabla \textbf{u} = -\nabla p /\rho + \nu \nabla^{2}\textbf{u} + \textbf{f}.

The forcing term :math:`\textbf{f}` is used to inject energy into the simulation to maintain a constant total energy. 
This dataset contains 5028 time steps spanning from 0 to 10.05 seconds which are sampled every 10 time steps from the original pseudo-spectral simulation.

.. figure:: ../../images/user_guide/super_res_dataset_sample.png
   :alt: Isotropic turbulence velocity field snap shot
   :width: 70.0%
   :align: center

   Snap shot of :math:`128^{3}` isotropic turbulence velocity fields


The objective is to build a surrogate model to learn the mapping between a low-resolution velocity field :math:`\textbf{U}_{l} = \left\{u_{l}, v_{l}, w_{l}\right\}` to the true high-resolution field :math:`\textbf{U}_{h} = \left\{u_{h}, v_{h}, w_{h}\right\}` for any low-resolution sample in this isotropic turbulence dataset :math:`\textbf{U}_{l} \sim p(\textbf{U}_{l})`.
Due to the size of the full simulation domain, this tutorial focuses on predicting smaller volumes such that the surrogate learns with a low resolution dimensionality of :math:`32^{3}` to a high-resolution dimensionality of :math:`128^{3}`.
Use the :ref:`super_res` in this tutorial, but :ref:`pix2pix` is also integrated into this problem and can be used instead if desired.

.. figure:: ../../images/user_guide/super_res_surrogate.png
   :alt: Isotropic turbulence velocity field snap shot
   :width: 70.0%
   :align: center

   Super resolution network for predicting high-resolution turbulent flow from low-resolution input

Writing a Custom Data-Driven Constraint
---------------------------------------

This example demonstrates how to write your own data-driven constraint.
PhysicsNeMo Sym ships with a standard supervised learning constraint for structured data called ``SupervisedGridConstraint`` used in the :ref:`darcy_fno` example.
However, if you want to have a problem specific loss you can extend the base ``GridConstraint``.
Here, you will set up a constraint that can pose a loss between various measures for the fluid flow including velocity, continuity, vorticity, enstrophy and strain rate. 

.. note::

   The python script for this problem can be found at ``examples/super_resolution/super_resolution.py``.

.. literalinclude:: ../../../examples/super_resolution/super_resolution.py
   :language: python
   :lines: 42-75

An important part to note is that you can control which losses you want to contribute using a ``loss_weighting`` dictionary which will be provided from the config file.
Each one of these measures are calculated in ``examples/super_resolution/ops.py``, which can be referenced for more information.
However, as a general concept, finite difference methods are used to calculate gradients of the flow field and the subsequent measures.

.. literalinclude:: ../../../examples/super_resolution/super_resolution.py
   :language: python
   :lines: 77-134

Override the loss calculation with the custom method which calculates the relative MSE between predicted and target velocity fields using flow measures defined in the weight dictionary, ``self.loss_weighting``.

.. literalinclude:: ../../../examples/super_resolution/super_resolution.py
   :language: python
   :lines: 197-211

The resulting complete loss for this problem is the following:

.. math::

    \mathcal{L} = RMSE(\hat{U}_{h}, U_{h}) + \lambda_{dU}RMSE(\hat{dU}_{h}, dU_{h}) + \lambda_{cont}RMSE(\nabla\cdot\hat{U}_{h}, \nabla\cdot U_{h}) + \lambda_{\omega}RMSE(\hat{\omega}_{h}, \omega_{h}) \\
    + \lambda_{strain}RMSE(|\hat{D}|_{h}, |D|_{h}) + \lambda_{enst}RMSE(\hat{\epsilon}_{h}, \epsilon_{h}),

in which :math:`\hat{U}_{h}` is the prediction from the neural network and :math:`U_{h}` is the target.
:math:`dU` is the velocity tensor, :math:`\omega` is the vorticity, :math:`|D|` is the magnitude of the strain rate and :math:`\epsilon` is the flow's enstrophy.
All of these can be turned on and off in the configuration file under the ``custom.loss_weights`` config group.

Writing a Custom Data-Driven Validator
---------------------------------------

Similarly, because the input and output are of different dimensionality, the built in ``GridValidator`` in PhysicsNeMo Sym will not work since it expects all tensors to be the same size.
You can easily extend this to write out the high-resolution outputs and low-resolution outputs into separate VTK uniform grid files.

.. literalinclude:: ../../../examples/super_resolution/super_resolution.py
   :language: python
   :lines: 214-297

Here the ``grid_to_vtk`` function in PhysicsNeMo Sym is used, which writes tensor data to VTK Image Datasets (Uniform grids), which can then be viewed in Paraview. 
When your data is structured ``grid_to_vtk`` is preferred to ``var_to_polyvtk`` due to the lower memory footprint of a VTK Image Dataset vs VTK Poly Dataset.

Case Setup
----------

Before proceeding, it is important to recognize that this problem needs to download the dataset upon its first run.
To download the dataset from `Johns Hopkins Turbulence Database <https://turbulence.idies.jhu.edu/home>`_ you will need to request an access token.
Information regarding this process can be found on the `database website <https://turbulence.idies.jhu.edu/database>`_.
Once aquired, please overwrite the default token in the config in the specified location.
Utilities used to download the data can be located in ``examples/super_resolution/jhtdb_utils.py``, but will not be discussed in this tutorial.

.. warning::

    By registering for an access token and using the Johns Hopkins Turbulence Database, you are agreeing to the terms and conditions of the dataset itself. This example will not work without an access token.

.. warning::

    The default training dataset size is 512 sampled and the validation dataset size is 16 samples. The download can take several hours depending on your internet connection. The total memory footprint of the data is around `13.5Gb`. A smaller dataset can be set in the config file.

Configuration
~~~~~~~~~~~~~~

The config file for this example is as follows. Note that both the super-resolution and pix2pix encoder-decoder architecture configuration are included to test.

.. literalinclude:: ../../../examples/super_resolution/conf/config.yaml
   :language: yaml

The ``custom`` config group can be used to store case specific parameters that will not be used inside of PhysicsNeMo Sym.
Here you can use this group to define parameters related to the dataset size, the domain size of the fluid volumes and the database  access token which you should replace with your own!

.. note::
    To just test the model with a toy dataset without a database access token you are recommended to use the below settings:

   .. code-block:: yaml

        jhtdb:
            n_train: 4
            n_valid: 1
            domain_size: 16
            access_token: "edu.jhu.pha.turbulence.testing-201311" 


Loading Data
~~~~~~~~~~~~~~

To load the dataset into memory, you will use the following utilities:

.. literalinclude:: ../../../examples/super_resolution/super_resolution.py
   :language: python
   :lines: 302-321

This will download and cache the dataset locally, so you will not need to download it with every run.

Initializing the Model
~~~~~~~~~~~~~~~~~~~~~~

Here you initialize the model following the standard PhysicsNeMo Sym process.
Note that the input and output keys have a ``size=3``, which tells PhysicsNeMo Sym that these variables have 3 dimensions (velocity components).

.. literalinclude:: ../../../examples/super_resolution/super_resolution.py
   :language: python
   :lines: 323-328

Adding Data Constraints
~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/super_resolution/super_resolution.py
   :language: python
   :lines: 330-343

Adding Data Validator
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/super_resolution/super_resolution.py
   :language: python
   :lines: 345-353


Training the Model
------------------

NVIDIA recommends that your first run be on a single GPU to download the dataset.
Only root process will download the data from the online database while the others will be idle.

.. code:: bash

   python super_resolution.py

However, parallel training is suggested for this problem once the dataset is downloaded. This example was trained on 4 V100 GPUs which can be run via Open MPI using the following command:

.. code:: bash

   mpirun -np 4 python super_resolution.py


Results and Post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since this example illustrated how to set up a custom data-driven loss that is controllable through the config, you can compare the impact of several different loss components on the model's performance.
The TensorBoard plot is shown below with the validation dataset loss being the bottom most graph.
Given the number of potential loss components, only a handful are compared here:

- ``U=1.0``: :math:`\mathcal{L} = RMSE(\hat{U}_{h}, U_{h})`
- ``U=1.0, omega=0.1``:  :math:`\mathcal{L} = RMSE(\hat{U}_{h}, U_{h}) + 0.1RMSE(\hat{\omega}_{h}, \omega_{h})`
- ``U=1.0, omega=0.1, dU=0.1``: :math:`\mathcal{L} = RMSE(\hat{U}_{h}, U_{h}) + 0.1RMSE(\hat{\omega}_{h}, \omega_{h}) + 0.1RMSE(\hat{dU}_{h}, dU_{h})`
- ``U=1.0, omega=0.1, dU=0.1, contin=0.1``: :math:`\mathcal{L} = RMSE(\hat{U}_{h}, U_{h}) + 0.1RMSE(\hat{\omega}_{h}, \omega_{h}) + 0.1RMSE(\hat{dU}_{h}, dU_{h}) + 0.1RMSE(\nabla\cdot\hat{U}_{h}, \nabla\cdot U_{h})`

The validation error is the L2 relative error between the predicted and true high-resolution velocity fields.
You can see that the inclusion of vorticity in the loss equation increases the model's accuracy, however the inclusion of other terms does not.
Loss combinations of additional fluid measures has proven successful in past works [#geneva2020multi]_ [#subramaniam2020turbulence]_. 
However, additional losses can potentially make the optimization more difficult for the model and adversely impact accuracy.

.. figure:: ../../images/user_guide/super_res_tensorboard.png
   :alt: Turbulence super-resolution tensorboard
   :width: 75.0%
   :align: center

   Tensorboard plot comparing different loss functions for turbulence super-resolution


The output VTK files can be found in the ``'outputs/super_resolution/validators'`` folder which you can then view in Paraview.
The volumetric plots of the velocity magnitude fields are shown below where you can see the model dramatically improves the low-resolution velocity field.

.. figure:: ../../images/user_guide/super_res_validation_0.png
    :alt: Validation volumetric plot
    :width: 80.0%
    :align: center
    
    Velocity magnitude for a validation case using the super resolution model for predicting turbulence

.. figure:: ../../images/user_guide/super_res_validation_1.png
    :alt: Validation volumetric plot
    :width: 80.0%
    :align: center
    
    Velocity magnitude for a validation case using the super resolution model for predicting turbulence


.. rubric:: References

.. [#geneva2020multi] Geneva, Nicholas and Zabaras, Nicholas. "Multi-fidelity generative deep learning turbulent flows" Foundations of Data Science (2020).

.. [#subramaniam2020turbulence] Subramaniam, Akshay et al. "Turbulence enrichment using physics-informed generative adversarial networks" arXiv preprint arXiv:2003.01907 (2020).
