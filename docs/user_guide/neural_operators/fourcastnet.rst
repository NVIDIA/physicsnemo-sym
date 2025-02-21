.. _fourcastnet_example:

FourCastNet
===========

Introduction
------------

This example reproduces FourCastNet [#pathak2022fourcastnet]_ using PhysicsNeMo Sym.
FourCastNet, short for **Four**\ier Fore\ **\Cast**\ing Neural **Net**\work, is a global data-driven weather forecasting model that provides accurate short to medium range global predictions at 0.25◦ resolution.
FourCastNet generates a week long forecast in less than 2 seconds, orders of magnitude faster than the ECMWF Integrated Forecasting System (IFS), a state-of-the-art Numerical Weather Prediction (NWP) model, with comparable or better accuracy.
It is trained on a small subset of the ERA5 reanalysis dataset [#hersbach2020era5]_ from the ECMWF, which consists of hourly estimates of several atmospheric variables at a latitude and longitude resolution of :math:`0.25^{\circ}`.
Given an initial condition from the ERA5 dataset as input, FourCastNet recursively applies an Adaptive Fourier Neural Operator (AFNO) network to predict their dynamics at later time steps.
In the current iteration, FourCastNet forecasts 20 atmospheric variables. These variables, listed in the table below are sampled from the ERA5 dataset at a temporal resolution of 6 hours.

.. list-table:: FourCastNet modeled variables
   :widths: 25 25
   :header-rows: 1

   * - Vertical Level
     - Variable
   * - Surface
     - U10, V10, T2M, SP, MSLP
   * - 1000 hPa
     - U, V, Z
   * - 850 hPa
     - T, U, V, Z, RH
   * - 500 hPa
     - T, U, V, Z, RH
   * - 50 hPa
     - Z
   * - Integrated
     - TCWV


In this tutorial, we will show you how to define, train and evaluate FourCastNet in PhysicsNeMo Sym.
The topics covered here are:

#. How to load the ERA5 dataset into PhysicsNeMo Sym

#. How to define the FourCastNet architecture in PhysicsNeMo Sym

#. How to train FourCastNet

#. How to generate weather forecasts and quantitatively assess performance

.. note::

    AFNOs are covered in detail in :ref:`afno` and :ref:`darcy_afno` and we recommend reading these chapters first.
    Please also refer to the `ArXiv pre-print <https://arxiv.org/abs/2202.11214>`_ for more details on the original implementation [#pathak2022fourcastnet]_.


.. warning::

    The ERA5 dataset is very large (5 TB+) and we do not provide it as part of this tutorial.  ERA5 data [#hersbach2020era5]_ was downloaded from the Copernicus Climate Change Service (C3S) Climate Data Store [#hersbach2018pl]_, [#hersbach2018sl]_.


Problem Description
-------------------

The goal of FourCastNet is to forecast modeled variables on a short time scale of upto 10 days. FourCastNet is initialized using an initial condition from the ERA5 reanalysis dataset.
The figure below shows an overview of how FourCastNet works:

.. _fig-fourcastnet_overview:

.. figure:: /images/user_guide/fourcastnet_overview.png
   :alt: Overview of FourCastNet
   :width: 75.0%
   :align: center

   FourCastNet overview. Figure reproduced with permission from [#pathak2022fourcastnet]_.

To make a weather prediction, 20 different ERA5 variables each defined on a regular  latitude/longitude grid of dimension :math:`720\times 1440` spanning the entire globe at some starting time step :math:`t` are given as inputs to the model (bottom left of figure).
Then, an AFNO architecture (middle left) is used to predict these variables at a later time step :math:`t+\Delta t` (the original paper uses a fixed time delta :math:`\Delta t` of 6 hours).
During inference, these predictions can be recursively fed back into the AFNO, which allows the model to predict multiple time steps ahead (bottom right).
Furthermore, we can train the network by either using a single step prediction, or by unrolling the network over :math:`n` steps and using a loss function which matches each predicted time step to training data (top right).
Typically, single step prediction is used for initial training, and then two step prediction is used for fine tuning, as it is more expensive.

.. note::

    The original paper employs an additional precipitation model (middle right), although we only implement the AFNO "backbone" model here.


Case Setup
----------

To train FourCastNet, we use the ERA5 data over the years 1979 to 2015 (both included). When testing its performance, we use out of sample ERA5 data from 2018.
Please see the original paper for a description of the 20 variables used and the preprocessing applied to the ERA5 dataset; they are specifically chosen to model important processes that influence low-level winds and precipitation.
The data is stored using the following directory structure:

.. code::

    era5
    ├── train
    │   ├── 1979.h5
    │   ├── ...
    │   ├── 2015.h5
    ├── test
    │   ├── 2018.h5
    └── stats
        ├── global_means.npy
        └── global_stds.py

where each HDF5 file contains all of the variables for each year, over 1460 time steps with 6 hour time deltas (i.e. it has a shape (1460, 20, 720, 1440)).


.. note::

   All of the python scripts for this example are in ``examples/fourcastnet/``.


Configuration
~~~~~~~~~~~~~~

The configuration file for FourCastNet is similar to the configuration file used in the :ref:`darcy_afno` example and is shown below.

.. literalinclude:: ../../../examples/fourcastnet/conf/config_FCN.yaml
   :language: yaml

In addition, we have added the ``custom.tstep`` and ``custom.n_tsteps`` parameters which define the time delta between the AFNO's input and output time steps (in multiples of 6 hours, typically set to 1) and the number of time steps FourCastNet is unrolled over during training.



Loading Data
~~~~~~~~~~~~~

PhysicsNeMo Sym FourCastNet currently has two options for loading the data:

#. DALI-based dataloader which uses `NVIDIA Data Loading Library (DALI) <https://developer.nvidia.com/dali>`_ for accelerated data loading and processing.

#. Standard PyTorch dataloader.

DALI dataloader is the default option, but can be changed by setting ``custom.train_dataset.kind`` option to ``pytorch``.

Both dataloaders use a shared implementation which supports ERA5 data format and is defined in ``fourcastnet/src/dataset.py``:

.. literalinclude:: ../../../examples/fourcastnet/src/dataset.py
   :language: python
   :lines: 26-142

Given an example index, the dataset's ``__getitem__`` method returns a single PhysicsNeMo Sym input variable, ``x_t0``,
which is a tensor of shape (20, 720, 1440) which contains the 20 ERA5 variables at a starting time step,
and multiple output variables with the same shape, ``x_t1``, ``x_t2``, ..., one for each predicted time step FourCastNet is unrolled over:

.. literalinclude:: ../../../examples/fourcastnet/src/dataset.py
   :language: python
   :lines: 145-194

Inside the training script, ``fourcastnet/era5_FCN.py``, the ERA5 datasets are initialized using the following:

.. literalinclude:: ../../../examples/fourcastnet/fcn_era5.py
   :language: python
   :lines: 50-72

FourCastNet Model
~~~~~~~~~~~~~~~~~

Next, we need to define FourCastNet as a custom PhysicsNeMo Sym architecture.
This model is found inside ``fourcastnet/src/fourcastnet.py`` which is a wrapper class of PhysicsNeMo Sym' AFNO model.
FourCastNet has two training phases: the first is single step prediction and the second is two step predictions.
This small wrapper allows AFNO to be executed for any ``n_tsteps`` of time steps using autoregressive forward passes.

.. literalinclude:: ../../../examples/fourcastnet/src/fourcastnet.py
   :language: python
   :lines: 26-

The FourCastNet model is initialized in the training script, ``fourcastnet/era5_FCN.py``:

.. literalinclude:: ../../../examples/fourcastnet/fcn_era5.py
   :language: python
   :lines: 81-91

Adding Constraints
~~~~~~~~~~~~~~~~~~

With the custom dataset for loading the ERA5 data and the FourCastNet model created, the next step is setting up the PhysicsNeMo Sym training domain.
The main training script is ``fourcastnet/era5_FCN.py`` and constraints the standard steps needed for training a model in PhysicsNeMo Sym.
A standard data-driven grid constraint is created:

.. literalinclude:: ../../../examples/fourcastnet/fcn_era5.py
   :language: python
   :lines: 93-104

A validator is also added to the training script:

.. literalinclude:: ../../../examples/fourcastnet/fcn_era5.py
   :language: python
   :lines: 106-114


Training the Model
------------------

The training can now be simply started by executing the python script.

.. code:: bash

   python fcn_era5.py


Results and Post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~
With the trained model ``fourcastnet/inferencer.py`` is used  to calculate the latitude weighted Root Mean Squared Error (RMSE) and the latitude weighted Anomaly Correlation Coefficient (ACC) values.
The inferencer script uses runs the trained model on multiple initial conditions provided in the test dataset.
Below the ACC and RMSE values of the model trained in PhysicsNeMo Sym is compared to the results of the original work with excellent comparison to the original work [#pathak2022fourcastnet]_.
Additionally, a 24 hour forecast is also illustrated comparing the integrated vertical column of atmospheric water vapor predicted by PhysicsNeMo Sym and the target ERA5 dataset.

.. note::

   See the original ArXiv paper or ``src/metrics.py`` for more details on how these metrics are calculated.
   Multiple dataset statistics are needed to properly calculate the metrics of interest.

.. figure:: /images/user_guide/fourcastnet_acc.png
   :alt: PhysicsNeMo Sym FourCastNet ACC
   :width: 90.0%
   :align: center

   Comparison of the anomaly correlation coefficient (ACC) of the predicted 10 meter `u` component wind
   speed (`u10`) and geopotential height (`z500`) using the original FourCastNet model (Original) and the version trained in PhysicsNeMo Sym.

.. figure:: /images/user_guide/fourcastnet_rmse.png
   :alt: PhysicsNeMo Sym FourCastNet RMSE
   :width: 90.0%
   :align: center

   Comparison of the predictive root mean square error (RMSE) of each variable between the original FourCastNet model (Original) and the version trained in PhysicsNeMo Sym.

.. figure:: /images/user_guide/fourcastnet_tcwv.png
   :alt: PhysicsNeMo Sym FourCastNet TCWV
   :width: 50.0%
   :align: center

   24 hour prediction of the integrated vertical column of atmospheric water vapor predicted by PhysicsNeMo Sym compared to the ground truth ERA5 dataset from ECMWF.


.. rubric:: References
.. [#pathak2022fourcastnet] Pathak, Jaideep, et al. "FourCastNet : A global data-driven high-resolution weather model using adaptive Fourier neural operators" arXiv preprint arXiv:2202.11214 (2022).
.. [#hersbach2020era5] Hersbach, Hans, et al. "The ERA5 global reanalysis" Quarterly Journal of the Royal Meteorological Society (2020).
.. [#hersbach2018pl] Hersbach, Hans, et al. "ERA5 hourly data on pressure levels from 1959 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS)." , 10.24381/cds.bd0915c6 (2018)
.. [#hersbach2018sl] Hersbach, Hans et al., "ERA5 hourly data on single levels from 1959 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS)." , 10.24381/cds.adbb2d47 (2018)
.. [#guibas2021adaptive] Guibas, John, et al. "Adaptive fourier neural operators: Efficient token mixers for transformers" International Conference on Learning Representations, 2022.
