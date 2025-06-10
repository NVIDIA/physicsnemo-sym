.. _darcy_afno:


Darcy Flow with Adaptive Fourier Neural Operator
================================================

Introduction
------------
This tutorial demonstrates the use of transformer networks based on the Adaptive Fourier Neural Operators (AFNO) in PhysicsNeMo Sym. 
Note that in contrast with the :ref:`fno` which has a convolutional architecture, the AFNO leverages contemporary transformer architectures in the computer vision domain. 
Vision transformers have delivered tremendous success in computer vision. 
This is primarily due to effective self-attention mechanisms. 
However, self-attention scales quadratically with the number of pixels, which becomes infeasible for high-resolution inputs. 
To cope with this challenge, `Guibas et al.` [#guibas2021adaptive]_ proposed the Adaptive Fourier Neural Operator (AFNO) as an efficient attention mechanism in the Fourier domain. 
AFNO is based on a principled foundation of operator learning which allows us to frame attention as a continuous global convolution without any dependence on the input resolution. 
This principle was previously used to design FNO, which solves global convolution efficiently in the Fourier domain. 
To handle challenges in vision such as discontinuities in images and high resolution inputs, AFNO proposes principled architectural modifications to FNO which results in memory and computational efficiency. 
This includes imposing a block diagonal structure on the channel mixing weights, adaptively sharing weights across tokens, and sparse frequency modes via soft-thresholding and shrinkage.

This tutorial presents the use of the AFNO transformer for modeling a PDE system. 
While AFNO has been designed for scaling to extremely high resolution inputs that the FNO cannot handle as well (see [#pathak2022fourcastnet]_), here only a simple example using Darcy flow is presented.
This problem is intended as an illustrative starting point for data-driven training using AFNO in PhysicsNeMo Sym but should not be regarded as leveraging the full extent of AFNO's functionality.

This is an extension of the :ref:`darcy_fno` chapter. The unique topics you will learn in this tutorial include:

#. How to use the AFNO transformer architecture in PhysicsNeMo Sym

#. Differences between AFNO transformer and the Fourier Neural Operator

.. note::

   This tutorial assumes that you are familiar with the basic functionality of PhysicsNeMo Sym and understand the AFNO architecture.
   Please see the :ref:`Introductory Example` and :ref:`afno` sections for additional information.
   Additionally, this tutorial builds upon the :ref:`darcy_fno` which should be read prior to this one.

.. warning::

   The Python package `gdown <https://github.com/wkentaro/gdown>`_ is required for this example if you do not already have the example data downloaded and converted.
   Install using ``pip install gdown``.

Problem Description
-------------------

This problem develops a surrogate model that learns the mapping between a permeability and pressure field of a Darcy flow system.
The mapping learned, :math:`\textbf{K} \rightarrow \textbf{U}`, should be true for a distribution of permeability fields :math:`\textbf{K} \sim p(\textbf{K})`
not a single solution.
As discussed further in the :ref:`afno` theory, the AFNO is based on an image transformer backbone.
As with all transformer architectures, the AFNO tokenizes the input field. 
Each token is embedded from a patch of the image. 
The tokenized image is processed by the transformer layers followed by a linear decoder which generates the output image.

.. _fig-afno_darcy:

.. figure:: /images/user_guide/afno_darcy.png
   :alt: AFNO surrogate model for 2D Darcy Flow
   :width: 75.0%
   :align: center

   AFNO surrogate model for 2D Darcy flow

Case Setup
----------

Similar to the FNO chapter, the training and validation data for this example can be found on the `Fourier Neural Operator Github page <https://github.com/zongyi-li/fourier_neural_operator>`_.
The example also includes an automated script for downloading and converting this dataset.
This requires the package `gdown <https://github.com/wkentaro/gdown>`_ which can easily installed through ``pip install gdown``.

.. note::

   The python script for this problem can be found at ``examples/darcy/darcy_afno.py``.

Configuration
~~~~~~~~~~~~~~
The AFNO is based on the ViT transformer architecture [#dosovitskiy2020image]_ and requires tokenization of the inputs. 
Here each token is a patch of the image with a patch size defined in the configuration file through the parameter ``patch_size``
The ``embed_dim`` parameter defines the size of the latent embedded features used inside the model for each patch.

.. literalinclude:: ../../../examples/darcy/conf/config_AFNO.yaml
   :language: yaml


Loading Data
~~~~~~~~~~~~~

Loading both the training and validation datasets into memory follows a similar process as the :ref:`darcy_fno` example.

.. literalinclude:: ../../../examples/darcy/darcy_AFNO.py
   :language: python
   :lines: 33-49

The inputs for AFNO need to be perfectly divisible by the specified patch size (in this example ``patch_size=16``), which is not
the case for this dataset. Therefore, trim the input/output features such that they are an appropriate dimensionality ``241x241 -> 240x240``.

.. literalinclude:: ../../../examples/darcy/darcy_AFNO.py
   :language: python
   :lines: 51-67

Initializing the Model
~~~~~~~~~~~~~~~~~~~~~~

Initializing the model and domain follows the same steps as in other examples. For AFNO, calculate the size of the domain after loading
the dataset. The domain needs to be defined in the AFNO model, which is provided with the inclusion of the keyword argument ``img_shape``
in the ``instantiate_arch`` call.

.. literalinclude:: ../../../examples/darcy/darcy_AFNO.py
   :language: python
   :lines: 69-76

Adding Data Constraints and Validators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data-driven constraints and validators are then added to the domain.
For more information, see the :ref:`darcy_fno` chapter.

.. literalinclude:: ../../../examples/darcy/darcy_AFNO.py
   :language: python
   :lines: 78-96

Training the Model
------------------

The training can now be simply started by executing the python script. 

.. code:: bash

   python darcy_AFNO.py


Training with model parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With model parallelism, The AFNO model can be parallelized so multiple GPUs can split up and process even a single batch element in parallel. This
can be very beneficial when trying to strong scale and get to convergence faster or to reduce the memory footprint of the model per GPU in cases 
where the activations and model parameters are too big to fit on a single GPU.

The python script for the model parallel version of this example is at ``examples/darcy/darcy_AFNO_MP.py``. There are two main changes compared to the 
standard AFNO example. The first is changing the model architecture from ``afno`` to ``distributed_afno`` in the config file.

.. literalinclude:: ../../../examples/darcy/conf/config_AFNO_MP.yaml
   :language: yaml

The second change is to set the ``MODEL_PARALLEL_SIZE`` environment variable to initialize the model parallel communication backend.

.. literalinclude:: ../../../examples/darcy/darcy_AFNO_MP.py
   :language: python
   :lines: 34-35

This configures the distributed AFNO model to use 2 GPUs per model instance. The number of GPUs to use can be changed as long as the following conditions are satisfied:

1. The total number of GPUs in the job must be an exact multiple of ``MODEL_PARALLEL_SIZE``,
2. The ``num_blocks`` parameter in the config must be an exact multiple of ``MODEL_PARALLEL_SIZE`` and
3. The embedding dimension ``embed_dim`` must be an exact multiple of ``MODEL_PARALLEL_SIZE``.

Training the model parallel version of the example can then be launched using:

.. code:: bash

    mpirun -np 2 python darcy_AFNO_MP.py

.. warning::

   If running as root (typically inside a container), then OpenMPI requires adding a ``--allow-run-as-root`` option:
   ``mpirun --allow-run-as-root -np 2 python darcy_AFNO_MP.py``


Results and Post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The checkpoint directory is saved based on the results recording frequency
specified in the ``rec_results_freq`` parameter of its derivatives. See :ref:`hydra_results` for more information.
The network directory folder (in this case ``'outputs/darcy_afno/validators'``) contains several plots of different 
validation predictions.

.. figure:: /images/user_guide/afno_darcy_pred1.png
   :alt: AFNO Darcy Prediction 1
   :width: 80.0%
   :align: center

   AFNO validation prediction 1. (Left to right) Input permeability, true pressure, predicted pressure, error.

.. figure:: /images/user_guide/afno_darcy_pred2.png
   :alt: AFNO Darcy Prediction 2
   :width: 80.0%
   :align: center

   AFNO validation prediction 2. (Left to right) Input permeability, true pressure, predicted pressure, error.

.. figure:: /images/user_guide/afno_darcy_pred3.png
   :alt: AFNO Darcy Prediction 3
   :width: 80.0%
   :align: center

   AFNO validation prediction 3. (Left to right) Input permeability, true pressure, predicted pressure, error.

It is important to recognize that AFNO's strengths lie in its ability to scale to a much larger model size and datasets than what is used in this chapter [#guibas2021adaptive]_ [#pathak2022fourcastnet]_. 
While not illustrated here, this example demonstrates the fundamental implementation of data-driven training using the AFNO architecture in PhysicsNeMo Sym for users to extend to larger problems.


.. rubric:: References

.. [#guibas2021adaptive] Guibas, John, et al. "Adaptive fourier neural operators: Efficient token mixers for transformers" International Conference on Learning Representations, 2022.
.. [#pathak2022fourcastnet] Pathak, Jaideep, et al. "FourCastNet : A global data-driven high-resolution weather model using adaptive Fourier neural operators" arXiv preprint arXiv:2202.11214 (2022).
.. [#dosovitskiy2020image] Dosovitskiy, Alexey et al. "An image is worth 16x16 words: Transformers for image recognition at scale" arXiv preprint arXiv:2010.11929 (2020).
