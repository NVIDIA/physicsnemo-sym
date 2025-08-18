.. _darcy_pino:

Darcy Flow with Physics-Informed Fourier Neural Operator
========================================================

Introduction
------------

This tutorial solves the 2D Darcy flow problem using Physics-Informed Neural Operators (PINO) [#li2021physics]_. 
In this tutorial, you will learn:

#. Differences between PINO and Fourier Neural Operators (FNO).

#. How to set up and train PINO in PhysicsNeMo Sym.

#. How to define a custom PDE constraint for grid data.

.. note::

   This tutorial assumes that you are familiar with the basic functionality of PhysicsNeMo Sym and understand the PINO architecture.
   Please see the :ref:`Introductory Example` and :ref:`pino` sections for additional information.
   Additionally, this tutorial builds upon the :ref:`darcy_fno` which should be read prior to this one.
   
.. warning::

   The Python package `gdown <https://github.com/wkentaro/gdown>`_ is required for this example if you do not already have the example data downloaded and converted.
   Install using ``pip install gdown``.

Problem Description
-------------------

This problem illustrates developing a surrogate model that learns the mapping between a permeability and pressure field of
a Darcy flow system. The mapping learned, :math:`\textbf{K} \rightarrow \textbf{U}`, 
should be true for a distribution of permeability fields :math:`\textbf{K} \sim p(\textbf{K})` and not just a single solution.

The key difference between PINO and FNO is that PINO adds a physics-informed term to the loss function of FNO. 
As discussed further in the :ref:`pino` theory, the PINO loss function is described by:

.. math:: \mathcal{L} = \mathcal{L}_{data} + \mathcal{L}_{pde},

where

.. math:: \mathcal{L}_{data} = \lVert u - \mathcal{G}_\theta(a)  \rVert^2 ,

where :math:`\mathcal{G}_\theta(a)` is a FNO model with learnable parameters :math:`\theta` and input field :math:`a`, and 
:math:`\mathcal{L}_{pde}` is an appropriate PDE loss. For the 2D Darcy problem (see :ref:`darcy_fno`) this is given by

.. math:: \mathcal{L}_{pde} = \lVert -\nabla \cdot \left(k(\textbf{x})\nabla \mathcal{G}_\theta(a)(\textbf{x})\right) - f(\textbf{x}) \rVert^2 ,

where :math:`k(\textbf{x})` is the permeability field, :math:`f(\textbf{x})` is the forcing function equal to 1 in this case, and :math:`a=k` in this case.

Note that the PDE loss involves computing various partial derivatives of the FNO ansatz, :math:`\mathcal{G}_\theta(a)`. 
In general this is nontrivial; in PhysicsNeMo Sym, three different methods for computing these are provided. These are based on the original PINO paper:

#. Numerical differentiation computed via finite difference Method (FDM)
#. Numerical differentiation computed via spectral derivative
#. Numerical differentiation based on the "exact" Fourier / automatic differentiation approach [#ft1]_

Note that the last approach only works for a fixed decoder model.
Upon enabling "exact" gradient calculations, the decoder network will switch to a 2 layer fully-connected model with Tanh activations.
This is because this approach requires an expensive Hessian calculation.
The Hessian calculation is explicitly defined for this two layer model, thus avoiding automatic differentiation which would be otherwise be extremely expensive.


Case setup
----------

The setup for this problem is largely the same as the FNO example (:ref:`darcy_fno`), except that the PDE loss is defined and the FNO model is constrained using it. 
This process is described in detail in :ref:`darcy_pino_pde` below.

Similar to the FNO chapter, the training and validation data for this example can be found on the `Fourier Neural Operator Github page <https://github.com/zongyi-li/fourier_neural_operator>`_.
However, an automated script for downloading and converting this dataset has been included.
This requires the package `gdown <https://github.com/wkentaro/gdown>`_ which can easily installed through ``pip install gdown``.

.. note::

   The python script for this problem can be found at ``examples/darcy/darcy_pino.py``.
   
Configuration
~~~~~~~~~~~~~~

The configuration for this problem is similar to the FNO example, 
but importantly there is an extra parameter ``custom.gradient_method`` where the method for computing the gradients in the PDE loss is selected.
This can be one of ``fdm``, ``fourier``, ``exact`` corresponding to the three options above.
The balance between the data and PDE terms in the loss function can also be controlled using the ``loss.weights`` parameter group.

.. literalinclude:: ../../../examples/darcy/conf/config_PINO.yaml
   :language: yaml


.. _darcy_pino_pde:

Defining PDE Loss
~~~~~~~~~~~~~~~~~

For this example, a custom PDE residual calculation is defined using the various approaches proposed above.
Defining a custom PDE residual using sympy and automatic differentiation is discussed in :ref:`transient`, but in this problem you will not be relying on standard automatic differentiation for calculating the derivatives.
Rather, you will explicitly define how the residual is calculated using a custom ``torch.nn.Module`` called ``Darcy``. 
The purpose of this module is to compute and return the Darcy PDE residual given the input and output tensors of the FNO model, 
which is done via its ``.forward(...)`` method:

.. code:: python

    from physicsnemo.models.layers import fourier_derivatives # helper function for computing spectral derivatives
    from ops import dx, ddx # helper function for computing finite difference derivatives
    
.. literalinclude:: ../../../examples/darcy/darcy_PINO.py
   :language: python
   :start-after: [pde-loss] 
   :end-before: [pde-loss]
   
The gradients of the FNO solution are computed according to the gradient method selected above.
The FNO model automatically outputs first and second order gradients when the ``exact`` method is used, and so no extra computation of these is necessary.
Furthermore, note that the gradients of the permeability field are already included as tensors in the FNO input training data (with keys ``Kcoeff_x`` and ``Kcoeff_y``)
and so these do not need to be computed.

Next, incorporate this module into PhysicsNeMo Sym by wrapping it into a PhysicsNeMo Sym ``Node``. 
This ensures the module is incorporated into PhysicsNeMo Sym' computational graph and can be used to optimise the FNO.

.. code:: python

    from physicsnemo.sym.node import Node
    
.. literalinclude:: ../../../examples/darcy/darcy_PINO.py
   :language: python
   :start-after: [init-node] 
   :end-before: [init-node]
   
Finally, define the PDE loss term by adding a constraint to the ``darcy`` output variable (see :ref:`darcy_pino_constraints` below).

Loading Data
~~~~~~~~~~~~~

Loading both the training and validation datasets follows a similar process as the FNO example:

.. literalinclude:: ../../../examples/darcy/darcy_PINO.py
   :language: python
   :start-after: [datasets] 
   :end-before: [datasets]

Initializing the Model
~~~~~~~~~~~~~~~~~~~~~~

Initializing the model also follows a similar process as the FNO example:

.. literalinclude:: ../../../examples/darcy/darcy_PINO.py
   :language: python
   :start-after: [init-model] 
   :end-before: [init-model]

However, in the case where the ``exact`` gradient method is used, 
you need to additionally instruct the model to output the appropriate gradients by specifying these gradients in its output keys.

.. _darcy_pino_constraints:

Adding Constraints
~~~~~~~~~~~~~~~~~~

Finally, add constraints to your model in a similar fashion to the FNO example.
The same ``SupervisedGridConstraint`` can be used; 
to include the PDE loss term you need to define additional target values for the ``darcy`` output variable defined above (zeros, to minimise the PDE residual)
and add them to the ``outvar_train`` dictionary:

.. literalinclude:: ../../../examples/darcy/darcy_PINO.py
   :language: python
   :start-after: [constraint] 
   :end-before: [constraint]

The same data validator as the FNO example is used.

Training the Model
------------------

The training can now be simply started by executing the python script. 

.. code:: bash

   python darcy_PINO.py


Results and Post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The checkpoint directory is saved based on the results recording frequency
specified in the ``rec_results_freq`` parameter of its derivatives. See :ref:`hydra_results` for more information.
The network directory folder (in this case ``'outputs/darcy_pino/validators'``) contains several plots of different 
validation predictions.

.. figure:: ../../images/user_guide/pino_darcy_pred.png
   :alt: PINO Darcy Prediction
   :width: 80.0%
   :align: center

   PINO validation prediction. (Left to right) Input permeability and its spatial derivatives, true pressure, predicted pressure, error.

Comparison to FNO
~~~~~~~~~~~~~~~~~

The TensorBoard plot below compares the validation loss of PINO (all three gradient methods) and FNO. 
You can see that with large amounts of training data (1000 training examples), both FNO and PINO perform similarly.

.. figure:: ../../images/user_guide/pino_darcy_tensorboard1.png
   :alt: FNO vs PINO Darcy Tensorboard
   :width: 70.0%
   :align: center

   Comparison between PINO and FNO accuracy for surrogate modeling Darcy flow.
   
A benefit of PINO is that the PDE loss regularizes the model, meaning that it can be more effective in "small data" regimes.
The plot below shows the validation loss when both models are trained with only 100 training examples:

.. figure:: ../../images/user_guide/pino_darcy_tensorboard2.png
   :alt: FNO vs PINO Darcy Tensorboard (small data regime)
   :width: 70.0%
   :align: center

   Comparison between PINO and FNO accuracy for surrogate modeling Darcy flow (small data regime).
   
You can observe that, in this case, the PINO outperforms the FNO.


.. rubric:: References / Footnotes

.. [#li2021physics] Li, Zongyi, et al. "Physics-informed neural operator for learning partial differential equations." arXiv preprint arXiv:2111.03794 (2021).

.. [#ft1] Note that the "exact" method is technically not exact because it uses a combination of numerical spectral derivatives and exact differentiation. See the original paper for more details.
