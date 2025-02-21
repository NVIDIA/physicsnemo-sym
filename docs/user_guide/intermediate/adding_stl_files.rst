.. _stl:

STL Geometry: Blood Flow in Intracranial Aneurysm
=================================================

Introduction
------------

In this tutorial, you will import an STL file for a complicated geometry
and use PhysicsNeMo Sym' SDF library to sample points on the surface and the
interior of the STL and train the PINNs to predict flow in this complex
geometry. In this tutorial you will learn the following:

#. How to import an STL file in PhysicsNeMo Sym and sample points in the interior
   and on the surface of the geometry.

.. figure:: /images/user_guide/aneurysm.png
   :alt: Aneurysm STL file
   :width: 40.0%
   :align: center

   Aneurysm STL file

.. note::
   This tutorial assumes that you have completed tutorial :ref:`Introductory Example`
   and have familiarized yourself with the basics
   of the PhysicsNeMo Sym APIs. Additionally, to use the modules
   described in this tutorial, make sure your system satisfies the
   requirements for SDF library (:ref:`system_requirements`).
   
   For the interior sampling to work, ensure that the STL
   geometry is watertight. This requirement is not necessary for sampling
   points on the surface.

   All the python scripts for this problem can be found at ``examples/aneurysm/``.

Problem Description
-------------------

This simulation, uses a no-slip boundary condition on the walls
of the aneurysm :math:`u,v,w=0`. For the inlet, a parabolic flow
where the flow goes in the normal direction of the inlet and has peak
velocity 1.5, is used. The outlet has a zero pressure condition, :math:`p=0`. The
kinematic viscosity of the fluid is :math:`0.025` and the density is a
constant :math:`1.0`.

Case Setup
----------

In this tutorial, you will use PhysicsNeMo Sym' ``Tessellation`` module to sample points
using a STL geometry. The module works similar to PhysicsNeMo Sym' geometry
module. Which means you can use ``PointwiseInteriorConstraint`` and ``PointwiseBoundaryConstraint`` to sample points in the interior and the boundary of the geometry and define
appropriate constraints. Separate STL files for each boundary of the 
geometry and another watertight geometry for sampling points in the interior of the
geometry are required.

Importing the required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The list of required packages can be found below. Import PhysicsNeMo Sym'
``Tessellation`` module to the sample points on the STL geometry.

.. literalinclude:: ../../../examples/aneurysm/aneurysm.py
   :language: python
   :lines: 15-37

Using STL files to generate point clouds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import the STL geometries using the ``Tessellation.from_stl()``
function. This function takes in the path of the STL geometry as input.
You will need to specify the value of attribute ``airtight`` as ``False``
for the open surfaces (eg. boundary STL files).

Then these mesh objects can be used to create boundary or interior 
constraints similar to tutorial :ref:`Introductory Example` using the ``PointwiseBoundaryConstraint`` or
``PointwiseInteriorConstraint``. 

.. note::
   For this tutorial, you can normalize the geometry by scaling it and centering 
   it about the origin (0, 0, 0). This will help in speeding up the training process.
 
The code to sample using STL geometry, define all these functions,
boundary conditions is shown below.

.. literalinclude:: ../../../examples/aneurysm/aneurysm.py
   :language: python
   :lines: 43-98

Defining the Equations, Networks and Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This process is similar to other tutorials. In this problem you are only solving
for laminar flow, so you can use only ``NavierStokes`` and
``NormalDotVec`` equations and define a network similar to
tutorial :ref:`Introductory Example`. The code to generate the Network and required nodes is shown below.

.. literalinclude:: ../../../examples/aneurysm/aneurysm.py
   :language: python
   :lines: 112-125


Setting up Domain and adding Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that you have all the nodes and geometry elements defined, you can use the tesselated/mesh 
objects to create boundary or interior constraints similar to tutorial :ref:`Introductory Example` using 
the ``PointwiseBoundaryConstraint`` or ``PointwiseInteriorConstraint``. 

.. literalinclude:: ../../../examples/aneurysm/aneurysm.py
   :language: python
   :lines: 109-110, 126-192


Adding Validators and Monitors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The process of adding validation data and monitors is similar to previous tutorials. 
This example uses the simulation from OpenFOAM for validating the
PhysicsNeMo Sym results. Also, you can create a monitor for pressure drop across
the aneurysm to monitor the convergence and compare against OpenFOAM
data. The code to generate the these domains is shown below.

.. literalinclude:: ../../../examples/aneurysm/aneurysm.py
   :language: python
   :lines: 194-235

Training the model 
------------------

Once the python file is setup, the training can be simply started by executing the python script. 

.. code:: bash

   python aneurysm.py

Results and Post-processing
---------------------------

We use this tutorial to give an example of overfitting of training data in the
PINNs. :numref:`fig-aneurysm_errors` shows the comparison of the
validation error plots achieved for two different point densities. The
case using 10 M points shows an initial
convergence which later diverges even when the training error keeps
reducing. This implies that the network is overfitting the sampled
points while sacrificing the accuracy of flow in between them.
Increasing the points to 20 M solves that problem and the flow field is generalized 
to a better resolution.

.. _fig-aneurysm_errors:

.. figure:: /images/user_guide/val_errors.png
   :alt: Convergence plots for different point density
   :width: 100.0%
   :align: center

   Convergence plots for different point density

:numref:`fig-aneurysm-p` shows the pressure developed inside the
aneurysm and the vein. A cross-sectional view in 
:numref:`fig-aneurysm-v` shows the distribution of velocity magnitude
inside the aneurysm. One of the key challenges of this problem is
getting the flow to develop inside the aneurysm sac and the streamline
plot in :numref:`fig-aneurysm-stream` shows that PhysicsNeMo Sym
successfully captures the flow field inside.


.. _fig-aneurysm-v:

.. figure:: /images/user_guide/aneurysm_v_mag_labelled.png
   :alt: Cross-sectional view aneurysm showing velocity magnitude. Left: PhysicsNeMo Sym. Center: OpenFOAM. Right: Difference
   :width: 100.0%
   :align: center

   Cross-sectional view aneurysm showing velocity magnitude. Left: PhysicsNeMo Sym. Center: OpenFOAM. Right: Difference


.. _fig-aneurysm-p:

.. figure:: /images/user_guide/aneurysm_p_labelled.png
   :alt: Pressure across aneurysm. Left: PhysicsNeMo Sym. Center: OpenFOAM. Right: Difference
   :width: 100.0%
   :align: center

   Pressure across aneurysm. Left: PhysicsNeMo Sym. Center: OpenFOAM. Right: Difference


.. _fig-aneurysm-stream:

.. figure:: /images/user_guide/aneurysm_streamlines3_crop.png
   :alt: Flow streamlines inside the aneurysm generated from PhysicsNeMo Sym simulation.
   :width: 50.0%
   :align: center

   Flow streamlines inside the aneurysm generated from PhysicsNeMo Sym simulation.


Accelerating the Training of Neural Network Solvers via Transfer Learning
-------------------------------------------------------------------------

Numerous applications in science and engineering require repetitive
simulations, such as simulation of blood flow in different
patient specific models. Traditional solvers simulate these models
independently and from scratch. Even a minor change to the model
geometry (such as an adjustment to the patient specific medical image
segmentation) requires a new simulation. Interestingly, and unlike the
traditional solvers, neural network solvers can transfer knowledge
across different neural network models via transfer learning. In
transfer learning, the knowledge acquired by a (source) trained neural
network model for a physical system is transferred to another (target)
neural network model that is to be trained for a similar physical system
with slightly different characteristics (such as geometrical
differences). The network parameters of the target model are initialized
from the source model, and are retrained to cope with the new system
characteristics without having the neural network model trained from
scratch. This transfer of knowledge effectively reduces the time to
convergence for neural network solvers. As an example, :numref:`fig-aneurysm_transfer_learning` shows the application of
transfer learning in training of neural network solvers for two
intracranial aneurysm models with different sac shapes.

.. _fig-aneurysm_transfer_learning:

.. figure:: /images/user_guide/aneurysm_transfer_learning.png
   :alt: Transfer learning accelerates intracranial aneurysm simulations. Results are for two intracranial aneurysms with different sac shapes.
   :width: 70.0%
   :align: center

   Transfer learning accelerates intracranial aneurysm simulations. Results are for two intracranial aneurysms with different sac shapes.

To use transfer learning in PhysicsNeMo Sym, set ``'initialize_network_dir'`` in the configs
to the source model network checkpoint. Also, since in transfer learning
you fine-tune the source model instead of training from scratch, use a
relatively smaller learning rate compared to a full run, with smaller
number of iterations and faster decay, as shown below.

.. code:: yaml
    
    defaults :
      - physicsnemo_default
      - arch:
          - fully_connected
      - scheduler: tf_exponential_lr
      - optimizer: adam
      - loss: sum
      - _self_
    
    scheduler:
      decay_rate: 0.95
      #decay_steps: 15000 # full run
      decay_steps: 6000   # TL run
    
    network_dir : "network_checkpoint_target"
    initialization_network_dir : "../aneurysm/network_checkpoint_source/"

    training:
      rec_results_freq : 10000
      rec_constraint_freq: 50000
      #max_steps: 1500000 # full run
      max_steps: 400000   # TL run
    
    batch_size:
      inlet: 1100
      outlet: 650
      no_slip: 5200
      interior: 6000
      integral_continuity: 310

