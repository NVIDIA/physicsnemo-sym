.. _ParameterizedSim:

Parameterized 3D Heat Sink
==========================

Introduction
------------

This tutorial walks through the process of simulating a
parameterized problem using PhysicsNeMo Sym. The neural networks in PhysicsNeMo Sym allow
us to solve problems for multiple parameters/independent variables in a single
training. These parameters can be geometry variables, coefficients of a PDE or even boundary conditions. 
Once the training is complete, it is possible to run inference on several geometry/physical parameter
combinations as a post-processing step, without solving the forward
problem again. You will see that such parameterization increases the
computational cost only fractionally while solving the entire desired
design space.

To demonstrate this feature, this example will solve the flow and heat over a 3-fin
heat sink whose fin height, fin thickness, and fin length are variable.
We will then perform a design optimization to find out the most optimal
fin configuration for the heat sink example. By the end of this
tutorial, you would learn to easily convert any simulation to a
parametric design study using PhysicsNeMo Sym's CSG module and Neural Network
solver. In this tutorial, you would learn the following:

#. How to set up a parametric simulation in PhysicsNeMo Sym.

.. note::
   This tutorial is an extension of tutorial :ref:`cht` which
   discussed how to use PhysicsNeMo Sym for solving Conjugate Heat problems. This 
   tutorial uses the same geometry setup and solves it for a
   parameterized setup at an increased Reynolds number. Hence, it is recommended
   that you to refer tutorial :ref:`cht` for any additional details
   related to geometry specification and boundary conditions. 

   The same scripts used in example :ref:`cht` will be used. To make the simulation parameterized and turbulent, you will set the custom flags ``parameterized`` and ``turbulent`` both as ``true`` in the config files. 

.. note::
   In this tutorial the focus will be on parameterization which is independent of the physics being solved and can be applied to any class of problems covered in the User Guide.


Problem Description
-------------------

Please refer the geometry and boundary conditions for a 3-fin heat sink
in tutorial :ref:`cht`. We will parameterize this problem to solve
for several heat sink designs in a single neural network training. We
will modify the heat sink’s fin dimensions (thickness, length and
height) to create a design space of various heat sinks. The Re for this case is now 500 and you will incorporate turbulence using
Zero Equation turbulence model.

For this problem, you will vary the height (:math:`h`),
length (:math:`l`), and thickness (:math:`t`) of the central fin and the
two side fins. The height, length, and thickness of the two side fins
are kept the same, and therefore, there will be a total of six geometry
parameters. The ranges of variation for these geometry parameters are
given in equation :eq:`param_ranges`.

.. math::
   :label: param_ranges

   \begin{split}
   h_{central fin} &= (0.0, 0.6),\\
   h_{side fins} &= (0.0, 0.6),\\
   l_{central fin} &= (0.5, 1.0) \\
   l_{side fins} &= (0.5, 1.0) \\
   t_{central fin} &= (0.05, 0.15) \\
   t_{side fins} &= (0.05, 0.15)
   \end{split}

.. figure:: /images/user_guide/parameterized_3fin.png
   :alt: Examples of some of the 3 Fin geometries covered in the chosen design space
   :name: fig:param_3_fin_1
   :width: 50.0%
   :align: center

   Examples of some of the 3 Fin geometries covered in the chosen design space

Case Setup
----------

In this tutorial, you will use the 3D geometry module from PhysicsNeMo Sym to
create the parameterized 3-fin heat sink geometry. Discrete parameterization can sometimes lead to discontinuities in the solution making the training harder. 
Hence tutorial only covers parameters that are continuous. Also, you will be training the
parameterized model and validating it by performing inference on a case
where :math:`h_{central fin}=0.4`, :math:`h_{side fins}=0.4`,
:math:`l_{central fin}=1.0`, :math:`l_{side fins}=1.0`,
:math:`t_{central fin}=0.1`, and :math:`t_{side fins}=0.1`. At the end
of the tutorial a comparison between results for
the above combination of parameters obtained from a parameterized model
versus results obtained from a non-parameterized model trained on just a
single geometry corresponding to the same set of values is presented. This will
highlight the usefulness of using PINNs for doing parameterized
simulations in comparison to some of the traditional methods.

Since the majority of the problem definition and setup was covered in :ref:`cht`, this tutorial will focus only on important elements for the parameterization. 

Creating Nodes and Architecture for Parameterized Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameters chosen for variables act as additional inputs to the neural network. The outputs remain the same. Also, for this example since the variables are geometric only, 
no change needs to be made for how the equation nodes are defined (except the addition of turbulence model). In cases where the coefficients of a PDE are parameterized, the corresponding coefficient needs to be defined symbolically (i.e. using string) in the equation node. 

Note for this example, the viscosity is set as a string in the ``NavierStokes`` constructor for the purposes of turbulence model. The ``ZeroEquation`` equation node ``'nu'`` as the output node which acts as input to the momentum equations in Navier-Stokes. 


The code for this parameterized problem is shown below. Note that ``parameterized`` and ``turbulent`` are set to ``true`` in the config file.    

Parameterized flow network:

.. literalinclude:: ../../../examples/three_fin_3d/three_fin_flow.py
   :language: python
   :lines: 51-86
   :emphasize-lines: 12-23

Parameterized heat network:
 
.. literalinclude:: ../../../examples/three_fin_3d/three_fin_thermal.py
   :language: python
   :lines: 51-95
   :emphasize-lines: 9-20


Setting up Domain and Constraints 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section is again very similar to :ref:`cht` tutorial. The only difference being, now 
the input to ``parameterization`` argument is a dictionary of key value pairs where the keys are strings for each design variable and the values are tuples of float/ints specifying the range of variation for those variables.

The code to setup these dictionaries for parameterized inputs and constraints can be found below.

Setting the parameter ranges (``three_fin_geometry.py``)

.. literalinclude:: ../../../examples/three_fin_3d/three_fin_geometry.py
   :language: python
   :lines: 33-59

.. literalinclude:: ../../../examples/three_fin_3d/three_fin_geometry.py
   :language: python
   :lines: 75-96
   :emphasize-lines: 4-10

Setting the ``parameterization`` argument in the constraints. 
Here, only a few BCs from the flow domain are shown for example purposes. 
But the same settings are applied to all the other BCs. 

.. literalinclude:: ../../../examples/three_fin_3d/three_fin_flow.py
   :language: python
   :lines: 97-113
   :emphasize-lines: 14

.. literalinclude:: ../../../examples/three_fin_3d/three_fin_flow.py
   :language: python
   :lines: 182-200
   :emphasize-lines: 14

Training the Model
------------------

This part is exactly similar to tutorial :ref:`cht` and once all
the definitions are complete, you can execute the parameterized problem
like any other problem. 


Design Optimization
-------------------

As discussed previously, you can optimize the design once the training
is complete as a post-processing step. A typical design
optimization usually contains an objective function that is
minimized/maximized subject to some physical/design constraints.

For heat sink designs, usually the peak temperature that can be reached
at the source chip is limited. This limit arises from the operating
temperature requirements of the chip on which the heat sink is mounted
for cooling purposes. The design is then constrained by the maximum
pressure drop that can be successfully provided by the cooling system
that pushes the flow around the heat sink. Mathematically this can be
expressed as below:

.. table:: Optimization problem
   :align: center

   +-----------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------+
   |                 | Variable/Function                                                                                      | Description                                          |
   +-----------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------+
   | minimize        | :math:`Peak \text{ } Temperature`                                                                      | Minimize the peak temperature at the source chip     |
   +-----------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------+
   | with respect to | :math:`h_{central fin}, h_{side fins}, l_{central fin}, l_{side fins}, t_{central fin}, t_{side fins}` | Geometric Design variables of the heat sink          |
   +-----------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------+
   | subject to      | :math:`Pressure \text{ } drop < 2.5`                                                                   | Limit on the pressure drop (Max pressure             |
   |		     |													      |	drop that can be provided by cooling system          |
   +-----------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------+

Such optimization problems can be easily achieved in PhysicsNeMo Sym once you have a trained, parameterized model ready. 
As it can be noticed, while solving the parameterized simulation you
created some monitors to track the peak temperature and the pressure
drop for some design variable combination. You will basically would follow the
same process and use the ``PointwiseMonitor`` constructor to find the values for
multiple combinations of the design variables. You can create
this simply by looping through the multiple designs. Since these monitors can be for large number of design variable combinations, you are recommended to use these
monitors only after the training is complete to achieve better
computational efficiency. Do do this, once the models are trained, you can run the flow and thermal models in the ``'eval'`` mode by specifying: ``'run_mode=eval'``
in the config files. 

After the models are run in the ``'eval'`` mode, the pressure drop and peak temperature values will be saved in form of a ``.csv`` file. Then, 
one can write a simple scripts to sift through the various samples and pick the most optimal ones that minimize/maximize the objective function while meeting the
required constraints (for this example, the design with the least peak temperature and the maximum pressure drop < 2.5):

.. literalinclude:: ../../../examples/three_fin_3d/three_fin_design.py
   :language: python
   :lines: 1-


Results
-------

The design parameters for the optimal heat sink for this problem
are: :math:`h_{central fin} = 0.4`, :math:`h_{side fins} = 0.4`,
:math:`l_{central fin} = 0.83`, :math:`l_{side fins} = 1.0`,
:math:`t_{central fin} = 0.15`, :math:`t_{side fins} = 0.15`. The above
design has a pressure drop of 2.46 and a peak temperature of 76.23
:math:`(^{\circ} C)` :numref:`fig-optimal_3Fin`

.. _fig-optimal_3Fin:

.. figure:: /images/user_guide/optimal_3fin.png
   :alt: Three Fin geometry after optimization
   :width: 30.0%
   :align: center

   Three Fin geometry after optimization

:numref:`table-parameterized1` represents the computed pressure
drop and peak temperature for the OpenFOAM single geometry and PhysicsNeMo Sym
single and parameterized geometry runs. It is evident that the results
for the parameterized model are close to those of a single geometry
model, showing its good accuracy.

.. _table-parameterized1:

.. table:: A comparison for the OpenFOAM and PhysicsNeMo Sym results
   :align: center

   +-----------------------+-----------------+------------+-----------------+
   | Property              | OpenFOAM Single | Single Run | Parameterized   |
   |                       | Run             |            | Run             |
   +-----------------------+-----------------+------------+-----------------+
   | Pressure Drop         | 2.195           | 2.063      | 2.016           |
   | :math:`(Pa)`          |                 |            |                 |
   +-----------------------+-----------------+------------+-----------------+
   | Peak                  | 72.68           | 76.10      | 77.41           |
   | Temperature           |                 |            |                 |
   | :math:`(^{\circ} C)`  |                 |            |                 |
   +-----------------------+-----------------+------------+-----------------+

By parameterizing the geometry, PhysicsNeMo Sym significantly accelerates design
optimization when compared to traditional solvers, which are limited to
single geometry simulations. For instance, 3 values (two end values of
the range and a middle value) per design variable would result in
:math:`3^6 = 729` single geometry runs. The total compute time required
by OpenFOAM for this design optimization would be 4099 hrs. (on 20 processors). PhysicsNeMo Sym can achieve 
the same design optimization at ~17x lower computational cost. Large number
of design variables or their values would only magnify the difference in
the time taken for two approaches.

.. note::
   The PhysicsNeMo Sym calculations were done using 4 NVIDIA V100 GPUs. The OpenFOAM calculations were done using 20 processors.

.. figure:: /images/user_guide/flow_field_with_T_4_cropped.png
   :name: fig:optimal_3Fin_result
   :width: 70.0%
   :align: center

   Streamlines colored with pressure and temperature profile in the fluid for optimal three fin geometry

Here, the 3-Fin heatsink was solved for arbitrary heat properties chosen
such that the coupled conjugate heat transfer solution was possible.
However, such approach causes issues when the conductivities are orders
of magnitude different at the interface. We will revisit the conjugate
heat transfer problem in tutorial :ref:`2d_heat` and :ref:`limerock` to see some advanced tricks/schemes that one can use to handle the 
issues that arise in Neural network training when real material properties are involved.
 
