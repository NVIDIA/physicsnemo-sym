.. _linear_elasticity:

Linear Elasticity
=================

Introduction
------------

This tutorial illustrates the linear elasticity
implementation in PhysicsNeMo Sym. PhysicsNeMo Sym offers the capability of solving the
linear elasticity equations in the differential or variational form,
allowing to solve a wide range of problems with a variety of boundary
conditions. Three examples are presented in this chapter, namely the 3D
bracket, the fuselage panel, and the plane displacement, to discuss the
details of the linear elasticity in PhysicsNeMo Sym. In this tutorial, you
will learn:

-  How to solve linear elasticity equations using the differential and
   variational forms.

-  How to solve linear elasticity problems for 3D and thin 2D structures
   (plane stress).

-  How to nondimensionalize the elasticity equations.

.. note::
   This tutorial assumes that you have completed :ref:`Introductory Example` tutorial
   and have familiarized yourself with the basics of
   the PhysicsNeMo Sym APIs. See :ref:`weak-solutions-pinn` 
   for more details on weak solutions of PDEs. Some of the boundary
   conditions are defined more elaborately in the tutorial
   :ref:`variational-example` . 

   The linear elasticity equations in PhysicsNeMo Sym can be found in the 
   source code directory ``physicsnemo/eq/pdes/linear_elasticity.py``.

.. warning::

   The Python package `quadpy <https://github.com/nschloe/quadpy>`_ is required for this example.
   Install using ``pip install quadpy`` (Also refer to :ref:`install_physicsnemo_docker`).

Linear Elasticity in the Differential Form
------------------------------------------

Linear elasticity equations in the displacement form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The displacement form of the (steady state) linear elasticity
equations, known as the Navier equations, is defined as

.. math:: (\lambda + \mu) u_{j,ji} + \mu u_{i,jj} + f_i = 0,

where :math:`u_i` is the displacement vector, :math:`f_i` is the body
force per unit volume, and :math:`\lambda, \mu` are the Lamé parameters
defined as

.. math:: \lambda = \frac{E \nu}{(1+\nu)(1-2\nu)},

.. math:: \mu = \frac{E}{2(1+\nu)}.

Here, :math:`E`, :math:`v` are, respectively, the Young’s modulus and
Poisson’s ratio.

Linear elasticity equations in the mixed form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the displacement formulation, linear elasticity can also
be described by the mixed-variable formulation. Based on internal experiments 
as well as the studies reported in [#rao2020physics]_, the mixed-variable formulation is
easier for a neural network solver to learn, possibly due to the lower
order differential terms. In the mixed-variable formulation, the
equilibrium equations are defend as:

.. math::
   :label: eqn-equilibrium
       
       \sigma_{ji,j} + f_i = 0,

where :math:`\sigma_{ij}` is the Cauchy stress tensor. the
stress-displacement equations are also defined as

.. math:: \sigma_{ij} = \lambda \epsilon_{kk} \delta_{ij} + 2 \mu \epsilon_{ij},

where :math:`\delta_{ij}` is the Kronecker delta function and
:math:`\epsilon_{ij}` is the strain tensor that takes the following form

.. math:: \epsilon_{ij} = \frac{1}{2}\left( u_{i,j} + u_{j,i} \right).

Nondimensionalized linear elasticity equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is often advantageous to work with the nondimensionalized and
normalized variation of the elasticity equations to improve the training
convergence and accuracy. The 
nondimensionalized variables can be defined as:

.. math:: \hat{x}_i = \frac{x_i}{L},

.. math:: \hat{u}_i = \frac{u_i}{U},

.. math:: \hat{\lambda} = \frac{\lambda}{\mu_c},

.. math:: \hat{\mu} = \frac{\mu}{\mu_c}

Here, :math:`L` is the characteristic length, :math:`U` is the
characteristic displacement, and :math:`\mu_c` is the
nondimensionalizing shear modulus. The
nondimensionalized Navier and equilibrium equations can be obtained by multiplying both
sides of equations by :math:`L^2/\mu_c U`, as follows:

.. math:: (\hat{\lambda} + \hat{\mu}) \hat{u}_{j,ji} + \hat{\mu} \hat{u}_{i,jj} + \hat{f}_i = 0,

.. math:: \hat{\sigma}_{ji,j} + \hat{f_i} = 0,

where the nondimensionalized body force and stress tensor are

.. math:: \hat{f}_{i} = \frac{L^2}{\mu_c U} f_{i},

.. math:: \hat{\sigma}_{ij} = \frac{L}{\mu_c U}\sigma_{ij}.

Similarly, the nondimensionalized form of the stress-displacement
equations are obtained by multiplying both sides of equations by
:math:`L/\mu_c U`, as follows:

.. math::
   :label: eqn-stress_displacement
   
       \hat{\sigma}_{ij} = \hat{\lambda} \hat{\epsilon}_{kk} \delta_{ij} + 2 \hat{\mu} \hat{\epsilon}_{ij},

.. math:: \hat{\epsilon}_{ij} = \frac{1}{2}\left( \hat{u}_{i,j} + \hat{u}_{j,i} \right).

Plane stress equations
~~~~~~~~~~~~~~~~~~~~~~

In a plane stress setting for thin structures, it is assumed that

.. math:: \hat{\sigma}_{zz} =  \hat{\sigma}_{xz} = \hat{\sigma}_{yz} = 0,

and therefore, the following relationship holds

.. math:: \hat{\sigma}_{zz} = \hat{\lambda} \left(\frac{\partial \hat{u}}{\partial \hat{x}} + \frac{\partial \hat{v}}{\partial \hat{y}} + \frac{\partial \hat{w}}{\partial \hat{z}} \right) + 2 \hat{\mu} \frac{\partial \hat{w}}{\partial \hat{z}} = 0 \Rightarrow \frac{\partial \hat{w}}{\partial \hat{z}} = \frac{-\hat{\lambda}}{(\hat{\lambda} +2\hat{\mu})}  \left(\frac{\partial \hat{u}}{\partial \hat{x}} + \frac{\partial \hat{v}}{\partial \hat{y}} \right).

Accordingly, the equations for :math:`\hat{\sigma}_{xx}` and
:math:`\hat{\sigma}_{yy}` can be updated as follows

.. math::
   :label: eqn-plane-stress-x
   
       \hat{\sigma}_{xx} = \hat{\lambda} \left(\frac{\partial \hat{u}}{\partial \hat{x}} + \frac{\partial \hat{v}}{\partial \hat{y}} + \frac{-\hat{\lambda}}{(\hat{\lambda} +2\hat{\mu})}  \left(\frac{\partial \hat{u}}{\partial \hat{x}} + \frac{\partial \hat{v}}{\partial \hat{y}} \right)  \right) + 2 \hat{\mu} \frac{\partial \hat{u}}{\partial \hat{x}}

.. math::
   :label: eqn-plane-stress-y 
   
       \hat{\sigma}_{yy} = \hat{\lambda} \left(\frac{\partial \hat{u}}{\partial \hat{x}} + \frac{\partial \hat{v}}{\partial \hat{y}} + \frac{-\hat{\lambda}}{(\hat{\lambda} +2\hat{\mu})}  \left(\frac{\partial \hat{u}}{\partial \hat{x}} + \frac{\partial \hat{v}}{\partial \hat{y}} \right)  \right) + 2 \hat{\mu} \frac{\partial \hat{v}}{\partial \hat{y}}.


Problem 1: Deflection of a bracket
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This linear elasticity example, shows a 3D bracket 
in :numref:`fig-bracket-geometry`. This example is partially
adopted from the `MATLAB PDE toolbox <https://www.mathworks.com/help/pde/ug/deflection-analysis-of-a-bracket.html>`_.
The back face of this bracket is clamped, and a traction of
:math:`4 \times 10^4 \, \text{Pa}` is applied to the front face in the
negative-:math:`z` direction (this face is shown in red). The rest of
the surface is considered as traction free boundaries. Other properties are set as,
:math:`(E, \nu) = (100 \,\text{GPa}, 0.3)`. You can nondimensionalize the
linear elasticity equations by setting
:math:`L=1 \,\text{m}, U=0.0001 \,\text{m}, \mu_c=0.01 \mu`.

.. _fig-bracket-geometry:

.. figure:: /images/user_guide/bracket_geometry.png
   :width: 40.0%
   :align: center

   Geometry of the bracket. The back face of the bracket is clamped, and a shear stress is applied to the front face in the negative z-direction.


.. literalinclude:: ../../../examples/bracket/bracket.py
   :language: python
   :lines: 43-54

In general, the characteristic length can be chosen in such a
way to bound the largest dimension of the geometry to :math:`(-1,1)`.
The characteristic displacement and :math:`\mu_c` can also be chosen
such that the maximum displacement and the applied traction are close to
1 in order. Although the maximum displacement is not known a priori, it is 
observed that the convergence is not sensitive to the choice of the
characteristic displacement as long as it is reasonably selected based
on an initial guess for the approximate order of displacement. More
information on nondimensionalizing the PDEs can be found in `Scaling of
Differential
Equations <https://hplgit.github.io/scaling-book/doc/pub/book/html/sphinx-cbc/index.html>`_.

Case Setup and Results
^^^^^^^^^^^^^^^^^^^^^^

The complete python script for this problem can be found at
``examples/bracket/bracket.py``. Two separate neural networks for displacement and stresses are used as follows

.. literalinclude:: ../../../examples/bracket/bracket.py
   :language: python
   :lines: 56-79

The mixed form of the linear elasticity equations is used here in this
example, and therefore, the training constraints are defined as shown below:


.. literalinclude:: ../../../examples/bracket/bracket.py
   :language: python
   :lines: 81-

The training constraints consists of two different sets of interior
points (i.e., ``interior_support`` and ``interior_bracket``). This is
done only to generate the interior points more efficiently. 


:numref:`fig-bracket-results` shows
the PhysicsNeMo Sym results and also a comparison with a commercial solver
results. The results of these two solvers show good agreement, with only
a 8% difference in maximum bracket displacement.

.. _fig-bracket-results:

.. figure:: /images/user_guide/bracket_results_combined.png
   :width: 100.0%
   :align: center

   PhysicsNeMo Sym linear elasticity results for the bracket deflection example

Problem 2: Stress analysis for aircraft fuselage panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses PhysicsNeMo Sym for the analysis of stress
concentration in an aircraft fuselage panel. Depending on the altitude
of the flying plane, the fuselage panel is exposed to different values
of hoop stress that can cause accumulated damage to the panel over the
time. Therefore, in cumulative damage modeling of aircraft fuselage for
the purpose of design and maintenance of aircrafts, it is required to
perform several stress simulations for different hoop stress values.
Consider a simplified aircraft fuselage panel as shown in
:numref:`fig-panel-geometry`. This example constructs a parameterized model
that, once trained, can predict the stress and displacement in the panel
for different values of hoop stress.

.. _fig-panel-geometry:

.. figure:: /images/user_guide/panel.png
   :alt: Geometry and boundary conditions of the simplified aircraft fuselage panel.
   :width: 24.0%
   :align: center

   Geometry and boundary conditions of the simplified aircraft fuselage panel.

Case Setup and Results
^^^^^^^^^^^^^^^^^^^^^^

The panel material is Aluminium 2024-T3, with
:math:`(E, \nu) = (73 \,\text{GPa}, 0.33)`. The objective is to train a parameterized
model with varying :math:`\sigma_{hoop} \in (46, 56.5)`. Since the
thickness of the panel is very small (i.e., :math:`2 \, \text{mm}`), 
plane stress equations can be used in this example. The python script
for this problem can be found at
``examples/fuselage_panel/panel_solver.py``. The plane stress form of
the linear elasticity equations in PhysicsNeMo Sym can be called by using the
``LinearElasticityPlaneStress`` class:

.. literalinclude:: ../../../examples/fuselage_panel/panel.py
   :language: python
   :lines: 44-56

:numref:`fig-panel-results` shows
the PhysicsNeMo Sym results for panel displacements and stresses. For comparison,
the commercial solver results are also presented in :numref:`fig-panel-abaqus-results`. The PhysicsNeMo Sym
and commercial solver results are in close agreement, with a difference
of less than 5% in the maximum Von Mises stress.

.. _fig-panel-results:

.. figure:: /images/user_guide/panel_results_combined.png
   :alt: PhysicsNeMo Sym linear elasticity results for the aircraft fuselage panel example with parameterized hoop stress. The results are for :math:`\sigma_{hoop}` = 46
   :width: 100.0%
   :align: center

   PhysicsNeMo Sym linear elasticity results for the aircraft fuselage panel example with parameterized hoop stress. The results are for :math:`\sigma_{hoop}` = 46

.. _fig-panel-abaqus-results:

.. figure:: /images/user_guide/panel_commercial_results_combined.png
   :alt: Commercial solver linear elasticity results for the aircraft fuselage panel example with :math:`\sigma_{hoop}` = 46
   :width: 100.0%
   :align: center

   Commercial solver linear elasticity results for the aircraft fuselage panel example with :math:`\sigma_{hoop}` = 46


Linear Elasticity in the Variational Form
-----------------------------------------

Linear elasticity equations in the variational form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the differential form of the linear elasticity equations,
PhysicsNeMo Sym also enables the use of variational form of these equations
(:ref:`variational-example`). This section presents 
the linear elasticity equations in the variational
form as implemented in PhysicsNeMo Sym. Nondimensionalized
variables will be used in this derivation. The inner product of
Equation :eq:`eqn-equilibrium` and a vector test
function :math:`v \in \mathbf{V}`, and integrating over the domain is given as
follows

.. math:: \int_{\Omega} \hat{\sigma}_{ji,j} v_i d \mathbf{x}+ \int_{\Omega} \hat{f_i} v_i d \mathbf{x} = 0.

Using the integration by parts, 

.. math::
   :label: eqn-elasticity_variational
   
      \int_{\partial \Omega} \hat{T_i} v_i d \mathbf{s}  -\int_{\Omega} \hat{\sigma}_{ji} v_{j,i} d \mathbf{x} + \int_{\Omega} \hat{f_i} v_i d \mathbf{x} = 0,

where :math:`T_i` is the traction. The first term is zero for
the traction free boundaries. Equation
:eq:`eqn-elasticity_variational` is the
variational form of the linear elasticity equations that is adopted in
PhysicsNeMo Sym. The term :math:`\hat{\sigma}_{ji}` in this equation is computed
using the stress-displacement relation in Equation
:eq:`eqn-stress_displacement`.

Problem 3: Plane displacement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example solves the linear elasticity plane stress equations
in the variational form. Consider a square plate that is clamped from
one end and is under a displacement boundary condition on the other end,
as shown in :numref:`fig-plane_displacement`. The rest of the
boundaries are traction free. The material properties are assumed to be
:math:`(E, \nu) = (10 \,\text{MPa}, 0.2)`. This example is adopted from
[#rao2020physics]_.


Case Setup and Results
^^^^^^^^^^^^^^^^^^^^^^

To solve this problem in the variational form, the
displacement boundary conditions are enforced in the differential form. Also
interior and boundary points are generated to be used in evaluation of the
integrals in Equation :eq:`eqn-elasticity_variational`. The
python script for this problem can be found at
``examples/plane_displacement/plane_displacement.py``.

.. _fig-plane_displacement:

.. figure:: /images/user_guide/plane_displacement.png
   :width: 35.0%
   :align: center

   Geometry and boundary conditions of the plane displacement example. This example is adopted from [#rao2020physics]_.


.. literalinclude:: ../../../examples/plane_displacement/plane_displacement.py
   :language: python
   :lines: 205-


The displacement boundary conditions have been included in
the regular PINN loss function. For the variational constraints, first, ``VariationalDataset`` for each bc is created by specifying the ``invar``
and the required ``outvar_names``. These output variables will be used while computing the variational loss. A ``VariationalConstraint`` is then constructed using the dictionary of ``variational_datasets`` and the nodes of the network. For loss, a ``DGLoss()`` is specified which is a custom loss function that includes the variational loss. The remainder
of this subsection, covers how to generate this variational
loss (``DGLoss()``). First, the neural network solution and the gradients at the interior and
boundary points are read:

.. literalinclude:: ../../../examples/plane_displacement/plane_displacement.py
   :language: python
   :lines: 58-117

In the next step, the test functions are defined, and the test
functions and their required gradients are computed at the interior and boundary
points:

.. literalinclude:: ../../../examples/plane_displacement/plane_displacement.py
   :language: python
   :lines: 120-124

Here, a set of test functions consisting of Legendre
polynomials and trigonometric functions is constructed, and 2% of
these test functions are randomly sampled. Only the terms that appear in
the variational loss in Equation
:eq:`eqn-elasticity_variational` are computed here. For
instance, it is not necessary to compute the derivative of the test
functions with respect to input coordinates for the boundary points.

The next step is to compute the stress terms according to the plane
stress equations in Equations
:eq:`eqn-plane-stress-x`, :eq:`eqn-plane-stress-y`, and also the traction
terms:

.. literalinclude:: ../../../examples/plane_displacement/plane_displacement.py
   :language: python
   :lines: 127-173

Finally, following the Equation
:eq:`eqn-elasticity_variational`, 
variational interior and boundary integral terms are defined and the variational loss is formulated and added 
to the overall loss as follows:

.. literalinclude:: ../../../examples/plane_displacement/plane_displacement.py
   :language: python
   :lines: 175-202

:numref:`fig-plane_displacement-results` shows the PhysicsNeMo Sym results for this
plane displacement example. The results are in good agreement with the
FEM results reported in [#rao2020physics]_, verifying
the accuracy of the PhysicsNeMo Sym results in the variational form.

.. _fig-plane_displacement-results:

.. figure:: /images/user_guide/plane_displacement_results_combined.png
   :width: 100.0%
   :align: center

   PhysicsNeMo Sym results for the plane displacement example.



.. rubric:: References

.. [#rao2020physics] Chengping Rao, Hao Sun, and Yang Liu. Physics informed deep learning for computational elastodynamics without labeled data. arXiv preprint arXiv:2006.08472, 2020.
   
