.. role:: raw-latex(raw)
   :format: latex
..

.. _variational-example:

Interface Problem by Variational Method
=======================================

Introduction
------------

This tutorial demonstrates the process of solving a PDE
using the variational formulation. It shows how to use the variational
method to solve the interface PDE problem using PhysicsNeMo Sym. The use of
variational method (weak formulation) also allows you to handle problems
with point source with ease and this is covered in this tutorial too.
In this tutorial you will learn:

#. How to solve a PDE in its variational form (continuous and
   discontinuous) in PhysicsNeMo Sym.

#. How to generate test functions and their derivative data on desired
   point sets.

#. How to use quadrature in the PhysicsNeMo Sym.

#. How to solve a problem with a point source (Dirac Delta function).

.. note::
   This tutorial assumes that you have completed tutorial
   :ref:`Introductory Example` on Lid Driven Cavity and have familiarized yourself
   with the basics of the PhysicsNeMo Sym APIs. Also, see Section :ref:`weak-solutions-pinn` from the
   Theory chapter for more details on weak solutions of PDEs.

   All the scripts referred in this tutorial can be found in
   ``examples/discontinuous_galerkin/``.

.. warning::

   The Python package `quadpy <https://github.com/nschloe/quadpy>`_ is required for these examples.
   Install using ``pip install quadpy`` (Also refer to :ref:`install_physicsnemo_docker`).

Problem Description
-------------------

This tutorial, demonstrates solving the Poisson equation with Dirichlet
boundary conditions. The problem represents an interface between two
domains. Let :math:`\Omega_1 = (0,0.5)\times(0,1)`,
:math:`\Omega_2 = (0.5,1)\times(0,1)`, :math:`\Omega=(0,1)^2`. The interface
is :math:`\Gamma=\overline{\Omega}_1\cap\overline{\Omega}_2`, and the
Dirichlet boundary is :math:`\Gamma_D=\partial\Omega`. The domain for the
problem can be visualized in the :numref:`fig-domain`. The problem
was originally defined in [#zang2020weak]_.

.. _fig-domain:

.. figure:: ../../images/user_guide/domain_combine.png
   :alt: Left: Domain of interface problem. Right: True Solution
   :name: fig:domain
   :align: center

   Left: Domain of interface problem. Right: True Solution

The PDEs for the problem are defined as

.. math::
    :label: ex1-example

    -\Delta u = f \quad \mbox{in}\quad \Omega,

.. math::
    :label: ex2-example

    u = g_D \quad \mbox{on} \quad \Gamma_D,

.. math::
    :label: ex3-example

    \left[\frac{\partial u}{\partial \mathbf{n}} \right] =g_I \quad \mbox{on} \quad\Gamma,

where :math:`f=-2`, :math:`g_I=2` and

.. math::

   g_D =
   \begin{cases}
   x^2 &   0\leq x\leq \frac{1}{2}\\
   (x-1)^2 &   \frac{1}{2}< x\leq 1
   \end{cases}
   .

The :math:`g_D` is the exact solution of
:eq:`ex1-example`-:eq:`ex3-example`.

The jump :math:`[\cdot]` on the interface :math:`\Gamma` is defined by

.. math:: \left[\frac{\partial u}{\partial \mathbf{n}}\right]=\nabla u_1\cdot\mathbf{n}_1+\nabla u_2\cdot\mathbf{n}_2,\label{var_ex-example}

where :math:`u_i` is the solution in :math:`\Omega_i` and the
:math:`\mathbf{n}_i` is the unit normal on :math:`\partial\Omega_i\cap\Gamma`.

As suggested in the original reference, this problem does not accept a
strong (classical) solution but only a unique weak solution
(:math:`g_D`) which is shown in :numref:`fig:domain`.

.. note::
   
    Please be advised that, in the original paper [#zang2020weak]_, the PDE is incorrect and 
    :eq:`ex1-example`-:eq:`ex3-example` 
    defines the corrected PDEs for the problem.

Variational Form
----------------

Since :eq:`ex3-example` suggests that the solution’s
derivative is broken at interface (:math:`\Gamma`) , you will have to do the
variational form on :math:`\Omega_1` and :math:`\Omega_2` separately.
Equations :eq:`var_cont-example` and
:eq:`var_discont-example` show the continuous and
discontinuous variational formulation for the problem above. For
brevity, only the final variational forms are given here. For the
detailed derivation of these formulations, see the
Theory Appendix :ref:`variational-appendix`.

Variational form for Continuous type formulation :

.. math::
    :label: var_cont-example

    \int_{\Omega}(\nabla u\cdot\nabla v - fv) dx - \int_{\Gamma} g_Iv ds - \int_{\Gamma_D} \frac{\partial u}{\partial \mathbf{n}}v ds = 0



Variational form for Discontinuous type formulation :

.. math::
    :label: var_discont-example

    \sum_{i=1}^2(\nabla u_i\cdot v_i - fv_i) dx - \sum_{i=1}^2\int_{\Gamma_D}\frac{\partial u_i}{\partial \mathbf{n}} v_i ds-\int_{\Gamma}(g_I\langle v \rangle+\langle \nabla u \rangle[\![ v ]\!]) ds =0

The following subsections show how to implement these
variational forms in the PhysicsNeMo Sym.

Continuous type formulation
---------------------------

This subsection shows how to implement the continuous type
variational form :eq:`var_cont-example` in PhysicsNeMo Sym.
The code for this example can be found in ``./dg/dg.py``.

First, import all the packages needed:


.. literalinclude:: ../../../examples/discontinuous_galerkin/dg/dg.py
   :language: python
   :lines: 15-44


Creating the Geometry
~~~~~~~~~~~~~~~~~~~~~

Using the interface in the middle of the domain, you can define
the geometry by left and right parts separately. This allows you to
capture the interface information by sampling on the boundary that is
common to the two halves.

.. literalinclude:: ../../../examples/discontinuous_galerkin/dg/dg.py
   :language: python
   :lines: 164-176

In this example, you will use the variational form in conjunction
with traditional PINNs. The PINNs’ loss is essentially a point-wise
residual, and the loss function performs well for a smooth solution.
Therefore, impose the traditional PINNs’ loss for areas away from
boundaries and interfaces.

Defining the Boundary conditions and Equations to solve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the geometry defined for the problem, you can define the constraints for the boundary conditions and PDEs.

The PDE will be taken care by variational form. However, there is no conflict to apply the classic form PDE constraints
with the present of variational form. The rule of thumb is, with classic form PDE constraints, the neural network converges
faster, but the computational graph is larger. The code segment below applies the classic form PDE constraint. This part is optional because of variational constraints.

.. literalinclude:: ../../../examples/discontinuous_galerkin/dg/dg.py
   :language: python
   :lines: 178-290

For variational constraints, in the ``run`` function, first collect the data needed to formulate the variational form. 
For interior points, there are two options.
The first option is quadrature rule. PhysicsNeMo Sym has the functionality to create the
quadrature rule on some basic geometries and meshes based on `quadpy <https://github.com/nschloe/quadpy>`_ package.
The quadrature rule has higher accuracy and efficiency, so use the quadrature rules when possible.
The other option is using random points. You can use quasi-random points to increase the accuracy of the integral
by setting ``quasirandom=True`` in ``sample_interior``.
For this examples, you can use ``cfg.quad`` in Hydra configure file to choose the option.

You can also use the radial basis test function. If so, use the additional data
for the center of radial basis functions (RBFs).


Creating the Validator
~~~~~~~~~~~~~~~~~~~~~~

Since the closed form solution is known, create a validator to compare the prediction and ground truth solution.

.. literalinclude:: ../../../examples/discontinuous_galerkin/dg/dg.py
   :language: python
   :lines: 292-309


Creating the Inferencer
~~~~~~~~~~~~~~~~~~~~~~~

To generate the solution at the desired domain, add an inferencer.

.. literalinclude:: ../../../examples/discontinuous_galerkin/dg/dg.py
   :language: python
   :lines: 311-319


Creating the Variational Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This subsection, shows how to form the variational loss. Use the data collected and registered
to the ``VariationalConstraint`` to form this loss.

First, choose what test function to use.

In PhysicsNeMo Sym, Legendre, 1st and 2nd kind of Chebyshev polynomials
and trigonometric functions are already implemented as the test
functions and can be selected directly. You can also define your own
test functions by providing its name, domain, and SymPy expression in
``meta_test_function`` class. In the ``Test_Function``, you will need to
provide a dictionary of the name and order of the test functions
(``name_ord_dict`` in the parameter list), the upper and lower bound of
your domain (``box`` in the parameter list), and what kind of
derivatives you will need (``diff_list`` in the parameter list). For
example, if :math:`v_{xxy}` is needed, you might add ``[1,1,2]`` in the
``diff_list``. There are shortcuts for ``diff_list``. If you need all the
components of gradient of test function, you might add ``'grad'`` in
``diff_list``, and if the Laplacian of the test function is needed, you
might add ``'Delta'``. The ``box`` parameter if left unspecified, is set
to the default values, i.e. for Legendre polynomials :math:`[-1, 1]^n`,
for trigonometric functions :math:`[0, 1]^n`, etc.

The definition of test function will be put in initializer of the ``DGLoss`` class.

.. literalinclude:: ../../../examples/discontinuous_galerkin/dg/dg.py
   :language: python
   :lines: 47-63

Then, it suffices to define the ``forward`` function of ``DGLoss``. In ``forward``, you need to form and return the variational
loss. According to :eq:`var_cont-example`, the variational loss has been formed by the following code:

.. literalinclude:: ../../../examples/discontinuous_galerkin/dg/dg.py
   :language: python
   :lines: 65-150

``list_invar`` includes all the inputs from the geometry while the ``list_outvar`` includes all requested outputs. The test
function ``v`` can be evaluated by method ``v.eval_test``. The parameters are: the name of function you want, and the coordinates
to evaluate the functions.

Now, all the resulting variables of test function, like ``v_interior``, are :math:`N` by
:math:`M` tensors, where :math:`N` is the number of points, and
:math:`M` is the number of the test functions.

To form the integration, you can use the ``tensor_int`` function in the
PhysicsNeMo Sym. This function has three parameters ``w``, ``v``, and ``u``. The
``w`` is the quadrature weight for the integration. For uniform random
points or quasi-random points, it is precisely the average area. The
``v`` is an :math:`N` by :math:`M` tensor, and ``u`` is a :math:`1` by
:math:`M` tensor. If ``u`` is provided, this function will return a
:math:`1` by :math:`M` tensor, and each entry is
:math:`\int_\Omega u v_i dx`, for :math:`i=1,\cdots, M`. If ``u`` is not
provided, it will return a :math:`1` by :math:`M` tensor, and each entry
is :math:`\int_\Omega v_i dx`, for :math:`i=1,\cdots, M`.

Results and Post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solving the problem using different settings.

First, solve the problem by Legendre and trigonometric test function without quadrature rule.
The results are shown in :numref:`fig-dg_pinn`.

.. _fig-dg_pinn:

.. figure:: ../../images/user_guide/dg_pinns.png
   :alt: Left: PhysicsNeMo Sym. Center: Analytical. Right: Difference.
   :name: fig:dg_pinn
   :align: center

   Left: PhysicsNeMo Sym. Center: Analytical. Right: Difference.

By using quadrature rule, the results are shown in :numref:`fig-dg_quad`.

.. _fig-dg_quad:

.. figure:: ../../images/user_guide/dg_quad.png
   :alt: Left: PhysicsNeMo Sym. Center: Analytical. Right: Difference.
   :name: fig:dg_quad
   :align: center

   Left: PhysicsNeMo Sym. Center: Analytical. Right: Difference.

By using quadrature rule and RBF test function, the results are shown in :numref:`fig-dg_rbf`.

.. _fig-dg_rbf:

.. figure:: ../../images/user_guide/dg_rbf.png
   :alt: Left: PhysicsNeMo Sym. Center: Analytical. Right: Difference.
   :name: fig:dg_rbf
   :align: center

   Left: PhysicsNeMo Sym. Center: Analytical. Right: Difference.


Point source and Dirac Delta function
-------------------------------------

Weak formulation enables solution of PDEs with distributions, e.g.,
Dirac Delta function. The Dirac Delta function :math:`\delta(x)` is
defined as

.. math:: \int_{\mathbb{R}}f(x)\delta(x) dx  = f(0),

for all continuous compactly supported functions :math:`f`.

This subsection solves the following problem:

.. math::
   :label: delta_diff

   \begin{aligned}
   -\Delta u &= \delta \quad \mbox{ in } \Omega\\
   u &= 0 \quad \text{ on } \partial\Omega\end{aligned}

where :math:`\Omega=(-0.5,0.5)^2` (:numref:`fig-point-source`). In
physics, this means there is a point source in the middle of the domain
with :math:`0` Lebesgue measure in :math:`\mathbb{R}^2`. The corresponding weak
formulation is

.. math::
    :label: delta_var

    \int_{\Omega}\nabla u\cdot \nabla v dx - \int_{\Gamma} \frac{\partial u}{\partial \mathbf{n}}v ds = v(0, 0)

The code of this example can be found in ``./point_source/point_source.py``.

.. _fig-point-source:

.. figure:: ../../images/user_guide/point-source-fig.png
   :alt: Domain for the point source problem.
   :name: fig:point-source
   :align: center
   :width: 40.0%

   Domain for the point source problem.

Creating the Geometry
~~~~~~~~~~~~~~~~~~~~~

Use both the weak and differential form to solve
:eq:`delta_diff` and :eq:`delta_var`. Since the solution has a sharp gradient
around the origin, which causes issues for traditional PINNs, weight this area lower using the lambda weighting functions. The
geometry can be defined by:

.. literalinclude:: ../../../examples/discontinuous_galerkin/point_source/point_source.py
   :language: python
   :lines: 111-113


Creating the Variational Loss and Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As shown in :eq:`delta_var`, the only difference to the
previous examples is the right hand side term is the value of :math:`v`
instead of an integral. You only need to change the ``fv`` in the
code. The whole code of the ``DGLoss`` is the following:

.. literalinclude:: ../../../examples/discontinuous_galerkin/point_source/point_source.py
   :language: python
   :lines: 114-171

Results and Post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The results for the problem are shown in :numref:`fig:dg_point_source`.

.. figure:: ../../images/user_guide/point_source.png
   :alt: PhysicsNeMo Sym prediction.
   :name: fig:dg_point_source
   :align: center
   :width: 80.0%

   PhysicsNeMo Sym prediction

Since the ground truth solution is unbounded at origin, it is not useful to compare it with the exact solution.


.. [#zang2020weak] Zang, Y., Bao, G., Ye, X. and Zhou, H., 2020. Weak adversarial networks for high-dimensional partial differential equations. Journal of Computational Physics, 411, p.109409.
