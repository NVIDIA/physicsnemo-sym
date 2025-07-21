Physics Informed Neural Networks (PINNs) in PhysicsNeMo Sym
===========================================================

.. _nn_methodology:

In this section we provide a brief introduction to Physics Informed Neural
Networks (PINNs), a technique used to find solutions of partial differential
equations using neural networks.

Background
----------

The solution to a partial differential equation (PDE) is a function describing
how quantities of interest vary throughout a domain. For physics problems, this
domain is often physical space and possibly time, so we can write this solution
function as :math:`u(x, t)`. This function, as-written, is a member of an
infinite-dimensional function space -- precisely, a Hilbert space. Because of
this, fully describing a single solution to a PDE possibly requires an infinite
amount of information -- a property that presents significant challenges for
numerical methods.

Therefore, all major numerical methods for solving PDEs are based on the idea of
approximating the solution to a PDE with a finite-dimensional function -- a
process called discretization. Different fundamental approaches to
discretization are what give rise to the various broad families of numerical
methods for solving PDEs. For example:

- **Finite Difference Methods** represent the solution as values at discrete
  points on a regular grid, from which derivatives can be approximated by using
  information from neighboring points.

- **Finite Volume Methods** represent the solution as values on cells of a mesh,
  and focus on identifying fluxes across the boundaries of these cells using
  theory around conservation laws.

- **Finite Element Methods** represent the solution as a patchwork of local
  basis functions (often polynomials) over elements (subdomains) of the domain.

- **Spectral Methods** represent the solution as a sum of global Fourier-like
  basis functions.

Physics Informed Neural Networks (PINNs)
----------------------------------------

In Physics Informed Neural Networks (PINNs), the solution function to the PDE is
itself represented by a neural network: the neural network maps an input
location in space and time to the value of the solution function at that
location. This neural network forms a finite-dimensional approximation to the
solution function, due to the finite number of parameters in the neural network.
In order to ensure that the neural network satisfies the PDE, we optimize those
parameters to minimize a loss function that penalizes (a) local errors in the
PDE governing equations, and (b) local errors in the boundary conditions.

If the network is able to minimize this loss function then it will in effect,
solve the given differential equation. As with traditional numerical methods,
the restriction to a finite-dimensional function space means that this loss term
will often not be driven exactly to zero -- this is analogous to discretization
error.

To illustrate this idea we will give an example of solving the following
problem,

.. math::
   :label: 1d_equation

       \mathbf{P} : \left\{\begin{matrix}
   \frac{\delta^2 u}{\delta x^2}(x) = f(x), \\ \\ u(0) = u(1) = 0,
   \end{matrix}\right.

We start by constructing a neural network :math:`u_{net}(x)`. The input to this
network is a single value :math:`x \in \mathbb{R}` and its output is also a
single value :math:`u_{net}(x) \in \mathbb{R}`. We suppose that this neural
network is infinitely differentiable, :math:`u_{net} \in C^{\infty}`. The
typical neural network used is a deep fully connected network where the
activation functions are infinitely differentiable.

Next we need to construct a loss function to train this neural network. We
easily encode the boundary conditions as a loss in the following way:

.. math:: L_{BC} = u_{net}(0)^2 + u_{net}(1)^2

For encoding the equations, we need to compute the derivatives of
:math:`u_{net}`. Using automatic differentiation we can do so and compute
:math:`\frac{\delta^2 u_{net}}{\delta x^2}(x)`. This allows us to write a loss
function of the form:

.. math::
   :label: sumation_loss

     L_{residual} = \frac{1}{N}\sum^{N}_{i=0} \left( \frac{\delta^2
     u_{net}}{\delta x^2}(x_i) - f(x_i) \right)^2

Where the :math:`x_i` 's are a batch of points sampled in the interior,
:math:`x_i \in (0, 1)`. Our total loss is then :math:`L = L_{BC} +
L_{residual}`. Optimizers such as Adam [#kingma2014adam]_ are used to train this
neural network. Given :math:`f(x)=1`, the true solution is
:math:`\frac{1}{2}(x-1)x`. Upon solving the problem, you can obtain good
agreement between the exact solution and the neural network solution as shown in
:numref:`fig-single-parabola`.

.. _fig-single-parabola:

.. figure:: /images/user_guide/single_parabola.png
   :alt: Neural Network Solver compared with analytical solution
   :width: 60.0%
   :align: center

   Neural Network Solver compared with analytical solution.


Using the PINNs in PhysicsNeMo Sym, we were able to solve complex problems with
intricate geometries and multiple physics. In order to achieve this we have
deviated and improved on the current state-of-the-art in several important ways.
In this section we will briefly cover some topics related to this.

Monte Carlo integration for loss formulation
--------------------------------------------

In literature, the losses are often defined as a summation similar to our above
equation :eq:`sumation_loss`, [#raissi2017physics]_. In PhysicsNeMo Sym, we take
a different approach and view the losses as integrals. You can instead write
:math:`L_{residual}` in the form,

.. math:: L_{residual} = \int^1_0 \left( \frac{\delta^2 u_{net}}{\delta x^2}(x) - f(x) \right)^2 dx

Now there is a question of how we approximate this integral. We can easily see
that if we use Monte Carlo integration we arrive at the same summation in
equation :eq:`sumation_loss`.

.. math:: \int^1_0 \left( \frac{\delta^2 u_{net}}{\delta x^2}(x) - f(x) \right)^2 dx \approx (\int^1_0 dx) \frac{1}{N} \sum^{N}_{i=0} \left( \frac{\delta^2 u_{net}}{\delta x^2}(x_i) - f(x_i) \right)^2 = \frac{1}{N} \sum^{N}_{i=0} \left( \frac{\delta^2 u_{net}}{\delta x^2}(x_i) - f(x_i) \right)^2

We note that, this arrives at the exact same summation because :math:`\int^1_0
dx = 1`. However, this will scale the loss proportional to the area. We view
this as a benefit because it keeps the loss per area consistent across domains.
We also note that this opens the door to more efficient integration techniques.
In several examples, in this user guide we sample with higher frequency in
certain areas of the domain to approximate the integral losses more efficiently.

Integral Equations
------------------

Many PDEs of interest have integral formulations. Take for example the
continuity equation for incompressible flow,

.. math:: \frac{\delta u}{\delta x} + \frac{\delta v}{\delta y} + \frac{\delta w}{\delta z} = 0

We can write this in integral form as the following,


.. math:: 
    
    \iint_{S} (n_xu + n_yv + n_zw) dS = 0

Where :math:`S` is any closed surface in the domain and :math:`n_x, n_y, n_z`
are the normals. We can construct a loss function using this integral form and
approximate it with Monte Carlo Integration in the following way,

.. math:: L_{IC} = \left(\iint_{S} (n_xu + n_yv + n_zw) dS \right)^2 \approx \left((\iint_{S} dS) \frac{1}{N} \sum^N_{i=0} (n^i_xu_i + n^i_yv_i + n^i_zw_i)\right)^2

For some problems we have found that integrating such losses significantly
speeds up convergence.

Parameterized Geometries
------------------------

One important advantage of a neural network solver over traditional numerical
methods is its ability to solve parameterized geometries [#sun2020surrogate]_.
To illustrate this concept we solve a parameterized version of equation
:eq:`1d_equation`. Suppose we want to know how the solution to this equation
changes as we move the position on the boundary condition :math:`u(l)=0`. We can
parameterize this position with a variable :math:`l \in [1,2]` and our equation
now has the form,

.. math::
   :label: 1d_equation2

       \mathbf{P} : \left\{\begin{matrix}
   \frac{\delta^2 u}{\delta x^2}(x) = f(x), \\ \\ u(0) = u(l) = 0,
   \end{matrix}\right.

To solve this parameterized problem we can have the neural network take
:math:`l` as input, :math:`u_{net}(x,l)`. The losses then take the form,

.. math:: L_{residual} = \int_1^2 \int_0^l \left( \frac{\delta^2 u_{net}}{\delta x^2}(x,l) - f(x) \right)^2 dx dl \approx \left(\int_1^2 \int^l_0 dxdl\right) \frac{1}{N} \sum^{N}_{i=0} \left(\frac{\delta^2 u_{net}}{\delta x^2}(x_i, l_i) - f(x_i)\right)^2

.. math:: L_{BC} = \int_1^2 (u_{net}(0,l))^2 + (u_{net}(l,l) dl \approx \left(\int_1^2 dl\right) \frac{1}{N} \sum^{N}_{i=0} (u_{net}(0, l_i))^2 + (u_{net}(l_i, l_i))^2

In :numref:`fig-every-parabola` we see the solution to the differential equation
for various :math:`l` values after optimizing the network on this loss. While
this example problem is overly simplistic, the ability to solve parameterized
geometries presents significant industrial value. Instead of performing a single
simulation we can solve multiple designs at the same time and for reduced
computational cost. Examples of this will be given later in the user guide.

.. _fig-every-parabola:

.. figure:: /images/user_guide/every_parabola.png
   :alt: PhysicsNeMo Sym solving parameterized differential equation problem.
   :width: 60.0%
   :align: center

   PhysicsNeMo Sym solving parameterized differential equation problem.

Inverse Problems
----------------

Another useful application of a neural network solver is solving inverse
problems. In an inverse problem, we start with a set of observations and then
use those observations to calculate the causal factors that produced them. To
illustrate how to solve inverse problems with a neural network solver, we give
the example of inverting out the source term :math:`f(x)` from equation
:eq:`1d_equation`. Suppose we are given the solution :math:`u_{true}(x)` at 100
random points between 0 and 1 and we want to determine the :math:`f(x)` that is
causing it. We can do this by making two neural networks :math:`u_{net}(x)` and
:math:`f_{net}(x)` to approximate both :math:`u(x)` and :math:`f(x)`. These
networks are then optimized to minimize the following losses;

.. math:: L_{residual} \approx \left(\int^1_0 dx\right) \frac{1}{N} \sum^{N}_{i=0} \left(\frac{\delta^2 u_{net}}{\delta x^2}(x_i, l_i) - f_{net}(x_i)\right)^2

.. math:: L_{data} = \frac{1}{100} \sum^{100}_{i=0} (u_{net}(x_i) - u_{true}(x_i))^2

Using the function :math:`u_{true}(x)=\frac{1}{48} (8 x (-1 + x^2) - (3 sin(4
\pi x))/\pi^2)` the solution for :math:`f(x)` is :math:`x + sin(4 \pi x)`. We
solve this problem and compare the results in :numref:`fig-inverse-parabola`,
:numref:`fig-inverse-parabola-2`

.. _fig-inverse-parabola:

.. figure:: /images/user_guide/inverse_parabola.png
   :alt: Comparison of true solution for :math:`f(x)` and the function approximated by the NN.
   :width: 60.0%
   :align: center

   Comparison of true solution for :math:`f(x)` and the function approximated by
   the NN.

.. _fig-inverse-parabola-2:

.. figure:: /images/user_guide/inverse_parabola_2.png
   :alt: Comparison of :math:`u_{net}(x)` and train points from :math:`u_{true}`.
   :width: 60.0%
   :align: center

   Comparison of :math:`u_{net}(x)` and train points from :math:`u_{true}`.

.. _weak-solutions-pinn:

Weak solution of PDEs using PINNs
---------------------------------

In previous discussions on PINNs, we aimed at solving the classical solution of
the PDEs. However, some physics have no classical (or strong) form but only a
variational (or weak) form [#braess2007finite]_. This requires handling the PDEs
in a different approach other than its original (classical) form, especially for
interface problems, concave domain, singular problems, etc. In PhysicsNeMo Sym, we
can solve the PDEs not only in its classical form, but also in it weak form.
Before describing the theory for weak solutions of PDEs using PINNs, let's start
by the definitions of classical, strong and weak solutions.

**Note:** The mathematical definitions of the different spaces that are used in
the subsequent sections like the :math:`L^p`, :math:`C^k`, :math:`W^{k,p}`,
:math:`H`, etc. can be found in the :ref:`appendix`. For general theory of the
partial differential equations and variational approach, we recommend
[#gilbarg2015elliptic]_, [#evans1997partial]_.

Classical solution, Strong solution, Weak solution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we introduce the classical solution, strong solution, and weak
solution for the Dirichlet problem. Let us consider the following Poisson's
equation.

.. math::
   :label: variational_problem

   \left\{\begin{matrix} \Delta u = f \quad \text{ in } \Omega \\ \\ u = 0 \quad
   \text{ on } \partial \Omega \end{matrix}\right.


**Definition (Classical Solution):**

Let :math:`f\in C(\overline{\Omega})` in :eq:`variational_problem`, then there
is a unique solution :math:`u\in C^2(\Omega)\cap C_0^1(\Omega)` for
:eq:`variational_problem`. We call this solution as the classical solution of
:eq:`variational_problem`.

**Definition (Strong Solution):**

Let :math:`f\in L^2(\Omega)` in :eq:`variational_problem`, then there is a
unique solution :math:`u\in H^2(\Omega)\cap H_0^1(\Omega)` for
:eq:`variational_problem`. We call this solution as the strong solution of
:eq:`variational_problem`.

From the definition of strong solution and Sobolev space, we can see that the
solution of :eq:`variational_problem` is actually the solution of the following
problem: Finding a :math:`u\in H^2(\Omega)\cap H_0^1(\Omega)`, such that

.. math:: 
    :label: strong

    \int_{\Omega}(\Delta u + f)v dx = 0\qquad \forall v \in C_0^\infty(\Omega)

By applying integration by parts and :math:`u = 0`, we get

.. math:: \int_{\Omega}\nabla u\cdot\nabla v dx = \int_{\Omega} fv dx

This leads us to the definition of weak solution as the following.

**Definition (Weak Solution):**

Let :math:`f\in L^2(\Omega)` in :eq:`variational_problem`, then there is a
unique solution :math:`u\in H_0^1(\Omega)` for the following problem: Finding a
:math:`u\in H_0^1(\Omega)` such that

.. math:: 
   :label: weak

    \int_{\Omega} \nabla u \cdot\nabla v dx = \int_{\Omega}fv dx\qquad \forall
    v\in H_0^1(\Omega).

We call this solution as the weak solution of :eq:`variational_problem`.

In simpler terms, the difference between these three types of solutions can be
summarized as below:


The essential difference among classical solution, strong solution and weak
solution is their regularity requirements. The classic solution is a solution
with :math:`2`\ nd order continuous derivatives. The strong solution has
:math:`2`\ nd order weak derivatives, while the weak solution has weak
:math:`1`\ st order weak derivatives. Obviously, classical solution has highest
regularity requirement and the weak solution has lowest one.

PINNs for obtaining weak solution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we will discuss how PINNs can be used to handle the PDEs in approaches
different than its original (classical) form. In [#kharazmi2019variational]_,
[#kharazmi2021hp]_, the authors introduced the VPINN and hp-VPINN methods to
solve PDEs' integral form. This integral form is based on :eq:`strong`. Hence,
it is solving a strong solution, which is better than a classical solution.

To further improve the performance of PINNs, we establish the method based on
eq:`weak` i.e., we are solving the weak solution. Let us assume we are solving
:eq:`variational_problem`. To seek the weak solution, we may focus on the
following variational form:

.. math::
   :label: eq3

       \int_{\Omega}\nabla u\cdot\nabla v dx = \int_{\Omega} fv dx

.. math::
   :label: eq4

       u = 0 \quad\mbox{ on } \partial \Omega 

For :eq:`eq4`, we may handle it as the traditional PINNs: take random points
:math:`\{\mathbf{x_i}^b\}_{i=1}^{N_b}\subset\partial\Omega`, then the boundary
loss is

.. math:: MSE_b = \frac{1}{N_b}\sum_{i=1}^{N_b}\left(u_{NN}(\mathbf{x_i}^b)-0\right)^2

For :eq:`eq3`, we choose a quadrature rule
:math:`\{\mathbf{x_i}^q,w_i^q\}_{i=1}^{N_q}`, such that for :math:`u:
\Omega\mapsto\mathbb{R}`, we have

.. math:: \int_{\Omega} u dx \approx \sum_{i=1}^{N_q}w_i^q u(\mathbf{x_i}^q).

For uniform random points or quasi Monte Carlo points, :math:`w_i^q=1/N_q` for
:math:`i=1,\cdots, N_q`. Additionally, we choose a set of test functions
:math:`v_j\in V_h`, :math:`j=1,\cdots, M` and then the loss of the integral is

.. math:: MSE_v = \left[\sum_{i=1}^{N_q}w_i^q\left(\nabla u(\mathbf{x_i}^q)\cdot\nabla v_j(\mathbf{x_i}^q)-f(\mathbf{x_i}^q)v_j(\mathbf{x_i}^q)\right)\right]^2.

Then, the total loss is

.. math:: MSE=\lambda_v*MSE_v+\lambda_b*MSE_b,

where the :math:`\lambda_v` and :math:`\lambda_b` are the corresponding weights
for each terms.

The value of these :math:`\lambda` weights is a hyperparameter that can be tuned
to balance the importance of the PDE and boundary conditions. Suitable values
for these weights depend heavily on the governing PDE, the application, and the
point sampling strategy, and should be tuned for each problem.

As we will see in the tutorial example :ref:`variational-example`, this scheme
is flexible and can handle the interface and Neumann boundary condition easily.
We can also use more than one neural networks on different domains by applying
the discontinuous Galerkin scheme.

.. rubric:: References

.. [#kingma2014adam] Kingma, Diederik P., and Jimmy partial. "Adam: A method for
    stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
.. [#raissi2017physics] Raissi, Maziar, Paris Perdikaris, and George Em
    Karniadakis. "Physics informed deep learning (part i): Data-driven solutions
    of nonlinear partial differential equations." arXiv preprint
    arXiv:1711.10561 (2017).
.. [#sun2020surrogate] Sun, Luning, et al. "Surrogate modeling for fluid flows
    based on physics-constrained deep learning without simulation data."
    Computer Methods in Applied Mechanics and Engineering 361 (2020): 112732.
.. [#braess2007finite] Braess, Dietrich. Finite elements: Theory, fast solvers,
    and applications in solid mechanics. Cambridge University Press, 2007.
.. [#gilbarg2015elliptic] Gilbarg, David, and Neil S. Trudinger. Elliptic
    partial differential equations of second order. Vol. 224. springer, 2015.
.. [#evans1997partial] Evans, Lawrence C. "Partial differential equations and
    Monge-Kantorovich mass transfer." Current developments in mathematics 1997.1
    (1997): 65-126.
.. [#kharazmi2019variational] Kharazmi, Ehsan, Zhongqiang Zhang, and George Em
    Karniadakis. "Variational physics-informed neural networks for solving
    partial differential equations." arXiv preprint arXiv:1912.00873 (2019).
.. [#kharazmi2021hp] Kharazmi, Ehsan, Zhongqiang Zhang, and George Em
    Karniadakis. "hp-VPINNs: Variational physics-informed neural networks with
    domain decomposition." Computer Methods in Applied Mechanics and Engineering
    374 (2021): 113547.
