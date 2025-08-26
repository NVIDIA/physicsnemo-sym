Miscellaneous Concepts
======================

.. _generalized_polynomial_chaos:

Generalized Polynomial Chaos
----------------------------

In this section, we briefly introduce the generalized Polynomial Chaos
(gPC) expansion, an efficient method for assessing how the uncertainties
in a model input manifest in its output. Later in Section
:ref:`limerock_gPC_surrogate`, we show how the
gPC can be used as a surrogate for shape parameterization in both of the
tessellated and constructive solid geometry modules.

The :math:`p-th` degree gPC expansion for a :math:`d`-dimensional input
:math:`\mathbf{\Xi}` takes the following form

.. math:: u_p (\mathbf{\Xi}) = \sum_{\mathbf{i} \in \Lambda_{p,d}} c_{\mathbf{i}} \psi_{\mathbf{i}}(\mathbf{\Xi}),

where :math:`\mathbf{i}` is a multi-index and :math:`\Lambda_{p,d}` is
the set of multi-indices defined as

.. math:: \Lambda_{p,d} = \{\mathbf{i} \in \mathbb{N}_0^d: ||\mathbf{i}||_1 \leq p\},

and the cardinality of :math:`\Lambda_{d,p}` is

.. math:: C = |\Lambda_{p,d}| = \frac{(p+d)!}{p!d!}.

:math:`\{c_{\mathbf{i}}\}_{\mathbf{i} \in \mathbb{N}_0^d}` is the set of
unknown coefficients of the expansion, and can be determined based on
the methods of stochastic Galerkin, stochastic collocation, or least
square [#xiu2010numerical]_. For the example presented
in this user guide, we will use the least square method. Although the
number of required samples to solve this least square problem is
:math:`C`, it is recommended to use at least :math:`2C` samples for a
reasonable accuracy [#xiu2010numerical]_.
:math:`\{\psi_{\mathbf{i}}\}_{\mathbf{i} \in \mathbb{N}_0^d}` is the set
of orthonormal basis functions that satisfy the following condition

.. math::
   :label: pce_orthonormal
      
	\int \psi_\mathbf{m}(\mathbf{\xi}) \psi_\mathbf{n}(\mathbf{\xi}) \rho(\mathbf{\xi}) d\mathbf{\xi} = \delta_{\mathbf{m} \mathbf{n}}, \,\,\, \mathbf{m}, \mathbf{n} \in \mathbb{N}_0^d.

For instance, for a uniformly and normally distributed :math:`\psi`, the
normalized Legendre and Hermite polynomials, respectively, satisfy the
orthonormality condition in Equation
:eq:`pce_orthonormal`.

.. _appendix:

Relative Function Spaces and Integral Identities
------------------------------------------------

In this section, we give some essential definitions of Relative function
spaces, Sobolev spaces and some important equalities. All the integral
in this section should be understood by Lebesgue integral.

:math:`L^p` space
"""""""""""""""""

Let :math:`\Omega \subset \mathbb{R}^d` is an open set. For any real number
:math:`1<p<\infty`, we define

.. math:: L^p(\Omega)=\left\{u:\Omega\mapsto\mathbb{R}\bigg|u\mbox{ is measurable on }\Omega,\ \int_{\Omega}|u|^pdx<\infty \right\},

endowed with the norm

.. math:: \|u\|_{L^p(\Omega)}=\left(\int_{\Omega}|u|^pdx\right)^{\frac{1}{p}}.

For :math:`p=\infty`, we have

.. math:: L^\infty(\Omega)=\left\{u:\Omega\mapsto\mathbb{R}\bigg|u\mbox{ is uniformly bounded in $ \Omega $ except a zero measure set} \right\},

endowed with the norm

.. math:: \|u\|_{L^\infty(\Omega)}=\sup_{\Omega}|u|.

Sometimes, for functions on unbounded domains, we consider their local integrability.
To this end, we define the following local :math:`L^p` space

.. math:: L^p_{loc}(\Omega)=\left\{u:\Omega\mapsto\mathbb{R}\bigg|u\in L^p(V),\ \forall V\subset\subset\Omega \right\},

where :math:`V\subset\subset\Omega` means :math:`V` is a compact subset of
:math:`\Omega`.

:math:`C^k` space
"""""""""""""""""

Let :math:`k\geq 0` be an integer, and :math:`\Omega\subset \mathbb{R}^d` is an
open set. The :math:`C^k(\Omega)` is the :math:`k`-th differentiable
function space given by

.. math:: C^k(\Omega)=\left\{u:\Omega\mapsto\mathbb{R}\bigg|u\mbox{ is $ k $-times continuously differentiable}\right\}.

Let :math:`\mathbf{\alpha}=(\alpha_1,\alpha_2,\cdots,\alpha_d)` be a
:math:`d`-fold multi-index of order
:math:`|\mathbf{\alpha}|=\alpha_1+\alpha_2+\cdots+\alpha_n=k`. The :math:`k`-th
order (classical) derivative of :math:`u` is denoted by

.. math:: D^{\mathbf{\alpha}}u=\frac{\partial^k}{\partial x_1^{\alpha_1}\partial x_2^{\alpha_2}\cdots\partial x_d^{\alpha_d}}u.

For the closure of :math:`\Omega`, denoted by :math:`\overline{\Omega}`, we
have

.. math:: C^k(\overline{\Omega})=\left\{u:\Omega\mapsto\mathbb{R}\bigg|D^{\mathbf{\alpha}}u\mbox{ is uniformly continuous on bounded subsets of $ \Omega $, }\forall|\mathbf{\alpha}|\leq k\right\}.

When :math:`k=0`, we also write :math:`C(\Omega)=C^0(\Omega)` and
:math:`C(\overline{\Omega})=C^0(\overline{\Omega})`.

We also define the infinitely differentiable function space

.. math:: C^\infty(\Omega)=\left\{u:\Omega\mapsto\mathbb{R}\bigg|u\mbox{ is infinitely differentiable} \right\}=\bigcap_{k=0}^\infty C^k(\Omega)

and

.. math:: C^\infty(\overline{\Omega})=\bigcap_{k=0}^\infty C^k(\overline{\Omega}).

We use :math:`C_0(\Omega)` and :math:`C_0^k(\Omega)` denote these functions in
:math:`C(\Omega)`, :math:`C^k(\Omega)` with compact support.

:math:`W^{k,p}` space
"""""""""""""""""""""

The weak derivative is given by the following definition
[#evans1997partial]_.

**Definition**

Suppose :math:`u,\ v\in L^1_{loc}(\Omega)` and :math:`\mathbf{\alpha}` is a
multi-index. We say that :math:`v` is the :math:`\mathbf{\alpha}^{th}` weak
derivative of :math:`u`, written

.. math:: D^{\mathbf{\alpha}}u=v,

provided

.. math:: \int_\Omega uD^{\mathbf{\alpha}}\phi dx=(-1)^{|\mathbf{\alpha}|}\int_{\Omega}v\phi dx

for all test functions :math:`\phi\in C_0^\infty(\Omega)`.

As a typical example, let :math:`u(x)=|x|` and :math:`\Omega=(-1,1)`. For
calculus we know that :math:`u` is not (classical) differentiable at
:math:`x=0`. However, it has weak derivative

.. math::

   (Du)(x)=
   \begin{cases}
   1   & x>0,\\
   -1  &   x\leq 0.
   \end{cases}

**Definition**

For an integer :math:`k\geq 0` and real number :math:`p\geq 1`, the
Sobolev space is defined by

.. math:: W^{k,p}(\Omega)=\left\{u\in L^p(\Omega)\bigg|D^{\mathbf{\alpha}}u\in L^p(\Omega),\ \forall|\mathbf{\alpha}|\leq k\right\},

endowed with the norm

.. math:: \|u\|_{k,p}=\left(\int_{\Omega}\sum_{|\mathbf{\alpha}|\leq k}|D^{\mathbf{\alpha}}u|^p\right)^{\frac{1}{p}}.

Obviously, when :math:`k=0`, we have :math:`W^{0,p}(\Omega)=L^p(\Omega)`.

When :math:`p=2`, :math:`W^{k,p}(\Omega)` is a Hilbert space. And it also
denoted by :math:`H^k(\Omega)=W^{k,2}(\Omega)`. The inner product in
:math:`H^k(\Omega)` is given by

.. math:: \langle u, v \rangle =\int_{\Omega}\sum_{|\mathbf{\alpha}|\leq k}D^{\mathbf{\alpha}}uD^{\mathbf{\alpha}}v dx

A crucial subset of :math:`W^{k,p}(\Omega)`, denoted by
:math:`W^{k,p}_0(\Omega)`, is

.. math:: W^{k,p}_0(\Omega)=\left\{u\in W^{k,p}(\Omega)\bigg| D^{\mathbf{\alpha}}u|_{\partial\Omega}=0,\ \forall|\mathbf{\alpha}|\leq k-1\right\}.

It is customary to write :math:`H^k_0(\Omega)=W_0^{k,2}(\Omega)`.

Integral Identities
----------------------------

In this subsection, we assume :math:`\Omega\subset \mathbb{R}^d` is a Lipschitz
bounded domain (see [#monk1992finite]_ for the
definition of Lipschitz domain).

**Theorem (Green's formulae)**

Let :math:`u,\ v\in C^2(\overline{\Omega})`. Then

#. 

   .. math:: \int_\Omega \Delta u dx =\int_{\partial\Omega} \frac{\partial u}{\partial n} dS

#. 

   .. math:: \int_\Omega \nabla u\cdot\nabla v dx = -\int_\Omega u\Delta v dx+\int_{\partial\Omega} u \frac{\partial v}{\partial n} dS

#. 

   .. math:: \int_{\Omega} u\Delta v-v\Delta u dx = \int_{\partial\Omega} u\frac{\partial v}{\partial n}-v\frac{\partial u}{\partial n} dS

For curl operator we have some similar identities. To begin with, we
define the 1D and 2D curl operators. For a scalar function
:math:`u(x_1,x_2)\in C^1(\overline{\Omega})`, we have

.. math:: \nabla \times u = \left(\frac{\partial u}{\partial x_2},-\frac{\partial u}{\partial x_1}\right)

For a 2D vector function
:math:`\mathbf{v}=(v_1(x_1,x_2),v_2(x_1,x_2))\in(C^1(\overline{\Omega}))^2`, we
have

.. math:: \nabla \times \mathbf{v} = \frac{\partial v_2}{\partial x_1}-\frac{\partial v_1}{\partial x_2}

Then we have the following integral identities for curl operators.

**Theorem**

#. Let :math:`\Omega\subset \mathbb{R}^3` and
   :math:`\mathbf{u},\ \mathbf{v}\in (C^1(\overline{\Omega}))^3`. Then

   .. math:: \int_{\Omega}\nabla \times \mathbf{u}\cdot\mathbf{v} dx = \int_{\Omega}\mathbf{u}\cdot\nabla \times \mathbf{v} dx+\int_{\partial\Omega}\mathbf{n} \times \mathbf{u} \cdot \mathbf{v} dS,

   where :math:`\mathbf{n}` is the unit outward normal.

#. Let :math:`\Omega\subset \mathbb{R}^2` and
   :math:`\mathbf{u}\in (C^1(\overline{\Omega}))^2` and
   :math:`v\in C^1(\overline{\Omega})`. Then

   .. math:: \int_{\Omega}\nabla\times\mathbf{u} v dx = \int_{\Omega}\mathbf{u}\cdot\nabla\times v dx+\int_{\partial\Omega}\mathbf{\tau}\cdot\mathbf{u} vdS,

   where :math:`\mathbf{\tau}` is the unit tangent to :math:`\partial \Omega`.

.. _variational-appendix:

Derivation of Variational Form Example
---------------------------------------

Let :math:`\Omega_1 = (0,0.5)\times(0,1)`,
:math:`\Omega_2 = (0.5,1)\times(0,1)`, :math:`\Omega=(0,1)^2`. The interface
is :math:`\Gamma=\overline{\Omega}_1\cap\overline{\Omega}_2`, and the
Dirichlet boundary is :math:`\Gamma_D=\partial\Omega`. The domain for the
problem can be visualized in :numref:`fig-domain-appendix`.
The problem was originally defined in [#zang2020weak]_.

.. _fig-domain-appendix:

.. figure:: ../../images/user_guide/domain_combine.png
   :alt: Left: Domain of interface problem. Right: True Solution
   :align: center
   :width: 60.0%

   Left: Domain of interface problem. Right: True Solution

The PDEs for the problem are defined as


.. math::
   :label: var-prob

   \begin{aligned}
   -u &= f \quad \text{ in } \Omega_1 \cup \Omega_2\\
   u &= g_D \quad \text{ on } \Gamma_D\\
   \frac{\partial u}{\partial \textbf{n}} &=g_I \quad \text{ on } \Gamma\end{aligned}

where :math:`f=-2`, :math:`g_I=2` and

.. math::

   g_D =
   \begin{cases}
   x^2 &   0\leq x\leq \frac{1}{2}\\
   (x-1)^2 &   \frac{1}{2}< x\leq 1
   \end{cases}
   .

The :math:`g_D` is the exact solution of
:eq:`var-prob`.

The jump :math:`[\cdot]` on the interface :math:`\Gamma` is defined by

.. math:: 
   :label: var_ex

    \left[\frac{\partial u}{\partial \mathbf{n}}\right]=\nabla u_1\cdot\mathbf{n}_1+\nabla u_2\cdot\mathbf{n}_2,\label{var_ex}

where :math:`u_i` is the solution in :math:`\Omega_i` and the
:math:`\mathbf{n}_i` is the unit normal on :math:`\partial\Omega_i\cap\Gamma`.

As suggested in the original reference, this problem does not accept a
strong (classical) solution but only a unique weak solution
(:math:`g_D`) which is shown in :numref:`fig-domain-appendix`.

**Note:** It is noted that in the original paper
[#zang2020weak]_, the PDE is incorrect and
:eq:`var-prob` defines the corrected PDEs for the
problem.

We now construct the variational form of :eq:`var-prob`. This is the first step to obtain
its weak solution. Since the equation suggests that the solution’s
derivative is broken at interface (:math:`\Gamma`), we have to do the
variational form on :math:`\Omega_1` and :math:`\Omega_2` separately.
Specifically, let :math:`v_i` be a suitable test function on
:math:`\Omega_i`, and by integration by parts, we have for :math:`i=1,2`,

.. math:: \int_{\Omega_i}(\nabla u\cdot\nabla v_i-fv_i) dx - \int_{\partial\Omega_i}\frac{\partial u }{\partial \mathbf{n}}v_i ds = 0.

If we are using one neural network and a test function defined on whole
:math:`\Omega`, then by adding these two equalities, we have

.. math:: \int_{\Omega}(\nabla u\cdot\nabla v - fv) dx - \int_{\partial} g_Iv ds - \int_{\Gamma_D} \frac{\partial u}{\partial \mathbf{n}}v ds = 0\label{var_cont}

If we are using two neural networks, and the test functions are
different on :math:`\Omega_1` and :math:`\Omega_2`, then we may use the
discontinuous Galerkin formulation
[#cockburn2012discontinuous]_. To this end, we first
define the jump and average of scalar and vector functions. Consider the
two adjacent elements as shown in :numref:`fig-element`.
:math:`\mathbf{n}^+` and :math:`\mathbf{n}^-`\ and unit normals for :math:`T^+`,
:math:`T^-` on :math:`F=T^+\cap T^-`, respectively. As we can observe,
we have :math:`\mathbf{n}^+=-\mathbf{n}^-`.

Let :math:`u^+` and :math:`u^-` be two scalar functions on :math:`T^+`
and :math:`T^-`, and :math:`\mathbf{v}^+` and :math:`\mathbf{v}^-` are two vector
fields on :math:`T^+` and :math:`T^-`, respectively. The jump and the
average on :math:`F` is defined by

.. math::

   \begin{aligned}
   \langle u \rangle = \frac{1}{2}(u^++u^-)    &&  \langle \mathbf{v} \rangle = \frac{1}{2}(\mathbf{v}^++\mathbf{v}^-)\\
    [\![ u ]\!] = u^+\mathbf{n}^++u^-\mathbf{n}^-  && [\![ \mathbf{v} ]\!] = \mathbf{v} ^+\cdot\mathbf{n}^++\mathbf{v} ^-\cdot\mathbf{n}^-\end{aligned}

.. _fig-element:

.. figure:: ../../images/user_guide/element_new.png
   :alt: Adjacent Elements.
   :width: 60.0%   
   :align: center
	
   Adjacent Elements.

**Lemma**

On :math:`F` of :numref:`fig-element`, we have

.. math:: [\![ u\mathbf{v} ]\!] = [\![ u ]\!] \langle \mathbf{v} \rangle + [\![ \mathbf{v} ]\!] \langle u \rangle.

By using the above lemma, we have the following equality, which is an
essential tool for discontinuous formulation.

**Theorem**

Suppose :math:`\Omega` has been partitioned into a mesh. Let
:math:`\mathcal{T}` be the set of all elements of the mesh, :math:`\mathcal{F}_I`
be the set of all interior facets of the mesh, and :math:`\mathcal{F}_E` be
the set of all exterior (boundary) facets of the mesh. Then we have

.. math:: 
   :label: var_eqn

    \sum_{T\in\mathcal{T}}\int_{\partial T}\frac{\partial u}{\partial \mathbf{n}} v ds = \sum_{e\in\mathcal{F}_I}\int_e \left([\![ \nabla u ]\!] \langle v \rangle + \langle \nabla u \rangle [\![  v ]\!] \right)ds+\sum_{e\in\mathcal{F}_E}\int_e \frac{\partial u}{\partial \mathbf{n}} v ds\label{var_eqn}

Using :eq:`var_ex` and :eq:`var_eqn`, we have
the following variational form

.. math:: 
   :label: var_discont

    \sum_{i=1}^2(\nabla u_i\cdot v_i - fv_i) dx - \sum_{i=1}^2\int_{\Gamma_D}\frac{\partial u_i}{\partial \mathbf{n}} v_i ds-\int_{\partial}(g_I\langle v \rangle+\langle \nabla u \rangle [\![ v ]\!] ds =0\label{var_discont}

Details on how to use these forms can be found in tutorial
:ref:`variational-example`.

.. rubric:: References

.. [#evans1997partial] Evans, Lawrence C. "Partial differential equations and Monge-Kantorovich mass transfer." Current developments in mathematics 1997.1 (1997): 65-126.
.. [#xiu2010numerical] Xiu, Dongbin. Numerical methods for stochastic computations. Princeton university press, 2010.
.. [#monk1992finite] Monk, Peter. "A finite element method for approximating the time-harmonic Maxwell equations." Numerische mathematik 63.1 (1992): 243-261.
.. [#zang2020weak] Zang, Yaohua, et al. "Weak adversarial networks for high-dimensional partial differential equations." Journal of Computational Physics 411 (2020): 109409.
.. [#cockburn2012discontinuous] Cockburn, Bernardo, George E. Karniadakis, and Chi-Wang Shu, eds. Discontinuous Galerkin methods: theory, computation and applications. Vol. 11. Springer Science & Business Media, 2012.
