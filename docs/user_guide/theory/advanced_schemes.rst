.. _advanced_schemes:

Advanced Schemes and Tools
==========================

.. _adaptive_activations:

Adaptive Activation Functions
------------------------------

In training of neural networks, in addition to leaning the linear
transformations, one can also learn the nonlinear transformations to
potentially improve the convergence as well as the accuracy of the model
by using the global adaptive activation functions proposed by
[#jagtap2020adaptive]_. Global adaptive activations
consist of a single trainable parameter that is multiplied by the input
to the activations in order to modify the slope of activations.
Therefore, a nonlinear transformation at layer :math:`\ell` will take
the following form

.. math:: \mathcal{N}^\ell \left(H^{\ell-1}; \theta, a \right) = \sigma\left(a \mathcal{L}^{\ell} \left(H^{\ell-1}\right) \right),

where :math:`\mathcal{N}^\ell` is the nonlinear transformation at layer
:math:`\ell`, :math:`H^{\ell-1}` is the output of the hidden layer
:math:`\ell-1`, :math:`\theta` is the set of model weights and biases,
:math:`a` is the global adaptive activation parameter, :math:`\sigma` is
the activation function, and :math:`\mathcal{L}^{\ell}` is the linear
transformation at layer :math:`\ell`. Similar to the network weights and
biases, the global adaptive activation parameter :math:`a` is also a
trainable parameter, and these trainable parameters are optimized by

.. math:: \theta^*, a^* = \underset{{\theta,a}}{\operatorname{argmin}} L(\theta,a).


.. _stan_activation:

Self-scalable tanh (Stan) activation function
---------------------------------------------

As a variant of adaptive activation functions, [#gnanasambandam2022self]_ proposed self-scalable tanh (Stan)
activation function, which allows normal flow of gradients to compute the required derivatives and also enable systematic
scaling of the input-output mapping. Specifically, for :math:`i` th neuron in
:math:`k` th layer ( :math:`k = 1, 2, \dots , D − 1, i = 1, 2, \dots, N_k` ), Stan is defined as

.. math:: \sigma_k^i(x) = \text{tanh}(x) + \beta_k^i x \cdot \text{tanh}(x)

where :math:`\beta_k^i` is the neuron-wise trainable parameter initialized by 1.

An example for Stan activation is provided by ``examples/helmholtz/helmholtz_stan.py``.
As shown in :numref:`fig-helmholtz_stan`, one can see that Stan activation
yields faster convergence and better validation accuracy.


.. _fig-helmholtz_stan:

.. figure:: /images/user_guide/helmholtz_stan.png
   :alt: A comparison of the validation error results between default SiLU (red) and Stan activations (blue) for the Helmholtz example.
   :name: fig:helmholtz_stan
   :width: 50.0%
   :align: center

   A comparison of the validation error results between default SiLU and Stan activations for the Helmholtz example.


.. _sobolev_training:

Sobolev (Gradient Enhanced) Training
--------------------------------------
Sobolev or gradient-enhanced training of physics-informed neural networks, proposed in
Son et al. [#son2021sobolev]_ and Yu et al. [#yu2021gradient]_, leverages the derivative information of the PDE
residuals in the training of a neural network solver. Specifically, in standard training of the neural network solvers,
we enforce a proper norm of the PDE residual to be zero, while in Sobolev or gradient-enhanced training, one can
take the first or even higher-order derivatives of the PDE residuals w.r.t. the spatial inputs and set a proper norm
of those residual
derivatives to zero as well. It has been reported in the reference papers [#son2021sobolev]_ [#yu2021gradient]_ that
Sobolev or gradient-enhanced training can potentially give better training accuracies when compared to the standard
training of a neural network solver. However, care must be taken
when choosing the relative weights of the additional loss terms with respect to the standard PDE residual and
boundary condition loss terms or otherwise, Sobolev or gradient-enhanced training may even adversely affect the
training convergence and accuracy of a neural network solver. Additionally, Sobolev or gradient-enhanced training
increases the training time as the differentiation order will be increased and thus extra backpropagation
will be required. An example for Sobolev or gradient-enhanced training for navier-stokes equations
can be found at ``examples/annular_ring/annular_ring_gradient_enhanced/annular_ring_gradient_enhanced.py``


Importance Sampling
-------------------

Suppose our problem is to find the optimal parameters
:math:`\mathbf{\theta^*}` such that the Monte Carlo approximation of the
integral loss is minimized

.. math::

   \begin{aligned}
   \begin{split}
   \mathbf{\theta^*} &= \underset{{ \mathbf{\theta} }}{\operatorname{argmin}} \ \mathbb{E}_f \left[ \ell (\mathbf{\theta}) \right] \\
   & \approx \underset{{ \mathbf{\theta} }}{\operatorname{argmin}} \ \frac{1}{N} \sum_{i=1}^{N} \ell (\mathbf{\theta}; \mathbf{x_i} ),  \quad \mathbf{x_i} \sim {f}(\mathbf{x}),
   \label{IM_integral}
   \end{split}\end{aligned}

where :math:`f` is a uniform probability density function. In importance
sampling, the sampling points are drawn from an alternative sampling
distribution :math:`q` such that the estimation variance of the integral
loss is reduced, that is

.. math::

   \label{IM_unbiased}
   \mathbf{\theta^*}  \approx \underset{{ \mathbf{\theta} }}{\operatorname{argmin}} \ \frac{1}{N} \sum_{i=1}^{N} \frac{f(\mathbf{x_i})}{q(\mathbf{x_i})} \ell (\mathbf{\theta}; \mathbf{x_i} ),  \quad \mathbf{x_i} \sim q(\mathbf{x}).

PhysicsNeMo Sym offers point cloud importance sampling for improved convergence
and accuracy, as originally proposed in
[#nabian2021efficient]_. In this scheme, the training
points are updated adaptively based on a sampling measure :math:`q` for
a more accurate unbiased approximation of the loss, compared to uniform
sampling. Details on the importance sampling implementation in PhysicsNeMo Sym
are presented in ``examples/ldc/ldc_2d_importance_sampling.py`` script. 
:numref:`fig-annular-ring-importance-sampling` shows a comparison between
the uniform and importance sampling validation error results for the
annular ring example, showing better accuracy when importance sampling
is used. Here in this example, the training points are sampled according
to a distribution proportional to the 2-norm of the velocity
derivatives. The sampling probability computed at iteration 100K is also
shown in :numref:`fig-annular-ring-sample-prob`.

.. _fig-annular-ring-importance-sampling:

.. figure:: /images/user_guide/annular_ring_importance_sampling.PNG
   :alt: A comparison between the uniform and importance sampling validation error results for the annular ring example.
   :name: fig:annular_ring_importance_sampling
   :width: 80.0%
   :align: center

   A comparison between the uniform and importance sampling validation
   error results for the annular ring example.

.. _fig-annular-ring-sample-prob:

.. figure:: /images/user_guide/annular_ring_sample_prob.png
   :alt: A visualization of the training point sampling probability at iteration 100K for the annular ring example.
   :name: fig:annular_ring_sample_prob
   :width: 50.0%
   :align: center

   A visualization of the training point sampling probability at
   iteration 100K for the annular ring example.

.. _halton:

Quasi-Random Sampling
------------------------

Training points in PhysicsNeMo Sym are generated according to a uniform
distribution by default. An alternative to uniform sampling is the
quasi-random sampling, which provides the means to generate training
points with a low level of discrepancy across the domain. Among the
popular low discrepancy sequences are the Halton sequences
[#halton1960efficiency]_, the Sobol sequences, and the
Hammersley sets, out of which the Halton sequences are adopted in
PhysicsNeMo Sym. A snapshot of a batch of training points generated using uniform
sampling and Halton sequences for the annular ring example is shown in
:numref:`fig-halton-ring`. Halton
sequences for sample generation can be enabled by setting ``quasirandom=True`` during the 
constraint definition.  
A case study on the use of Halton
sequences to solve a conjugate heat transfer example is also presented
in tutorial :ref:`fpga`.

.. _fig-halton-ring:

.. figure:: /images/user_guide/halton_ring.png
   :alt: A snapshot of a batch of training points generated using uniform sampling (top) and Halton sequences (bottom) for the annular ring example.
   :name: fig:halton_ring
   :width: 50.0%
   :align: center

   A snapshot of a batch of training points generated using uniform
   sampling (top) and Halton sequences (bottom) for the annular ring
   example.


.. _exact_bc:

Exact Boundary Condition Imposition
-----------------------------------

The standard neural network solvers impose boundary conditions in a soft form,
by incorporating boundary conditions as constraints in form of additional loss terms in the loss function.
In this form, the boundary conditions are not exactly satisfied. The work [#sukumar2022exact]_
introduced a new approach to exactly impose the boundary conditions in neural network solvers.
For this, they introduced a geometry aware solution ansatz for the neural network solver that consists of
an Approximate Distance Function (ADF) :math:`\phi (\mathbf{x})` to the boundaries of the domain
using the theory of R-functions. First, we will look into how this ADF is computed, and next, we will discuss
the formation of the solution ansatz based on the type of the boundary conditions. [#mnabian]_

Let :math:`D \subset \mathbb{R}^d` denote the computational domain with boundary :math:`\partial D`.
The exact distance is the shortest distance between any point :math:`\mathbf{x} \in  \mathbb{R}^d`
to the domain boundaries :math:`\partial D`, and therefore, is zero on :math:`\partial D`.
The exact distance function is not second or higher-order differentiable, and thus, one can use
the ADF function :math:`\phi (\mathbf{x})` instead.

The exact boundary condition imposition in PhysicsNeMo Sym is currently limited to 2D geometries only.
Let :math:`\partial D \in \mathbb{R}^2` be a boundary composed of :math:`n` line segments
and curves :math:`D_i`, and :math:`\phi_i` denote the ADF to each curve or line segment
such that :math:`\phi_1 \cup \phi_2 \cup ... \cup \phi_n =\phi`. The properties of an ADF function are as follows:
(1) For any point :math:`\mathbf{x}` on :math:`\partial D`, :math:`\phi(x)=0`, and
(2) :math:`\phi(x)` is normalized to the :math:`m`-th order,
i.e., its derivative w.r.t the unit inward normal vector is one and second to :math:`m`-th order derivatives
are zero for all the points on :math:`\partial D`.

The elementary properties of R-functions, including R-disjunction (union), R-conjunction (intersection),
and R-negation, can be used for constructing a composite ADF, :math:`\phi (\mathbf{x})`, to the
boundary :math:`\partial D` , when ADFs  :math:`\phi_i(\mathbf{x})`, to the partitions of
:math:`\partial D` are known. Once the ADFs,
:math:`\phi_i(\mathbf{x})` to all the partitions of :math:`\partial D` are calculated, we can calculate the ADF
to :math:`\partial D` using the R-equivalence operation. When :math:`\partial D` is composed of :math:`n` pieces,
:math:`\partial D_i`, then the ADF :math:`\phi` that is normalized up to order :math:`m` is given by

.. math:: \phi(\phi_1,...,\phi_n):=\phi_1~...~\phi_n=\frac{1}{\sqrt[m]{\frac{1}{(\phi_1)^m}+\frac{1}{(\phi_2)^m}+...+\frac{1}{(\phi_n)^m}}}.

Next, we will see how the individual ADFs :math:`\phi_i` for line segments and arcs are calculated.
For more details, please refer to the reference paper [#sukumar2022exact]_. The ADF for a infinite line passing
through two pints :math:`\mathbf{x}_1 \equiv (x_1,y_1)` and :math:`\mathbf{x}_2 \equiv (x_2,y_2)`
is calculated as

.. math:: \phi_l(\mathbf{x}; \mathbf{x}_1, \mathbf{x}_2) = \frac{(x-x_1)(y_2-y_1)-(y-y_1)(y_2-y_1)}{L},

where :math:`L` is the distance between the two points. Similarly ADF for a circle of radius :math:`R`
and center located at :math:`\mathbf{x}_c \equiv (x_c, y_c)` is given by

.. math:: \phi_c(\mathbf{x}; R, \mathbf{x}_c) = \frac{R^2-(\mathbf{x}-\mathbf{x}_c).(\mathbf{x}-\mathbf{x}_c)}{2R}.

In order to calculate the ADF for line segments and arcs, one has to use trimming functions [#sukumar2022exact]_.
Let us consider a line segment of length :math:`L` with end points :math:`\mathbf{x}_1 \equiv (x_1,y_1)`
and :math:`\mathbf{x}_2 \equiv (x_2,y_2)`, midpoint :math:`\mathbf{x}_c=(\frac{x_1+x_2}{2},\frac{y_1+y_2}{2})`
and length :math:`L = ||\mathbf{x}_2-\mathbf{x}_1||`. Then ADF for the line segment
:math:`\phi(\mathbf{x})` can be calculated as follows.

.. math:: f = \phi_l(\mathbf{x}, \mathbf{x}_1, \mathbf{x}_2),
.. math:: t = \phi_c(\mathbf{x}; R=\frac{L}{2}, \mathbf{x}_c=\frac{\mathbf{x}_1 + \mathbf{x}_2}{2}),
.. math:: \Phi = \sqrt{t^2 + f^4},
.. math:: \phi(\mathbf{x}) = \sqrt{f^2+(\frac{\Phi - t}{2})^2}.

Note that here, :math:`f` is the ADF for an infinite line and :math:`t` is the trimming function which is
the ADF for a circle. In other words, the ADF for a line segment is obtained by trimming an infinite line
by a circle. Similarly, one can obtain the ADF for an arc by using the above equations and by setting :math:`f`
to the circle ADF and :math:`t` to the ADF for an infinite line segment.

Now that we understand how to form the ADFs for line segments and arcs, let us discuss how we can form the
solution ansatz using ADFs such that the boundary conditions are exactly satisfied. For Dirichlet boundary condition,
if :math:`u=g` is prescribed on :math:`\partial D`, then the solution ansatz is given by

.. math:: u_{sol} = g + \phi u_{net},

where  :math:`u_{sol}` is the approximate solution, and :math:`u_{net}` is the neural network output.
To see how the solution ansatz if formed for Neumann, Robin, and mixed boundary conditions, please refer to
the reference paper [#sukumar2022exact]_.

When different inhomogeneous essential boundary conditions are imposed on distinct subsets of :math:`\partial D`,
we can use transfinite interpolation to calculate the :math:`g` function, which represents
the boundary condition function for the entire boundary :math:`\partial D`. The transfinite interpolation
function can be written as

.. math:: g(\mathbf{x}) =  \sum_{i=1}^{M} w_i(\mathbf{x})g_i(\mathbf{x}),
.. math::  w_i(\mathbf{x}) = \frac{\phi_i^{-\mu_i}}{\sum_{j=1}^{M}\phi_j^{-\mu_j}} = \frac{\prod_{j=1;j \neq i}^{M} \phi_j^{-\mu_j}}{\sum_{k=1}^{M}\prod_{j=1;j \neq k}^{M} \phi_j^{-\mu_j} + \epsilon},

where weights :math:`w_i` add up to one, and interpolates :math:`g_i` on the set :math:`\partial D_i`.
:math:`\mu_i \geq 1` is a constant controlling the nature of interpolation. :math:`\epsilon` is a small number
to prevent division by zero. This boundary value function,
:math:`g(\mathbf{x})`, can be used in the solution ansatz for Dirichlet boundary conditions
to calculate the final solution with the exact imposition of BC.

The exact imposition of boundary conditions as proposed in the reference paper [#sukumar2022exact]_, however,
has certain challenges especially when solving PDEs consisting of second or higher-order derivatives.
Approximate distance functions constructed using the theory of R function are not normalized at the
joining points of lines and arcs, and therefore, the second and higher-order derivatives are not defined
at these points, and can take extremely large values close to those points which can affect the convergence behavior 
of the neural network. The solution represented in the reference paper [#sukumar2022exact]_ is not to sample the
collocation points in close proximity of these points. We found, however, that this can adversely affect the
convergence and final accuracy of the solution. As an alternative. we propose to use the first order formulation
of the PDEs by change of variables. By treating the first order derivatives of the quantities of interest
as new variables, we can rewrite the second order PDEs as a series of first order PDEs with additional compatibility
equations that appear as additional terms in the loss function. For instance, let us consider the Helmholtz 
equation which takes the following form

.. math:: k^2 u + \frac{\partial ^2 u}{\partial x ^2} + \frac{\partial ^2 u}{\partial y ^2} + \frac{\partial ^2 u}{\partial z ^2} = f,

where :math:`k` is the wave number and :math:`f` is the source term. One can define new variables
:math:`u_x`, :math:`u_y`, :math:`u_z`, that represent, respectively, derivatives of the solution 
with respect to :math:`x`, :math:`y`, and :math:`z` coordinates, and rewrite the Helmholtz equation as a 
set of first-order equations in the following form:

.. math:: k^2 u + \frac{\partial u_x}{\partial x } + \frac{\partial u_y}{\partial y} + \frac{\partial u_z}{\partial z} = f,
.. math:: u_x = \frac{\partial u}{\partial x },
.. math:: u_y = \frac{\partial u}{\partial y },
.. math:: u_z = \frac{\partial u}{\partial z }.

Using this form, the output of the neural network will now include :math:`u_x`, :math:`u_y`, :math:`u_z`
in addition to :math:`u`, but this in effect reduces the order of differentiation by one. As a couple of examples,
first-order implementation
of the Helmholtz and Navier-Stokes equations are available at ``examples/helmholtz/pdes/helmholtz_first_order.py``
and ``examples/annular_ring/annular_ring_hardBC/pdes/navier_stokes_first_order.py``, respectively.

An advantage of using the first-order formulation of PDEs is the potential speed-up in training iterations 
as extra backpropagations for computing the second-order derivatives are not performed anymore. 
Additionally, this formulation enables the use of Automatic Mixed Precision (AMP), which is currently
not suitable to be used for problems with second and higher-order derivatives. Use of AMP can further
accelerate the training.

The figure below shows a comparison of interior validation accuracy between a baseline model (soft BC
imposition and second-order PDE) and a model trained with hard BC imposition and first-order
PDEs. It is evident that the hard BC approach reduces the validation accuracy by about
one order of magnitude compared to the baseline model. Additionally, the boundary validation error for 
the model trained with hard BC imposition is exactly zero unlike the baseline model. These examples
are available at ``examples/helmholtz``.

.. _fig-helmholtz_hardBC:

.. figure:: /images/user_guide/helmholtz_hardBC.png
   :alt: Interior validation accuracy for models trained with soft BC (orange) and hard BC (blue) imposition for the Helmholtz example.
   :width: 45.0%
   :align: center

   Interior validation accuracy for models trained with soft BC (orange) and hard BC (blue) imposition for the Helmholtz example.

Using AMP, training of the model with exact BC imposition is 25% faster compared to the training
of the baseline model.

Another example for solving the Navier-Stokes equations in the first-order form and with exact BC
imposition can be found in ``examples/annular_ring/annular_ring_hardBC``. The boundary conditions
in this example consist of the following: Prescribed parabolic inlet velocity on the left wall,
zero pressure on the right wall, and no-slip BC on the top/bottom walls and the inner circle.
The figure below shows the solution for the annular ring example with hard BC imposition.

.. _fig-annular_ring_hardBC:

.. figure:: /images/user_guide/annular_ring_hardBC.png
   :alt: Solution for the annular ring example obtained using hard BC imposition.
   :width: 65.0%
   :align: center

   Solution for the the annular ring example obtained using hard BC imposition.



.. _causal_training:

Causal training
---------------

Suppose that we have a time-dependent system of PDEs taking the following general form

.. math::
   :label: pde

   u_t + \mathcal{N}[u] = 0, \quad t \in [0, T], x \in \Omega ,

where :math:`u` describes the unknown latent solution that is governed
by the PDE system and :math:`\mathcal{N}` is a possibly nonlinear differential operator.
As demonstrated in [#wang2022respecting]_,  continuous-time PINNs models can violate temporal causality, and hence are
susceptible to converge towards erroneous solutions for transient problems.
*Causal training* [#wang2022respecting]_ aims to address this fundamental limitation and a key source of error
by reformulating the PDE residual loss to account explicitly for physical
causality during model training. To introduce it, we split the time domain :math:`[0, T]`
into :math:`N` chunks :math:`\{ [t_i, t_{i+1}] \}_{i=0}^{N-1}` and define the PDE residual loss over the :math:`i`-th chunk

.. math::

        \mathcal{L}_i(\mathbf{\theta}) = \sum_j | \frac{\partial u_{\mathbf{\theta}}}{\partial t}(t_j, x_j)
                            + \mathcal{N}[u](t_j, x_j) |^2

with :math:`\{t_j, x_j\} \subset [t_{i-1}, t_i] \times \Omega`.

Then the total causal loss is given by

.. math::

   \mathcal{L}_r(\mathbf{\theta}) = \sum_{i=1}^N w_i \mathcal{L}_i(\mathbf{\theta}).

where

.. math::

        w_i = \exp(-\epsilon \sum_{k=1}^{i-1} \mathcal{L}_i(\mathbf{\theta}), \quad \text{for} i=2,3, \dots, N.

Note that :math:`w_i` is inversely exponentially proportional to the magnitude of the cumulative
residual loss from the previous chunks.  As a consequence, :math:`\mathcal{L}_i(\mathbf{\theta})`
will not be minimized unless all previous residuals decrease to
some small value such that :math:`w_i` is large enough. This simple algorithm enforces a PINN model to
learn the PDE solution gradually, respecting the inherent causal structure of its dynamic evolution.

Implementation Details on causal training in PhysicsNeMo Sym are presented in script
``examples/wave_equation/wave_1d_causal.py``. :numref:`fig-wave_1d_causal` presents a comparison of the validation error between
the baseline and causal training. It can be observed that causal training yields much better predictive
accuracy up to one order of magnitude.


.. _fig-wave_1d_causal:

.. figure:: /images/user_guide/wave_1d_causal.png
   :alt: Interior validation accuracy for models trained with (blue) and without (red) the causal loss function for the 1D wave equation example.
   :width: 50.0%
   :align: center

   Interior validation accuracy for models trained with (blue) and without (red) the causal loss function for the 1D wave equation example.

It is worth noting that causal training scheme can be seamlessly combined with the moving time-window and different
network architectures in PhysicsNeMo Sym. For instance, the script ``examples/taylor_green/taylor_green_causal.py`` illustrates
how to combine the causal loss function with the time-marching strategy for solving a complex transient Navier-Stokes problem.


.. _lr_annealing:

Learning Rate Annealing
------------------------------

The predominant approach in the training of PINNs is to represent the
initial/boundary constraints as additive penalty terms to the loss
function. This is usually done by multiplying a parameter
:math:`\lambda` to each of these terms to balance out the contribution
of each term to the overall loss. However, tuning these parameters
manually is not straightforward, and also requires treating these
parameters as constants. The idea behind the learning rate
annealing, as proposed in [#wang2021understanding]_, is
an automated and adaptive rule for dynamic tuning of these parameters
during the training. Let us assume the loss function for a steady state
problem takes the following form

.. math::
   :label: loss_annealing

	L(\theta) = L_{residual}(\theta) + \lambda^{(i)} L_{BC}(\theta),

where the superscript :math:`(i)` represents the training iteration
index. Then, at each training iteration, the learning rate
annealing scheme [#wang2021understanding]_ computes the
ratio between the gradient statistics for the PDE loss term and the
boundary term, as follows

.. math:: \bar{\lambda}^{(i)} = \frac{max\left(\left|\nabla_{\theta}L_{residual}\left(\theta^{(i)}\right)\right|\right)}{mean \left(\left|\nabla_{\theta}L_{BC}\left(\theta^{(i)}\right)\right|\right)}.

Finally, the annealing parameter :math:`\lambda^{(i)}` is computed using
an exponential moving average as follows

.. math:: \lambda^{(i)} = \alpha \bar{\lambda}^{(i)} + (1-\alpha) \lambda^{(i-1)},

where :math:`\alpha` is the exponential moving average decay.

.. _homoscedastic:

Homoscedastic Task Uncertainty for Loss Weighting
--------------------------------------------------

In [#kendall2018multi]_, the authors have proposed to
use a Gaussian likelihood with homoscedastic task uncertainty as the
training loss in multi-task learning applications. In this scheme, the
loss function takes the following form

.. math::
   :label: loss_homoscedastic

	L(\theta) = \sum_{j=1}^T \frac{1}{2\sigma_j^2} L_j(\theta) + \log \Pi_{j=1}^T \sigma_j,

where :math:`T` is the total number of tasks (or residual and
initial/boundary condition loss terms). Minimizing this loss is
equivalent to maximizing the log Gaussian likelihood with homoscedastic
uncertainty [#kendall2018multi]_, and the uncertainty
terms :math:`\sigma` serve as adaptive wrights for different loss terms.
:numref:`fig-uncertainty_loss_weighting` presents a comparison
between the uncertainty loss weighting and no loss weighting for the
annular ring example, showing that uncertainty loss weighting improves
the training convergence and accuracy in this example. For details on
this scheme, please refer to [#kendall2018multi]_.

.. _fig-uncertainty_loss_weighting:

.. figure:: /images/user_guide/uncertainty_loss_weighting.png
   :alt: A comparison between the uncertainty loss weighting vs. no loss weighting for the annular ring example.
   :width: 95.0%
   :align: center

   A comparison between the uncertainty loss weighting vs. no loss
   weighting for the annular ring example.

.. _softadapt:

SoftAdapt
---------
Softadapt is a simple loss balancing algorithm that dynamically tunes the loss weights throughout the training. It
measures the relative training progress for each loss term by measuring the ratio of the loss value at each iteration
to its value at the previous iteration, and the loss weights are determined using these relative progress measurements
passed through a softmax transformation, as follows

.. math:: w_j(i) = \frac{\exp \left( \frac{L_j(i)}{L_j(i-1)} \right)}{\Sigma_{k=1}^{n_{loss}} \exp \left( \frac{L_k(i)}{L_k(i-1)} \right)}.

Here, :math:`w_j(i)` is the weight for the loss term :math:`j` at iteration :math:`i`, :math:`L_j(i)` is
the value for the loss term :math:`j` at iteration :math:`i`, and :math:`n_{loss}` is the number of loss terms.
We have observed that this softmax transformation can easily cause overflow. Thus, we modify the softadapt equation
using a softmax trick, as follows

.. math:: w_j(i) = \frac{\exp \left( \frac{L_j(i)}{L_j(i-1) + \epsilon} - \mu(i) \right)}{\Sigma_{k=1}^{n_{loss}} \exp \left( \frac{L_k(i)}{L_k(i-1)+\epsilon} - \mu(i) \right)},

where :math:`\mu(i) = \max \left(L_j(i)/L_j(i-1) \right)`, and :math:`\epsilon` is a small number to prevent division by zero.


.. _relobralo:

Relative Loss Balancing with Random Lookback (ReLoBRaLo)
--------------------------------------------------------
Relative Loss Balancing with Random Lookback (ReLoBRaLo) [#bischof2021multi]_ is a modified version of the Softadapt,
which adopts a moving average for loss weights and also a random lookback mechanism. The loss weights at each iteration
are calculated as follows

.. math:: w_j(i) = \alpha \left( \beta w_j(i-1) + (1-\beta) \hat{w}_j^{(i;0)} \right) + (1-\alpha) \hat{w}_j^{(i;i-1)}.

Here, :math:`w_j(i)` is the weight for the loss term :math:`j` at iteration :math:`i`, :math:`\alpha` is the
moving average parameter, :math:`\beta` is a Bernoulli random variable with an expected value close to 1.
:math:`\hat{w}_j^{(i;i')}` takes the following form

.. math:: \hat{w}_j^{(i;i')} = \frac{n_{loss} \exp \left( \frac{L_j(i)}{\tau L_j(i')} \right)}{\Sigma_{k=1}^{n_{loss}} \exp \left( \frac{L_k(i)}{\tau L_k(i')}\right)},

where :math:`n_{loss}` is the number of loss terms, :math:`L_j(i)` is the value for the loss term :math:`j`
at iteration :math:`i`, and :math:`\tau` is called temperature [#bischof2021multi]_. With very large values for
temperature , loss weights tend to take similar values, while a value of zero for this parameter converts
the softmax to an argmax function [#bischof2021multi]_. Similar to the modified version of softadapt, we modify the
equation for :math:`\hat{w}_j^{(i;i')}` to prevent overflow and division by zero, as follows

.. math:: \hat{w}_j^{(i;i')} = \frac{n_{loss} \exp \left( \frac{L_j(i)}{\tau L_j(i') + \epsilon} - \mu(i) \right)}{\Sigma_{k=1}^{n_{loss}} \exp \left( \frac{L_k(i)}{\tau L_k(i') + \epsilon} - \mu(i) \right)},

where :math:`\mu(i) = \max \left(L_j(i)/L_j(i') \right)`, and :math:`\epsilon` is a small number.

.. _gradnorm:

GradNorm
--------
One of the most popular loss balancing algorithms in computer vision and multi-task learning is
GradNorm [#chen2018gradnorm]_. In this algorithm, an additional loss term is minimized throughout the training
that encourages the gradient norms for different loss terms to take similar relative magnitudes,
such that the network is trained for different loss terms at similar rates.
the loss weights are dynamically tuned throughout the training
based on the relative training rates of different losses, as follows

.. math:: L_{gn}(i, w_j(i)) = \sum_j \left| G_w^{(j)}(i) - \bar{G}_W(i) \times \left[ r_j(i) \right]^\alpha  \right|_1.

Here, :math:`L_{gn}` is the GradNorm loss. :math:`W` is the subset of the neural network weights
that is used in GradNorm loss, which is typically the weights
for the last layer of the network in order to save on training costs.
:math:`G_w^{(j)}(i) = || \nabla_W w_j(i) L_j(i)||_2`
is the :math:`L_2` norm of the gradient of the weighted loss term :math:`j` with respect to the weights :math:`W`
at iteration math:`i`.
:math:`\bar{G}_W(i)=E [G_w^{(j)}(i)]` is the average gradient norm across all training losses
at iteration  :math:`i`. Also, :math:`r_j(i)=\tilde{L}_j(i)/E[\tilde{L}_j(i)]` is the relative
inverse training rate corresponding to the loss term :math:`j`, where :math:`\tilde{L}_j(i)=L_j(i)/L_j(0)` measures
the inverse training rate.
:math:`\alpha` is a hyperparameter that defines the strength of training rate balancing [#chen2018gradnorm]_.

When taking the gradients of the GradNorm loss :math:`L_{gn}(i, w_j(i))`, the reference gradient norm
:math:`\bar{G}_W(i) \times \left[ r_j(i) \right]^\alpha` is treated as a constant, and the gradnorm loss is
minimized by differentiating only with respect to the loss weights :math:`w_j`. Finally, after each training iteration,
the weights :math:`w_j` are normalized such that :math:`\Sigma_j w_j(i)=n_{loss}`, where
:math:`n_{loss}` is the number of loss terms excluding the GradNorm loss.
For more details on the GradNorm algorithm, please refer to the reference paper [#chen2018gradnorm]_.

In the GradNorm algorithm, it is observed that in some cases the weights :math:`w_j` can take negative values
and that will adversely affect the training convergence of the neural network solver. To prevent this, in the PhysicsNeMo Sym
implementation of the GradNorm, we use an exponential transformation of the trainable weight parameters to weigh
the loss terms.

In the reference paper, GradNorm has shown to be effectively improving the accuracy and reducing overfitting for
various network architectures and in both classification and regression tasks. Here, we have observed that GradNorm
can also be effective in loss balancing of neural network solvers. In particular, we have tested the
performance of GradNorm on
the annular ring example by assigning a very small initial weight to the momentum loss terms and keeping the other
loss weights intact compared to the base case. This is to evaluate whether
it can recover appropriate loss weights throughout the training by starting from this poor initial loss weighting.
Validation results are shown in the figure below.
The blue line shows the base case, red shows the case where
momentum equation are weighted by :math:`1e-4` and no loss balancing algorithm is used, and orange shows the
same case but with GradNorm used for loss balancing. It is evident that failure to balance the weight of the loss
terms appropriately in this test case will result in convergence failure, and that GradNorm can effectively
accomplish this.

.. _fig-gradnorm:

.. figure:: /images/user_guide/gradnorm.png
   :alt: GradNorm performance for loss balancing in the annular example
   :width: 99.0%
   :align: center

   GradNorm performance for loss balancing in the annular example. Blue: base case, red: momentum losses multiplied by 1e-4 and no loss balancing is used, orange: momentum losses multiplied by 1e-4 and GradNorm is used.

ResNorm
--------
Residual Normalization (ResNorm) is a PhysicsNeMo Sym loss balancing scheme developed in collaboration with the National Energy Technology Laboratory (NETL). 
In this algorithm, which is a simplified variation of GradNorm, an additional loss term is minimized during training that encourages the individual losses to take similar relative magnitudes.
The loss weights are dynamically tuned throughout the training based on the relative training rates of different losses, as follows:

.. math:: L_{rn}(i, w_j(i)) = \sum_j \left| L_w^{(j)}(i) - \bar{L}(i) \times \left[ r_j(i) \right]^\alpha  \right|_1.

Here, :math:`L_{rn}` is the ResNorm loss, 
:math:`L_w^{(j)}(i)=w_j(i) L_j(i)`
is the weighted loss term :math:`j` at iteration :math:`i`, and 
:math:`\bar{L}(i)=E [L_j(i)]` is the average loss value across all training losses
at iteration  :math:`i`. Also, :math:`r_j(i)=\tilde{L}_j(i)/E[\tilde{L}_j(i)]` is the relative
inverse training rate corresponding to the loss term :math:`j`, where :math:`\tilde{L}_j(i)=L_j(i)/L_j(0)` measures
the inverse training rate.
:math:`\alpha` is a hyperparameter that defines the strength of training rate balancing.

Similar to GradNorm, when taking the gradients of the ResNorm loss :math:`L_{rn}(i, w_j(i))` with respect to the loss weights :math:`w_j(i)`, the term
:math:`\bar{L}_(i) \times \left[ r_j(i) \right]^\alpha` is treated as a constant. 
Finally, after each training iteration, the weights :math:`w_j(i)` are normalized such that :math:`\Sigma_j w_j(i)=n_{loss}`, where :math:`n_{loss}` is the number of loss terms excluding the ResNorm loss. 
Notice that unlike GradNorm, ResNorm does not require computing gradients with respect to model parameters and thus, ResNorm can be computationally more efficient compared to GradNorm. 
Again, similar to the implementation of GradNorm, to prevent the loss weights from taking negative values, we use an exponential transformation of the trainable weight parameters to weigh the loss terms.

We test the performance of ResNorm on the annular ring example by assigning a very small initial weight to the momentum loss terms and keeping the other loss weights intact compared to the base case. 
This is to evaluate whether it can recover appropriate loss weights throughout the training by starting from this poor initial loss weighting.
Validation results are shown in the figure below. 
It is evident that ResNorm can effectively find a good balance between the loss terms and provide reasonable convergence, while the baseline case without loss balancing fails to converge.

.. _fig-resnorm:

.. figure:: /images/user_guide/resnorm.png
   :alt: ResNorm performance for loss balancing in the annular example
   :width: 99.0%
   :align: center

   ResNorm performance for loss balancing in the annular example. 
   The following are plotted: Base line (orange), momentum losses multiplied by `1e-4` (light blue) and momentum losses multiplied by `1e-4` with ResNorm (dark blue).


Neural Tangent Kernel (NTK)
---------------------------
Neural Tangent Kernel (NTK) approach can be used to automatically assign weights to different loss terms. In the NTK perspective, the weight of each loss term should be proportional to the magnitude of NTK, so that every loss term
will converge uniformly. Assume the total loss :math:`\mathcal{L}(\boldsymbol{\theta})` is defined by

.. math::
    \mathcal{L}(\boldsymbol{\theta}) = \mathcal{L}_b(\boldsymbol{\theta}) + \mathcal{L}_r(\boldsymbol{\theta}),

where

.. math::
    \mathcal{L}_b(\boldsymbol{\theta}) = \sum_{i=1}^{N_b}|u(x_b^i,\boldsymbol{\theta})-g(x_b^i)|^2,

.. math::
    \mathcal{L}_r(\boldsymbol{\theta}) = \sum_{i=1}^{N_b}|r(x_r^i,\boldsymbol{\theta})|^2

are the boundary loss and PDE residual loss, respectively. And the :math:`r` is the PDE residual. Let :math:`\mathbf{J}_r`
and :math:`\mathbf{J}_b` be the Jacobian of :math:`\mathcal{L}_r` and :math:`\mathcal{L}_b`, respectively. The the NTK of them
are defined as

.. math::
    \mathbf{K}_{bb}=\mathbf{J}_b\mathbf{J}_b^T\qquad \mathbf{K}_{rr}=\mathbf{J}_r\mathbf{J}_r^T

According to [#wang2022when]_, the weights are given by

.. math::
    \lambda_b = \frac{Tr(\mathbf{K}_{bb})+Tr(\mathbf{K}_{rr})}{Tr(\mathbf{K}_{bb})}\quad
    \lambda_r = \frac{Tr(\mathbf{K}_{bb})+Tr(\mathbf{K}_{rr})}{Tr(\mathbf{K}_{rr})},

where :math:`Tr(\cdot)` is the trace operator.

We now assign the weights by NTK. The idea of NTK is, for each loss term, its convergence rate is indicated by its eigenvalues of NTK.
So, we reweight the loss terms by their eigenvalues such that each term has basically same convergence rate. For more details, please
refer [#wang2022when]_. In PhysicsNeMo Sym, NTK can be computed automatically and weights can be assigned on the fly. The script ``examples/helmholtz/helmholtz_ntk.py`` shows the NTK implementation
for a helmholtz problem. The :numref:`fig-no-ntk` shows the results before NTK weighting. We observe that the maximum error is 0.04. Using NTK weights, this error is reduced to 0.006 as shown
in the :numref:`fig-ntk`. 

.. _fig-no-ntk:

.. figure:: /images/user_guide/helmholtz_without_ntk.png
   :alt: Helmholtz problem without NTK weights
   :width: 80.0%
   :align: center

   Helmholtz problem without NTK weights

.. _fig-ntk:

.. figure:: /images/user_guide/helmholtz_with_ntk.png
   :alt: Helmholtz problem without NTK weights
   :width: 80.0%
   :align: center

   Helmholtz problem with NTK weights


Selective Equations Term Suppression (Equation terms attention)
---------------------------------------------------------------
Selective Equations Term Suppression (SETS) is a feature developed in collaboration with National Energy Technology Laboratory (NETL). 
For several PDEs, the terms in physical equations have different scales in time and magnitude (sometimes also known as stiff PDEs). 
For such PDEs, the loss equation can appear to be minimized despite poor treatment of the smaller terms. 
To tackle this, one can create multiple instances of the same PDE and freeze certain terms (freezing is achieved by stopping the gradient calls on the term using PyTorch's ``.detach()`` in the backend). 
During the optimization process, this forces the optimizer to use the value from former iteration for the frozen terms. 
Thus, the optimizer minimizes each term in the PDE and efficiently reduces the equation residual. 
This prevents any one term in the PDE dominating the loss gradients (attention to every term). 
Creating multiple instances with different frozen term in each allows the overall representation of the physics to remain same. 


However, creating multiple instances of the same equation (with different frozen terms) also creates multiple loss terms, each of which can be weighted differently. 
This scheme can be coupled with other loss balancing algorithms like ResNorm, etc. to come up with the optimal task weights for these different instances. 


An example of creating multiple instances of equations using PhysicsNeMo Sym APIs is provided in the script ``examples/annular_ring_equation_instancing/annular_ring.py``. 
Although the incompressible navier stokes equations used in this example is not the best test for the feature (because the system of PDEs does not 
exhibit any stiffness), creating multiple instances of the momentum equations with the advection and diffusion terms frozen separately, provides 
improvement over the baseline. The effectiveness of this scheme is primarily observed more for a stiff system of PDEs with 
large scale differences in the different terms. 

.. _fig-eqn-instancing:

.. figure:: /images/user_guide/equation_instancing.png
   :alt: Equation instancing for annular ring example
   :width: 99.0%
   :align: center

   Equation instancing for annular ring example. Base line (orange), equation instancing (one instance with diffusion terms frozen and other with advection terms frozen) (gray).




.. rubric:: References

.. [#wang2021understanding] Wang, Sifan, Yujun Teng, and Paris Perdikaris. "Understanding and mitigating gradient flow pathologies in physics-informed neural networks." SIAM Journal on Scientific Computing 43.5 (2021): A3055-A3081.
.. [#kendall2018multi] Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
.. [#jagtap2020adaptive] Jagtap, Ameya D., Kenji Kawaguchi, and George Em Karniadakis. "Adaptive activation functions accelerate convergence in deep and physics-informed neural networks." Journal of Computational Physics 404 (2020): 109136.
.. [#sukumar2022exact] Sukumar, N., and Ankit Srivastava. "Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks." Computer Methods in Applied Mechanics and Engineering 389 (2022): 114333.
.. [#son2021sobolev] Son, Hwijae, Jin Woo Jang, Woo Jin Han, and Hyung Ju Hwang. "Sobolev training for the neural network solutions of pdes." arXiv preprint arXiv:2101.08932 (2021).
.. [#yu2021gradient] Yu, Jeremy, Lu Lu, Xuhui Meng, and George Em Karniadakis. "Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems." arXiv preprint arXiv:2111.02801 (2021).
.. [#heydari2019softadapt] Heydari, A. Ali, Craig A. Thompson, and Asif Mehmood. "Softadapt: Techniques for adaptive loss weighting of neural networks with multi-part loss functions." arXiv preprint arXiv:1912.12355 (2019).
.. [#bischof2021multi] Bischof, Rafael, and Michael Kraus. "Multi-objective loss balancing for physics-informed deep learning." arXiv preprint arXiv:2110.09813 (2021).
.. [#chen2018gradnorm] Chen, Zhao, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. "Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks." In International Conference on Machine Learning, pp. 794-803. PMLR, 2018.
.. [#wang2022when] Wang, S., Yu, X. and Perdikaris, P., 2022. When and why PINNs fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449, p.110768.
.. [#mnabian] The contributors to the work on hard BC imposition using the theory of R-functions and the first-order formulation of the PDEs are: M. A. Nabian, R. Gladstone, H. Meidani, N. Sukumar, A. Srivastava.
.. [#nabian2021efficient] Nabian, Mohammad Amin, Rini Jasmine Gladstone, and Hadi Meidani. "Efficient training of physics‐informed neural networks via importance sampling." Computer‐Aided Civil and Infrastructure Engineering 36.8 (2021): 962-977.
.. [#halton1960efficiency] Halton, John H. "On the efficiency of certain quasi-random sequences of points in evaluating multi-dimensional integrals." Numerische Mathematik 2.1 (1960): 84-90.
.. [#wang2022respecting] Wang, Sifan, Sankaran, Shyam, and Perdikaris, Paris. Respecting causality is all you need for training physics-informed neural networks. arXiv preprint arXiv:2203.07404, 2022.
.. [#gnanasambandam2022self] Gnanasambandam, Raghav and Shen, Bo and Chung, Jihoon and Yue, Xubo and others. Self-scalable Tanh (Stan): Faster Convergence and Better Generalization in Physics-informed Neural Networks. arXiv preprint arXiv:2204.12589, 2022.
