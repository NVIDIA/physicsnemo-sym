
.. _architectures:

Architectures In PhysicsNeMo Sym
================================

In this section, we discuss some of the advanced and state-of-the-art
Deep learning architectures and schemes that have become a part of
PhysicsNeMo Sym library.

Fourier Network
--------------------

Neural networks are generally biased toward low-frequency solutions, a
phenomenon that is known as "spectral bias"
[#rahaman2019spectral]_. This can adversely affect the
training convergence as well as the accuracy of the model. One approach
to alleviate this issue is to perform input encoding, that is, to
transform the inputs to a higher dimensional feature space via
high-frequency functions
[#mildenhall2020nerf]_, [#rahaman2019spectral]_, [#tancik2020fourier]_.
This is done in PhysicsNeMo Sym using the Fourier networks, which takes the
following form:

.. math::
   :label: eqn_u_net

       u_{net}(\mathbf{x};\mathbf{\theta}) = \mathbf{W}_n \big \{\phi_{n-1} \circ \phi_{n-2} \circ \cdots \circ \phi_1 \circ \phi_E \big \} (\mathbf{x}) + \mathbf{b}_n,  \; \; \; \; \phi_{i}(\mathbf{x}_i) = \sigma \left( \mathbf{W}_i \mathbf{x}_i + \mathbf{b}_i \right),

where :math:`u_{net}(\mathbf{x};\mathbf{\theta})` is the approximate
solution, :math:`\mathbf{x} \in \mathbb{R}^{d_0}` is the input to
network, :math:`\phi_{i} \in \mathbb{R}^{d_i}` is the :math:`i^{th}`
layer of the network,
:math:`\mathbf{W}_i \in \mathbb{R}^{d_i \times d_{i-1}}, \mathbf{b}_i \in \mathbb{R}^{d_i}`
are the weight and bias of the :math:`i^{th}` layer,
:math:`\mathbf{\theta}` denotes the set of network’s trainable
parameters, i.e.,
:math:`\mathbf{\theta} = \{\mathbf{W}_1, \mathbf{b}_1, \cdots, \mathbf{b}_n, \mathbf{W}_n\}`,
:math:`n` is the number of layers, and :math:`\sigma` is the activation
function. :math:`\phi_E` is an input encoding layer, and by setting that
to identity function, we arrive at the standard feed-forward
fully-connected architecture. The input encoding layer in PhysicsNeMo Sym is a
variation of the one proposed in [#tancik2020fourier]_
with trainable encoding, and takes the following form

.. math::
   :label: eqn_fourier_feature

   \phi_E = \big[ \sin \left( 2\pi \mathbf{f} \times \mathbf{x} \right); \cos \left( 2\pi \mathbf{f} \times 
   \mathbf{x} \right) \big]^T,

where :math:`\mathbf{f} \in \mathbb{R}^{n_f \times d_0}` is the
trainable frequency matrix and :math:`n_f` is the number of frequency
sets.

In the case of parameterized examples, it is also possible to apply
encoding to the parameters in addition to the spatial inputs. In fact,
it has been observed that applying encoding to the parametric inputs in
addition to the spatial inputs will improve the accuracy and the
training convergence of the model. Note that PhysicsNeMo Sym applies the input
encoding to the spatial and parametric inputs in a fully decoupled
setting and then concatenates the spatial/temporal and parametric
Fourier features together. Details on the usage of Fourier net can be
found :ref:`config` while its application for
FPGA heat sink can be found in tutorial :ref:`fpga`.

.. _modified_fn:

Modified Fourier Network
------------------------------

In Fourier network, a standard fully-connected neural network is used as
the nonlinear mapping between the Fourier features and the model output.
In modified Fourier networks, we use a variant of the fully-connected
network similar to the one proposed in
[#wang2021understanding]_. Two transformation layers are
introduced to project the Fourier features to another learned feature
space, and are then used to update the hidden layers through
element-wise multiplications, similar to its standard fully connected
counterpart in [#wang2021understanding]_. It is shown in
tutorial :ref:`fpga` that this multiplicative interaction can
improve the training convergence and accuracy, although at the cost of
slightly increasing the training time per iteration. The hidden layers
in this architecture take the following form

.. math:: \phi_{i}(\mathbf{x}_i) = \left(1 - \sigma \left( \mathbf{W}_i \mathbf{x}_i + \mathbf{b}_i \right) \right) \odot \sigma \left( \mathbf{W}_{T_1} \phi_E + \mathbf{b}_{T_1} \right) + \sigma \left( \mathbf{W}_i \mathbf{x}_i + \mathbf{b}_i \right) \odot \sigma \left( \mathbf{W}_{T_2} \phi_E + \mathbf{b}_{T_2} \right),

where :math:`i>1` and
:math:`\{ \mathbf{W}_{T_1}, \mathbf{b}_{T_1}\}, \{ \mathbf{W}_{T_2}, \mathbf{b}_{T_2}\}`
are the parameters for the two transformation layers, and :math:`\phi_E`
takes the form in equation
:eq:`eqn_fourier_feature`. Details on how to use
the modified Fourier networks can be found in 
:ref:`config` while its application for the FPGA heat
sink can be found in tutorial :ref:`fpga`.

.. _highway_fn:

Highway Fourier Network
------------------------------

Highway Fourier network is a PhysicsNeMo Sym variation of the Fourier
network, inspired by the highway networks proposed in
[#srivastava2015training]_. Highway networks consist of
adaptive gating units that control the flow of information, and are
originally developed to be used in the training of very deep networks
[#srivastava2015training]_. The hidden layers in this
network take the following form

.. math:: \phi_{i}(\mathbf{x}_i) = \sigma \left( \mathbf{W}_i \mathbf{x}_i + \mathbf{b}_i \right) \odot \sigma_s \left( \mathbf{W}_{T} \phi_E + \mathbf{b}_{T} \right) + \left( \mathbf{W}_P \mathbf{x} + \mathbf{b}_P \right) \odot \left (1 - \sigma_s \left( \mathbf{W}_{T} \phi_E + \mathbf{b}_{T} \right) \right).

Here, :math:`\sigma_s` is the sigmoid activation,
:math:`\{ \mathbf{W}_{T}, \mathbf{b}_{T}\}` are the parameters of the
transformer layer, and :math:`\{ \mathbf{W}_{P}, \mathbf{b}_{P}\}` are
the parameters of the projector layer, which basically projects the
network inputs to another space to match with the dimensionality of
hidden layers. The transformer layer here controls the relative
contribution of the network’s hidden state and the network’s input to
the output of the hidden layer. It also offers a multiplicative
interaction mechanism between the network’s input and hidden states,
similar to the modified Fourier network. Details on how to use the
highway Fourier networks can be found in 
:ref:`config` while its application for the FPGA heat
sink can be found in tutorial :ref:`fpga`.

.. _multiscale_fn:

Multi-scale Fourier Feature Network
----------------------------------------

In [#wang2021eigenvector]_, Wang *et. al.* proposed a
multi-scale Fourier feature network architecture that aim to tackle
partial differential equations exhibiting multi-scale behaviors. The key
of the proposed architectures is to apply multiple Fourier feature
embeddings initialized with different frequencies to input coordinates
before passing these embedded inputs through the same fully-connected
neural network and finally concatenate the outputs with a linear layer.
The forward pass of the multi-scale Fourier feature networks is given by

.. math::

   \begin{aligned}
       &\phi_{E}^{(i)}(\mathbf{x})=[\sin (2 \pi \mathbf{f}^{(i)} \times \mathbf{x}) ; \cos (2 \pi \mathbf{f}^{(i)} \times \mathbf{x})]^{T},  \quad \text{ for } i=1, 2, \dots, M\\
       &\mathbf{H}^{(i)}_1 = \sigma(\mathbf{W}_1 \cdot\phi_{E}^{(i)}(\mathbf{x})  + \mathbf{b}_1),  \quad \text{ for } i=1, 2, \dots, M \\
       & \mathbf{H}^{(i)}_\ell = \sigma(\mathbf{W}_\ell \cdot \mathbf{H}^{(i)}_{\ell - 1}  + \mathbf{b}_\ell),  \quad \text{ for } \ell=2,  \dots, L,  i=1, 2, \dots, M\\
           & \mathbf{u}_{net}(\mathbf{x}, {\mathbf{\theta}}) = \mathbf{W}_{L+1} \cdot \left[  \mathbf{H}^{(1)}_L,  \mathbf{H}^{(2)}_L, \dots,   \mathbf{H}^{(M)}_L  \right] + \mathbf{b}_{L+1},\end{aligned}

where :math:`\phi_{E}^{(i)}` and :math:`\sigma` denote Fourier feature
mappings and activation functions, respectively, and each entry in
:math:`\mathbf{f}^{(i)} \in \mathbb{R}^{m \times d}` is sampled from a Gaussian
distribution :math:`\mathcal{N}(0, \sigma_i)`. Notice that the weights
and the biases of this architecture are essentially the same as in a
standard fully-connected neural network with the addition of the
trainable Fourier features. Here, we underline that the choice of
:math:`\sigma_i` is problem dependent and typical values can be
:math:`1, 10, 100,` etc.

.. 
   Spatio-temporal Fourier Feature Network
   ----------------------------------------
   
   For time-dependent problems, multi-scale behavior may exist not only
   across spatial directions but also across time. The authors
   [#wang2021eigenvector]_ proposed another novel multi-scale
   Fourier feature architecture to tackle multi-scale problems in
   spatio-temporal domains. Specifically, the feed-forward pass of the
   network is now defined as
   
   .. math::
   
      \begin{aligned}
           &\phi_{E}^{(x_i)}(x_i)=[\sin (2 \pi \mathbf{f}^{(x_i)} \times x_i) ; \cos (2 \pi \mathbf{f}^{(x_i)} \times \mathbf{x}_i)]^{T}, \\
        & \phi_{E}^{(t)}(t)=[\sin (2 \pi \mathbf{f}^{(t)} \times t) ; \cos (2 \pi \mathbf{f}^{(t)} \times x_i)]^{T}, \\ 
        & \mathbf{H}^{(x_i)}_1 = \sigma(\mathbf{W}_1 \cdot \phi_{E}^{(x_i)}(x_i) + \mathbf{b}_1),
          \quad \text{ for } i=1, 2, \dots, d,\\
        & \mathbf{H}^{(t)}_1 = \sigma(\mathbf{W}_1 \cdot \phi_{E}^{(t)}(t) + \mathbf{b}_1),\\
          & \mathbf{H}_{\ell}^{(x_i)} = \sigma(\mathbf{W}_\ell \cdot \mathbf{H}^{(x_i)}_{\ell-1}  + \mathbf{b}_\ell),  \quad \text{ for } \ell=2,  \dots, L \text{ and } i=1,2, \dots, d,\\
        & \mathbf{H}^{(t)}_{\ell} = \sigma(\mathbf{W}_\ell \cdot \mathbf{H}^{(t)}_{\ell-1}  + \mathbf{b}_\ell),  \quad \text{ for } \ell=2,  \dots, L, \\
          &    \mathbf{H}_{L} =  \prod_{i=1}^d H^{(x_i)}_{L} \cdot H^{(t)}_{L}  , \\
        & \mathbf{u}_{net}(\mathbf{x}, t; {\mathbf{\theta}}) = \mathbf{W}_{L+1} \cdot \mathbf{H}_{L} + \mathbf{b}_{L+1},\end{aligned}
   
   where :math:`\phi_{E}^{(x_i)}` and :math:`\phi_{E}^{(t)}` denote spatial
   and temporal Fourier feature mappings, respectively, and :math:`\odot`
   represents the point-wise multiplication. Here, each entry of
   :math:`\mathbf{f}^{(x_i)}` and :math:`\mathbf{f}^{(t)}` can be sampled
   from different Gaussian distributions. One key difference from the
   multi-scale Fourier feature network is that separate Fourier feature
   embeddings are applied to spatial and temporal input coordinates before
   passing the embedded inputs through the same fully-connected network.
   Another key difference is that network outputs are merged using
   point-wise multiplication and passing them through a linear layer.

.. _sirens:

Sinusoidal Representation Networks (SiReNs)
--------------------------------------------------

In [#sitzmann2020implicit]_, the authors propose a
neural network using Sin activation functions dubbed sinusoidal
representation networks or SiReNs. This network has similarities to the
Fourier networks above because using a Sin activation function has the
same effect as the input encoding for the first layer of the network. A
key component of this network architecture is the initialization scheme.
The weight matrices of the network are drawn from a uniform distribution
:math:`W \sim U(-\sqrt{\frac{6}{fan\_in}},\sqrt{\frac{6}{fan\_in}})`
where :math:`fan\_in is` is the input size to that layer. The input of
each Sin activation has a Gauss normal distribution and the output of
each Sin activation, an arcSin distribution. This preserves the
distribution of activations allowing deep architectures to be
constructed and trained effectively
[#sitzmann2020implicit]_. The first layer of the network
is scaled by a factor :math:`\omega` to span multiple periods of the Sin
function. This was empirically shown to give good performance and is in
line with the benefits of the input encoding in the Fourier network. The
authors suggest :math:`\omega=30` to perform well under many
circumstances and is the default value given in PhysicsNeMo Sym as well. Details
on how to use the SiReN architecture in PhysicsNeMo Sym can be found in 
:ref:`config`.

DGM Architecture
--------------------

The DGM architecture is proposed by
[#sirignano2018dgm]_, and consists of several
fully-connected layers each of which includes a number of sublayers,
similar in spirit to the LSTM architecture, as follows:

.. math::

   \begin{split}
   &S^1 = \sigma(XW^1 + b^1),\\
   &Z^\ell = \sigma(XV_z^{\ell} + S^{\ell}W_z^{\ell} + b_z^{\ell}), \>\>\>\> \forall \ell \in \{1,\cdots,n_{\ell}\},\\
   &G^\ell = \sigma(XV_g^{\ell} + S^{\ell}W_g^{\ell} + b_g^{\ell}), \>\>\>\> \forall \ell \in \{1,\cdots,n_{\ell}\},\\
   &R^\ell = \sigma(XV_r^{\ell} + S^{\ell}W_r^{\ell} + b_r^{\ell}), \>\>\>\> \forall \ell \in \{1,\cdots,n_{\ell}\},\\
   &H^\ell = \sigma(XV_h^{\ell} + (S^\ell \odot R^\ell)^{\ell}W_h^{\ell} + b_h^{\ell}), \>\>\>\> \forall \ell \in \{1,\cdots,n_{\ell}\},\\
   &S^{\ell+1} = (1-G^\ell) \odot H^\ell + Z^\ell \odot S^\ell,\\
   &u_{net}(X;\theta) = S^{n_\ell+1}W + b.
   \end{split}

The set of DGM network parameters include

.. math:: \theta = \{W^1,b^1,\left(V_z^{\ell},W_z^{\ell},b_z^{\ell}\right)_{\ell=1}^{n_\ell}, \left(V_g^{\ell},W_g^{\ell},b_g^{\ell}\right)_{\ell=1}^{n_\ell}, \left(V_r^{\ell},W_r^{\ell},b_r^{\ell}\right)_{\ell=1}^{n_\ell}, \left(V_h^{\ell},W_h^{\ell},b_h^{\ell}\right)_{\ell=1}^{n_\ell},W,b\}.

where :math:`X` is the input to the network, :math:`\sigma(\cdot)` is
the activation function, :math:`n_\ell` is the number of hidden layers,
:math:`\odot` is the Hadamard product, and :math:`u_{net}(X;\theta)` is
the network output. One important feature of this architecture is that
it consists of multiple element-wise multiplication of nonlinear
transformations of the input, and that can potentially help with
learning complicated functions [#sirignano2018dgm]_.
Application for this architecture using
the FPGA heat sink can be found in tutorial :ref:`fpga`.

Multiplicative Filter Network
------------------------------

Multiplicative filter networks
[#fathony2020multiplicative]_ consist of linear or
nonlinear transformations of Fourier or Gabor filters of the input,
multiplied together at each hidden layer, as follows:

.. math::

   \begin{split}
   &\mathbf{\phi}_1 = f(\mathbf{x}, \mathbf{\xi}_1),\\
   &\mathbf{\phi}_{i+1} = \sigma \left( \mathbf{W}_i \mathbf{\phi}_i + \mathbf{b}_i \right) \odot f(\mathbf{x}, \mathbf{\xi}_{i+1}), \>\>\>\> \forall i \in \{1,\cdots,n-1\},\\
   &u_{net}(\mathbf{x};\mathbf{\theta}) = \mathbf{W}_n \mathbf{\phi}_n + \mathbf{b}_n.
   \end{split}

Here, :math:`f(\mathbf{x}, \mathbf{\xi}_{i})` is a multiplicative
Fourier or Gabor filter. The set of multiplicative filter network
parameters are
:math:`\theta = \{\mathbf{W}_1, \mathbf{b}_1, \mathbf{\xi}_1, \cdots \mathbf{W}_n, \mathbf{b}_n, \mathbf{\xi}_n \}`.
Note that in the original implementation in
[#fathony2020multiplicative]_, no activation function is
used, and network nonlinearity comes from the multiplicative filters
only. In this setting, it has been shown in
[#fathony2020multiplicative]_ that the output of a
multiplicative Filter network can be represented as a linear combination
of Fourier or Gabor bases. In PhysicsNeMo Sym, the user can choose whether to use
activation functions or not. The Fourier filters take the following
form:

.. math:: f(\mathbf{x}, \mathbf{\xi}_{i}) = \sin(\mathbf{\omega}_i \mathbf{x} + \mathbf{\phi}_i),

where :math:`\mathbf{\xi}_i = \{\mathbf{\omega}_i, \mathbf{\phi}_i\}`.
The Gabor filters also take the following form:

.. math:: f(\mathbf{x}, \mathbf{\xi}_{i}) = \exp \left( - \frac{\mathbf{\gamma}_i}{2} ||\mathbf{x}-\mathbf{\mu}_i||_2^2 \right) \sin(\mathbf{\omega}_i \mathbf{x} + \mathbf{\phi}_i),

where
:math:`\mathbf{\xi}_i = \{\mathbf{\gamma}_i, \mathbf{\mu}_i, \mathbf{\omega}_i, \mathbf{\phi}_i\}`.
For details on the multiplicative filter networks and network
initialization, please refer to
. Details on how to use the
multiplicative filter networks can be found in 
:ref:`config`.

.. _fno:

Fourier Neural Operator
----------------------------------------

Fourier neural operator (FNO) is a data-driven architecture which can be used to parameterize solutions
for a distribution of PDE solutions [#zongyi2020fourier]_.
The key feature of FNO is the spectral convolutions:  operations that place the integral kernel in Fourier space.
The spectral convolution (Fourier integral operator) is defined as follows:

.. math::
   :label: spectral_conv

       (\mathcal{K}(\mathbf{w})\phi)(x) = \mathcal{F}^{-1}(R_{\mathbf{W}}\cdot \left(\mathcal{F}\right)\phi)(x), \quad \forall x \in D

where :math:`\mathcal{F}` and :math:`\mathcal{F}^{-1}` are the forward and inverse Fourier transforms, respectively.
:math:`R_{\mathbf{w}}` is the transformation which contains the learnable parameters :math:`\mathbf{w}`. Note this operator is calculated
over the entire `structured Euclidean` domain :math:`D` discretized with :math:`n` points.
Fast Fourier Transform (FFT) is used to perform the Fourier transforms efficiently and the resulting transformation :math:`R_{\mathbf{w}}`
is just finite size matrix of learnable weights.
In side the spectral convolution, the Fourier coefficients are truncated to only the lower modes which intern allows
explicit control over the dimensionality of the spectral space and linear operator.

The FNO model is a the composition of a fully-connected "lifting" layer, :math:`L` spectral convolutions with point-wise 
linear skip connections and a decoding point-wise fully-connected neural network at the end.

.. math::
   :label: fno

        u_{net}(\Phi;\theta) = \mathcal{Q}\circ \sigma(W_{L} + \mathcal{K}_{L}) \circ ... \circ \sigma(W_{1} + \mathcal{K}_{1})\circ \mathcal{P}(\Phi), \quad \Phi=\left\{\phi(x); \forall x \in D\right\}
    
in which :math:`\sigma(W_{i} + \mathcal{K}_{i})` is the spectral convolution layer :math:`i` with the point-wise linear
transform :math:`W_{i}` and activation function :math:`\sigma(\cdot)`.
:math:`\mathcal{P}` is the point-wise lifting network that projects the input into a higher dimensional latent space, 
:math:`\mathcal{P}: \mathbb{R}^{d_in} \rightarrow \mathbb{R}^{k}`.
Similarly :math:`\mathcal{Q}` is the point-wise fully-connected decoding network, :math:`\mathcal{P}: \mathbb{R}^{k} \rightarrow \mathbb{R}^{d_out}`.
Since all fully-connected components of FNO are point-wise operations, the model is invariant to the dimensionality of
the input.
Additional information on FNO and its implementation in PhysicsNeMo Sym can be found in the example :ref:`darcy_fno`.

.. note::
    While FNO is technically invariant to the dimensionality of the discretized domain :math:`D`, this domain *must* be
    a structured grid in Euclidean space. The inputs to FNO are analogous to images, but the model is invariant to the image
    resolution.

.. _afno:

Adaptive Fourier Neural Operator
--------------------------------

The Adaptive Fourier Neural Operator (AFNO) [#guibas2021adaptive]_ architecture is highly effective and computationally efficient for high-resolution inputs. 
It combines a key recent advance in modeling PDE systems, namely the Fourier Neural Operator (FNO) with the powerful Vision Transformer (ViT) model for image processing. 
FNO has shown great results in modeling PDE systems such as Navier-Stokes flows. 
The ViT and related variants of transformer models have achieved SOTA performance in image processing tasks. 
The multi-head self-attention (MHSA) mechanism of the ViT is key to its impressive performance. 
The self-attention mechanism models long range interactions at each layer of the neural network, a feature that is absent in most convolutional neural networks. 
The drawback of the ViT self-attention architecture is that it scales as a quadratic function of the length of the token sequence, and thus scales quadratically with input image resolution. 
The AFNO provides a solution to the scaling complexity of the ViT. The AFNO model implements a token mixing operation in the Fourier Domain. 
The computational complexity of the mixing operation is :math:`\mathcal{O}(N_{token}\log N_{token})` as opposed to the :math:`\mathcal{O}({N_{token}^2})` complexity of the vanilla ViT architecture.

The first step in the architecture involves dividing the input image into a regular grid with :math:`h \times w` equal sized patches of size :math:`p\times p`. 
The parameter :math:`p` is referred to as the patch size. For simplicity, we consider a single channel image. Each patch is embedded into a token of size :math:`d`, the embedding dimension. 
The patch embedding operation results in a token tensor (:math:`X_{h\times w \times d}`) of size :math:`h \times w \times d`. 
The patch size and embedding dimension are user selected parameters. 
A smaller patch size allows the model to capture fine scale details better while increasing the computational cost of training the model. 
A higher embedding dimension also increases the parameter count of the model. The token tensor is then processed by multiple layers of the transformer architecture performing spatial and channel mixing. 
The AFNO architecture implements the following operations in each layer.

The token tensor is first transformed to the Fourier domain with

.. math:: z_{m,n} = [\mathrm{DFT}(X)]_{m,n},

where :math:`m,n` is the index the patch location and DFT denotes a 2D discrete Fourier transform.
The model then applies token weighting in the Fourier domain and promotes sparsity with a Soft-Thresholding and Shrinkage operation as

.. math:: \tilde{z}_{m,n} = S_{\lambda} ( \mathrm{MLP}(z_{m,n})),

where :math:`S_{\lambda}(x) = \mathrm{sign}(x) \max(|x| - \lambda, 0)` with the sparsity controlling parameter :math:`\lambda`, and :math:`\mathrm{MLP(\cdot)}` is a two layer perceptron with block diagonal weight matrices which are shared across all patches. 
The number of blocks in the block diagonal MLP weight matrices is a user selected hyperparameter that should be tuned appropriately.
The last operation in a ANFO layer is an inverse Fourier to transform back to the patch domain and add a residual connection as

.. math:: y_{m,n} = [\mathrm{IDFT}(\tilde{Z})]_{m,n} + X_{m,n}.

At the end of all the transformer layers, a linear decoder converts the feature tensor back to the image space.

There are several important hyperparameters that affect the accuracy and computational cost of the AFNO. 
Empirically, the most important hyperparameters that should be tuned keeping in mind the task at hand are the number of layers, patch size, the embedding dimension and the number of blocks.
Additional information on AFNO and its implementation in PhysicsNeMo Sym can be found in the example :ref:`darcy_afno`.

.. _pino:

Physics Informed Neural Operator
----------------------------------------
The Physics-Informed Neural Operator (PINO) was introduced in [#li2021physics]_.
The PINO approach for surrogate modeling of PDE systems effectively combines the data-informed supervised learning framework of the :ref:`fno` with the physics-informed learning framework. 
The PINO incorporates a PDE loss :math:`\mathcal{L}_{pde}` to the Fourier Neural Operator. 
This reduces the amount of data required to train a surrogate model, since the PDE loss constrains the solution space. 
The PDE loss also enforces physical constraints on the solution computed by a surrogate ML model, making it an attractive option as a verifiable, accurate and interpretable ML surrogate modeling tool.

We consider a stationary PDE system for simplicity, although the PINO method can be applied to dynamical systems as well. 
Following the notation used in [#li2021physics]_, we consider a PDE represented by,

.. math:: \mathcal{P}(u, a) = 0 , \text{ in } D \subset \mathbb{R}^d, 
.. math::  u = g ,  \text{ in } \partial D.

Here, :math:`\mathcal{P}` is a Partial Differential Operator, :math:`a` are the coefficients/parameters and :math:`u` is the PDE solution.

In the FNO framework, the surrogate ML model is given by a the solution operator :math:`\mathcal{G}^\dagger_{\theta}`, which maps any given coefficient in the coefficient space :math:`a` to the solution :math:`u`. 
The FNO is trained in a supervised fashion using training data in the form of input/output pairs :math:`\lbrace a_j, u_j \rbrace_{j = 1}^N`.
The training loss for the FNO is given by summing the data loss, :math:`\mathcal{L}_{data}(\mathcal{G}_\theta) = \lVert u - \mathcal{G}_\theta(a)  \rVert^2` 
over all training pairs :math:`\lbrace a_i, u_i,  \rbrace_{i=1}^N`,

In the PINO framework, the solution operator is optimized with an additional PDE loss given by :math:`\mathcal{L}_{pde}(a, \mathcal{G}_{\theta}(a))` computed over i.i.d. samples :math:`a_j` from an appropriate supported distribution in parameter/coefficient space.

In general, the PDE loss involves computing the PDE operator which in turn involves computing the partial derivatives of the Fourier Neural Operator ansatz. In general this is nontrivial. The key set of innovations in the PINO are the various ways to compute the partial derivatives of the operator ansatz. The PINO framework implements the differentiation in four different ways.

#. Numerical differentiation using a finite difference Method (FDM).
#. Numerical differentiation computed via spectral derivative. 
#. Hybrid differentiation based on a combination of first-order "exact" derivatives and second-order FDM derivatives. 

.. _deeponet_theory:

DeepONet
--------
Deep operator network (DeepONet) was first introduced in [#lulu2021deeponet]_, a network architecture that aims to learn operators between infinite dimensional function spaces.
Later a physics-informed version was proposed in [#wang2021deeponet]_, introducing an effective regularization mechanism for biasing the outputs of DeepOnet models towards ensuring physical consistency.


Suppose that an operator :math:`G` is defined by

.. math:: G:\quad u\mapsto G(u),

where :math:`u` and :math:`G(u)` are two functions. We denote the variable at the domain of :math:`G(u)` as :math:`y`.


As showed in [#lulu2021deeponet]_, the DeepONet consists of two subnetworks referred as branch and trunk nets. The branch net takes :math:`u` as input and outputs
a feature embedding of :math:`q` dimensions, where :math:`u=[u(x_1), u(x_2), \dots, u(x_m)]` represents a function u evaluated
at a collection of fixed sensors :math:`\{x_i\}_i^m`. The trunk net takes the coordinates :math:`y` as the input and also outputs
a feature embedding of :math:`q` dimensions. The final output of DeepONet is obtained by merging the outputs of the branch and trunk nets via
a dot product.

The plain DeepONet is trained in a supervised fashion by minimizing a summing data loss
:math:`\mathcal{L}_{data}(\mathcal{G}_\theta) = \lVert \mathcal{G}_\theta(u)(y) - \mathcal{G}(u)(y) \rVert^2`
over all training input output triplets :math:`\lbrace u_i, y_i, G(u_i)(y_i) \rbrace_{i=1}^N`.
The physics-informed DeepONet is trained with an additional PDE loss given by


.. math:: \mathcal{L}_{pde}(\mathcal{G}_\theta) = \lVert  \mathcal{N}[\mathcal{G}_\theta(u)](y) \rVert^2

where :math:`\mathcal{N}` is a differential operator denoting the governing PDE of the underlying physical laws.


It is worth mentioning  that users have flexibility to choose the architecture of branch and trunk net. For example, CNN can be used as a backbone of the branch net to extract features
from high-dimensional data. This will be more efficient than a fully-connected neural network as has been mentioned in [#lulu2021deeponet]_.



.. _pix2pix:

Pix2Pix Net
-----------

Pix2Pix network in PhysicsNeMo Sym is a convolutional encoder-decoder based on the pix2pix [#isola2017imagetoimage]_ and pix2pixHD [#wang2018high]_ generator models.
The implementation inside of PhysicsNeMo Sym is a streamlined version of these models that can be used for various problems involving data that is structured.
This model consists of three main components: downscaling layers, residual blocks and upscaling layers.

.. figure:: /images/user_guide/pix2pix_arch.png
   :alt: Pix2Pix architecture in PhysicsNeMo Sym
   :width: 60.0%
   :align: center

The downscaling part of the model consists of a set of :math:`n` convolutions that reduce the dimensionality of the feature input.
Each layer consists as a set of convolutions with a stride of 2, normalization operation and activation function :math:`\sigma`:

.. math:: z = (\textrm{Conv}(s=2) \circ \textrm{BatchNorm}(\cdot) \circ \sigma(\cdot))^{\circ n}(x).

The middle of the model consists to :math:`m` residual blocks of the following form:

.. math:: z = \left(\textrm{z}_{i} + (\textrm{Conv} \circ \textrm{BatchNorm}(\cdot) \circ \sigma(\cdot) \circ \textrm{Conv} \circ \textrm{BatchNorm}(\cdot) \circ \sigma(\cdot))\right)^{\circ m}(z),

where :math:`\textrm{z}_{i}` indicates the output from the previous convolutional block.
Lastly, the upscaling section mirrors the downscaling section with transposed convolutions:

.. math:: y = (\textrm{ConvT}(s=2) \circ \textrm{BatchNorm}(\cdot) \circ \sigma(\cdot))^{\circ n}(z).

The pix2pix encoder-decoder also allows users to upscale the resolution of the input output feature with an additional set of transpose convolutional layers.
Information regarding using this model in PhysicsNeMo Sym can be found in the example :ref:`turbulence_super_res`.

.. _super_res:

Super Resolution Net
--------------------

The super resolution network in PhysicsNeMo Sym is a convolutional decoder that is specifically designed for super resolution problems [#ledig2012high]_.
This model can be particularly useful for mapping between low to high resolution data that is on a structured grid.
This model consists of just two parts: convolutional residual blocks and upscaling blocks.

.. figure:: /images/user_guide/super_res_arch.png
   :alt: Super resolution architecture in PhysicsNeMo Sym
   :width: 50.0%
   :align: center

The front of the model consists to :math:`m` residual blocks consisting of two standard convolutional operations, normalization, and activation function :math:`\sigma`:

.. math:: z = \left(\textrm{z}_{i} + (\textrm{Conv} \circ \textrm{BatchNorm}(\cdot) \circ \sigma(\cdot) \circ \textrm{Conv} \circ \textrm{BatchNorm}(\cdot) \circ \sigma(\cdot))\right)^{\circ m}(x),

where :math:`\textrm{z}_{i}` indicates the output from the previous convolutional block.
The second part of the model consists of :math:`n` upscaling blocks which each consist of a convolutional operation, a pixel shuffle upscaling and activation function. 

.. math:: y = (\textrm{Conv} \circ \textrm{PixShuffle}(s=2) \circ \sigma(\cdot))^{\circ n}(z).

Each upscaling layer increases the dimensionality of the feature by a factor of 2.
Additional information regarding this model and its use in PhysicsNeMo Sym can be found in the example :ref:`turbulence_super_res`.


.. rubric:: References
   
.. [#rahaman2019spectral] Rahaman, Nasim, et al. "On the spectral bias of neural networks." International Conference on Machine Learning. PMLR, 2019.
.. [#mildenhall2020nerf] Mildenhall, Ben, et al. "Nerf: Representing scenes as neural radiance fields for view synthesis." European conference on computer vision. Springer, Cham, 2020.
.. [#tancik2020fourier] Tancik, Matthew, et al. "Fourier features let networks learn high frequency functions in low dimensional domains." Advances in Neural Information Processing Systems 33 (2020): 7537-7547.
.. [#wang2021understanding] Wang, Sifan, Yujun Teng, and Paris Perdikaris. "Understanding and mitigating gradient flow pathologies in physics-informed neural networks." SIAM Journal on Scientific Computing 43.5 (2021): A3055-A3081.
.. [#srivastava2015training] Srivastava, Rupesh K., Klaus Greff, and Jürgen Schmidhuber. "Training very deep networks." Advances in neural information processing systems 28 (2015).
.. [#sitzmann2020implicit] Sitzmann, Vincent, et al. "Implicit neural representations with periodic activation functions." Advances in Neural Information Processing Systems 33 (2020): 7462-7473
.. [#sirignano2018dgm] Sirignano, Justin, and Konstantinos Spiliopoulos. "DGM: A deep learning algorithm for solving partial differential equations." Journal of computational physics 375 (2018): 1339-1364.
.. [#fathony2020multiplicative] Fathony, Rizal, et al. "Multiplicative filter networks." International Conference on Learning Representations. 2020.
.. [#wang2021eigenvector] Wang, Sifan, Hanwen Wang, and Paris Perdikaris. "On the eigenvector bias of fourier feature networks: From regression to solving multi-scale pdes with physics-informed neural networks." Computer Methods in Applied Mechanics and Engineering 384 (2021): 113938.
.. [#zongyi2020fourier] Li, Zongyi, et al. "Fourier Neural Operator for Parametric Partial Differential Equations." International Conference on Learning Representations. 2020.
.. [#guibas2021adaptive] Guibas, John, et al. "Adaptive fourier neural operators: Efficient token mixers for transformers" International Conference on Learning Representations, 2022.
.. [#li2021physics] Li, Zongyi, et al. "Li, Zongyi, et al. "Physics-informed neural operator for learning partial differential equations." arXiv preprint arXiv:2111.03794 (2021).
.. [#lulu2021deeponet] Lu, L., Jin, P., Pang, G., Zhang, Z. and Karniadakis, G.E., 2021. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3), pp.218-229.
.. [#wang2021deeponet] Wang, S., Wang, H. and Perdikaris, P., 2021. Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. Science advances, 7(40), p.eabi8605.
.. [#isola2017imagetoimage] Isola, Phillip, et al. "Image-To-Image translation With conditional adversarial networks" Conference on Computer Vision and Pattern Recognition, 2017.
.. [#wang2018high] Wang, Ting-Chun, et al. "High-Resolution image synthesis and semantic manipulation with conditional GANs" Conference on Computer Vision and Pattern Recognition, 2018.
.. [#ledig2012high] Ledig, Christian, et al. "Photo-Realistic single image super-resolution using a generative adversarial network" Conference on Computer Vision and Pattern Recognition, 2017.
