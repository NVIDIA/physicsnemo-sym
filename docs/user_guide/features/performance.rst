Performance
=============

A collection of various methods for accelerating PhysicsNeMo Sym are presented below. 
The figures below show a summary of performance improvements using various PhysicsNeMo Sym features over different releases. 

.. _fig-v100_speedup:

.. figure:: /images/user_guide/perf-comparisons-v100.png
   :alt: Speed-up across different PhysicsNeMo Sym releases on V100 GPUs
   :width: 60.0%
   :align: center

   Speed-up across different PhysicsNeMo Sym releases on V100 GPUs. (MFD: Meshless Finite Derivatives)

.. _fig-a100_speedup:

.. figure:: /images/user_guide/perf-comparisons-a100.png
   :alt: Speed-up across different PhysicsNeMo Sym releases on A100 GPUs
   :width: 60.0%
   :align: center

   Speed-up across different PhysicsNeMo Sym releases on A100 GPUs. (MFD: Meshless Finite Derivatives)

.. note::
    The higher vRAM in A100 GPUs means that we can use twice the batch size/GPU compared to the V100 runs. 
    For comparison purposes, the total batch size is held constant, hence the A100 plots use 2 A100 GPUs.
    
.. note::
    These figures are only for summary purposes and the runs were performed on the flow part of the example presented in :ref:`limerock`. 
    For more details on performance gains due to individual features, please refer to the subsequent sections.  


Running jobs using TF32 math mode
---------------------------------

`TensorFloat-32 (TF32) <https://blogs.NVIDIA.com/blog/2020/05/14/tensorfloat-32-precision-format/>`_ is a new math mode available on NVIDIA A100 GPUs
for handing matrix math and tensor operations used during the training
of a neural network. 

On A100 GPUs, the TF32 feature is "ON" by default and you do not need to
make any modifications to the regular scripts to use this feature. With
this feature, you can obtain up to 1.8x speed-up over FP32 on A100 GPUs 
for the FPGA problem. This allows us to achieve same results with 
dramatically reduced training times (:numref:`fig-fpga_tf32_speedup`) without change in accuracy and loss convergence (:numref:`tab-fpga-tf32` and :numref:`fig-fpga_tf32`).

.. _fig-fpga_tf32_speedup:

.. figure:: /images/user_guide/fpga_TF32_speedup.png
   :alt: Speed-up using TF32 on an A100 GPU.
   :width: 60.0%
   :align: center

   Achieved speed-up using the TF32 compute mode on an A100 GPU for the FPGA example

.. _tab-fpga-tf32:

.. table:: Comparison of results with and without TF32 math mode
   :align: center

   +---------------------------+-----------------------+
   | **Case Description**      | :math:`P_{drop}`      |
   |                           | :math:`(Pa)`          |
   +---------------------------+-----------------------+
   | **PhysicsNeMo Sym:** Fully    | 29.24                 |
   | Connected Networks        |                       |
   | with FP32                 |                       |
   +---------------------------+-----------------------+
   | **PhysicsNeMo Sym:** Fully    | 29.13                 |
   | Connected Networks        |                       |
   | with TF32                 |                       |
   +---------------------------+-----------------------+
   | **OpenFOAM Solver**       | 28.03                 |
   +---------------------------+-----------------------+
   | **Commercial Solver**     | 28.38                 |
   +---------------------------+-----------------------+
   
   

.. _fig-fpga_tf32:

.. figure:: /images/user_guide/TF32vFP32.png
   :alt: Loss convergence plot for FPGA simulation with TF32 feature
   :name: fig:fpga_tf32
   :width: 60.0%
   :align: center

   Loss convergence plot for FPGA simulation with TF32 feature

Running jobs using Just-In-Time (JIT) compilation
---------------------------------------------------
JIT compilation is a feature where elements of the computational graph 
can be compiled from native PyTorch to the `TorchScript <https://pytorch.org/docs/stable/jit.html>`_ backend. This 
allows for optimizations like avoiding python's Global Interpreter 
Lock (GIL) as well as compute optimizations including dead code 
elimination, common substring elimination and pointwise kernel fusion. 

PINNs used in PhysicsNeMo Sym have many peculiarities including the presence 
of many pointwise operations. Such operations, while being computationally 
inexpensive, put a large pressure on the memory subsystem of a GPU. JIT 
allows for kernel fusion, so that many of these operations can be computed 
simultaneously in a single kernel and thereby reducing the number of memory 
transfers from GPU memory to the compute units.

JIT is enabled by default in PhysicsNeMo Sym through the ``jit`` option in the config 
file. You can optionally disable JIT by adding a ``jit: false`` option in the
config file or add a ``jit=False`` command line option.

CUDA Graphs
------------

PhysicsNeMo Sym supports CUDA Graph optimization which can accelerate problems that are launch latency bottlenecked and improve parallel performance.
Due to the strong scaling of GPU hardware, some machine learning problems can struggle keeping the GPU saturated resulting in work submission latency.
This also impacts scalability due to work getting delayed from these bottlenecks.
CUDA Graphs provides a solution to this problem by allowing the CPU to submit a sequence of jobs to the GPU rather than individually.
For problems that are not matrix multiplied bound on the GPU, this can produce a notable speed up.
Regardless of performance gains, it is recommended to use CUDA Graphs when possible, particularly when using multi-GPU and multi-node training.
For additional details on CUDA Graphs in PyTorch, the reader is refered to the `PyTorch Blog <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>`_.

There are three steps to using CUDA Graphs:

1. Warm-up phase where training is executed normally.
2. Recording phase during which the forward and backward kernels during one training iteration are recorded into a graph.
3. Replay of the recorded graph which is used for the rest of training.

PhysicsNeMo Sym supports this PyTorch utility and is turned on by default.
CUDA Graphs can be enabled using Hydra.
It is suggested to use at least 20 warm-up steps, which is the default.
After 20 training iterations, PhysicsNeMo Sym will then attempt to record a CUDA Graph and if successful it will replay it for the remainder of training.

.. code-block:: yaml
    
    cuda_graphs: True
    cuda_graph_warmup: 20

.. warning::
    CUDA Graphs is presently a beta feature in PyTorch and may change in the future.
    This feature requires newer `NCCL versions <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html>`_ and host GPU drivers (R465 or greater). 
    If errors are occurring please verify your drivers are up to date.

.. warning::
    CUDA Graphs do not work for all user guide examples when using multiple GPUs. 
    Some examples requires :code:`find_unused_parameters` when using DDP, which is not compatible with CUDA Graphs.

.. note::
    NVTX markers do not work inside of CUDA Graphs, thus we suggest shutting this feature off when profiling the code.

Meshless Finite Derivatives
---------------------------

Meshless finite derivatives is an alternative approach for calculating derivatives for physics-informed learning.
Rather than relying on automatic differentiation to compute analytical gradients, meshless finite derivatives queries stencil points on the fly to approximate the gradients using finite difference.
With autodiff, multiple automatic differentiation calls are needed to calculate the higher-order derivatives as well as the backward pass for optimization.
The trouble is that computational complexity exponentially increases for every additional autodiff pass needed, which can significantly slow training.
Meshless finite derivatives replaces the need for autodiff with additional forward passes.
Since the finite difference stencil points are queried on demand, no grid discretion is needed preserving mesh free training.

For many problems, the additional computation needed for the foward passes in meshless finite derivatives is far less than the autodiff equivalent.
This approach can potentially yield anywhere from a :math:`2-4` times speed-up over the autodiff approach with comparable accuracy.

To use meshless finite derivatives, one just needs to define a :code:`MeshlessFiniteDerivative` node and add it to a constraint that will require gradient quantities.
PhysicsNeMo Sym will prioritize the use of meshless finite derivatives over autodiff when provided.
When creating a  :code:`MeshlessFiniteDerivative` node, the derivatives that will be needed must be explicitly defined.
This can be done though just a list, or accessing needed derivatives from other nodes.
Additionally, this node requires a node that has the inputs consist of the independent variables and output being the quantities derivatives are needed for.
For example, the derivative :math:`\partial f / \partial x` with require a node with input variables that contain :math:`x` and outputs :math:`f`.
Switching to meshless finite derivatives is straight forward for most problems.
As an example, for LDC the following code snippet turns on meshless finite derivative providing a :math:`3` times speed-up:

.. code:: python

    from physicsnemo.sym.eq.derivatives import MeshlessFiniteDerivative

    # Make list of nodes to unroll graph on
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected
    )
    flow_net_node = flow_net.make_node(name="flow_network", jit=cfg.jit)
    # Define derivatives needed to be calculated
    # Requirements for 2D N-S
    derivatives_strs = set(["u__x", "v__x", "p__x", "v__x__x", "u__x__x", "u__y", "v__y", \
        "p__y", "u__y__y", "v__y__y"])
    derivatives = Key.convert_list(derivatives_strs)
    # Or get the derivatives from the N-S node itself
    derivatives = []
    for node in ns.make_nodes():
        for key in node.derivatives:
            derivatives.append(Key(key.name, size=key.size, derivatives=key.derivatives))

    # Create MFD node
    mfd_node = MeshlessFiniteDerivative.make_node(
        node_model=flow_net_node,
        derivatives=derivatives,
        dx=0.001,
        max_batch_size=4*cfg.batch_size.Interior,
    )
    # Add to node list
    nodes = ns.make_nodes() + [flow_net_node, mfd_node]


.. warning::
    Meshless Finite Derivatives is a development from the PhysicsNeMo Sym team and is presently in beta. 
    Use at your own discretion; stability and convergence is not garanteed.
    API subject to change in future versions.


Present Pitfalls
^^^^^^^^^^^^^^^^

* Setting the ``dx`` parameter is a very critical part of meshless finite derivatives. 
  While classical numerical methods offer clear guidance on this topic, these do not directly apply here due additional stability constraints placed by the backwards pass and optimization.
  For most problems in our user guide a ``dx`` close to `0.001` works well and yields good convergence, lower will likely lead to instability during training with a ``float32`` precision model.
  Additional details, tools and guidance on the specification of ``dx`` will be forthcoming in the near future.

* Meshless finite derivatives can increase the noise during training compared to automatic differentiation due its approximate nature. 
  Thus this feature is currently not suggested for problems that are exhibit unstable training characteristics for automatic differentiation.

* Meshless finite derivatives can converge to the wrong solution and accuracy is highly dependent on the ``dx`` used.

* Performance gains are problem specific and is based on the derivatives needed.
  Presently the best way to further increase the performance of meshless finite derivatives, users should increase ``max_batch_size`` when creating the meshless finite derivative node.

* PhysicsNeMo Sym will add automatic differentiation nodes if all required derivatives are not specified to the meshless finite derivative.

Running jobs using multiple GPUs
--------------------------------

To boost performance and to run larger problems, PhysicsNeMo Sym supports
multi-GPU and multi-node scaling. This allows for multiple
processes, each targeting a single GPU, to perform independent forward
and backward passes and aggregate the gradients collectively before
updating the model weights. The :numref:`fig-fpga_scaling` shows the scaling performance of
PhysicsNeMo Sym on the laminar FPGA test problem (script can be found at
``examples/fpga/laminar/fpga_flow.py``) up to 1024 A100 GPUs on 128
nodes. The scaling efficiency from 16 to 1024 GPUs is almost 85%.

This data parallel fashion of multi-GPU training keeps the number of
points sampled per GPU constant while increasing the total effective
batch size. You can use this to your advantage to increase the number of
points sampled by increasing the number of GPUs allowing you to handle
much larger problems.

To run a PhysicsNeMo Sym solution using multiple GPUs on a single compute node,
one can first find out the available GPUs using

.. code:: bash

   nvidia-smi

Once you have found out the available GPUs, you can run the job using
``mpirun -np #GPUs``. Below command shows how to run the job using 2
GPUs.

.. code:: bash

   mpirun -np 2 python fpga_flow.py 


PhysicsNeMo Sym supports running a problem on multiple nodes as well using a 
SLURM scheduler. Simply launch a job using ``srun`` and the appropriate 
flags and PhysicsNeMo Sym will set up the multi-node distributed process group.
The command below shows how to launch a 2 node job with 8 GPUs per node 
(16 GPUs in total):

.. code:: bash

   srun -n 16 --ntasks-per-node 8 --mpi=none python fpga_flow.py

PhysicsNeMo Sym also supports running on other clusters that do not have a SLURM 
scheduler as long as the following environment variables are set for each
process:

- ``MASTER_ADDR``: IP address of the node with rank 0
- ``MASTER_PORT``: port that can be used for the different processes to communicate on
- ``RANK``: rank of that process
- ``WORLD_SIZE``: total number of participating processes
- ``LOCAL_RANK`` (optional): rank of the process on it's node

For more information, see `Environment variable initialization <https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization>`_

.. _fig-fpga_scaling:

.. figure:: /images/user_guide/fpga_multi_node_scaling.png
   :alt: FPGA scaling
   :width: 60.0%
   :align: center

   Multi-node scaling efficiency for the FPGA example
