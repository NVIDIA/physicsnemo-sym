.. _config: 

PhysicsNeMo Sym Configuration
=============================

PhysicsNeMo Sym employs an extension of the `Hydra configuration framework <https://hydra.cc/>`_ to offer a highly customizable but user friendly method
for configuring the majority of PhysicsNeMo Sym' features.
This is achieved by using easy to understand YAML files which contain essential hyperparameters for any physics-informed
deep learning model.
While you can still achieve the same level of customization in PhysicsNeMo Sym as any deep learning library, our built in
configuration framework allows many of the internal features to be much more accessible.
This section provides an overview of the built in configurable API PhysicsNeMo Sym provides.

Minimal Example
----------------
Generally speaking, PhysicsNeMo Sym follows the same work flow as Hydra with just some minor differences.
For each PhysicsNeMo Sym program you should create a YAML configuration file that is then loaded into
a Python ``PhysicsNeMoConfig`` object which is used by PhysicsNeMo Sym. Consider the following example:

.. code-block:: yaml
   :caption: conf/config.yaml

   defaults:
    - physicsnemo_default
    - arch:
       - fully_connected
    - optimizer: adam
    - scheduler: exponential_lr
    - loss: sum

.. code-block:: python
    :caption: main.py

    import physicsnemo.sym
    from physicsnemo.sym.hydra import to_yaml
    from physicsnemo.sym.hydra.config import PhysicsNeMoConfig

    @physicsnemo.main(config_path="conf", config_name="config")
    def run(cfg: PhysicsNeMoConfig) -> None:
        print(to_yaml(cfg))

    if __name__ == "__main__":
       run()

Here, a minimal configuration (config) YAML file is shown.
A defaults list in ``config.yaml`` is used to load predefined configurations that are supported by PhysicsNeMo Sym.
This config file is then loaded into python using the ``@physicsnemo.main()`` decorator, in which you specify 
the location and name of your custom config.
The config object, ``cfg``, is then ingested into PhysicsNeMo Sym and used to setup all sorts of internals all of which 
can be individually customized as discussed in the following sections.

For this example, PhysicsNeMo Sym has been configured to load a fully connected neural network, ADAM optimizer, exponential 
decay LR scheduler and a summation loss aggregation.
Each of the included examples present in this user guide has its own config file which can be referenced.

Config Structure
----------------
Configs in PhysicsNeMo Sym are required to follow a common structure to ensure that all necessary parameters are provided independent
of the user explicitly providing them.
This is done by specifying the ``physicsnemo_default`` schema at the top of the defaults list in every configuration file which will
create the following config structure:

.. code-block:: yaml

   config
    |
    | <General parameters>
    |
    |- arch
        | <Architecture parameters>  
    |- training
        | <Training parameters>
    |- loss
        | <Loss parameters>  
    |- optimizer
        | <Optimizer parameters>  
    |- scheduler
        | <Scheduler parameters>  
    |- profiler
        | <Profiler parameters>  

This config object has multiple configuration groups that each contain separate parameters pertaining to various
features needed inside of PhysicsNeMo Sym.
As seen in the example above, these groups can be quickly populated in the defaults list (e.g. ``optimizer: adam`` will 
populate the ``optimizer`` configuration group with parameters needed for ADAM).
The next section takes a look at each of these groups in greater detail.

.. warning::
    ``- physicsnemo_default`` should always be placed at the top of your defaults list in PhysicsNeMo Sym config files. Without this, essential parameters
    will not be initialized and PhysicsNeMo Sym will not run!

.. note::
    The ``--help`` flag can be used with your PhysicsNeMo Sym program to bring
    up some useful information on different config groups or get documentation links.

Configuration Groups
---------------------

Global Parameters
^^^^^^^^^^^^^^^^^
Some essential parameters that you will find in a PhysicsNeMo Sym configuration include:

* ``jit``: Turn on TorchScript
* ``save_filetypes``: Types of file outputs from constraints, validators and inferencers
* ``debug``: Turn on debug logging
* ``initialization_network_dir``: Custom location to load pretrained models from

Architecture
^^^^^^^^^^^^
The architecture config group holds a list of model configurations that can be used to create different built in neural networks
present within PhysicsNeMo Sym.
While not required by the PhysicsNeMo Sym solver, this parameter group allows you to tune model architectures through the YAML
config file or even the command line.

To initialize an architecture using the config, PhysicsNeMo Sym provides an ``instantiate_arch()`` method that allows different architectures
to be initialized easily.
The following two examples initialize the same neural network.

.. code-block:: python
    :caption: Config model intialization

    # config/config.yaml
    defaults:
        - physicsnemo_default
        - arch:
            - fully_connected

    # Python code
    import physicsnemo.sym
    from physicsnemo.sym.hydra import instantiate_arch
    from physicsnemo.sym.hydra.config import PhysicsNeMoConfig

    @physicsnemo.main(config_path="conf", config_name="config")
    def run(cfg: PhysicsNeMoConfig) -> None:
        model = instantiate_arch(
            input_keys=[Key("x"), Key("y")],
            output_keys=[Key("u"), Key("v"), Key("p")],
            cfg=cfg.arch.fully_connected,
        )


    if __name__ == "__main__":
        run()


.. code-block:: python
    :caption: Explicit model intialization

    # Python code
    import physicsnemo.sym
    from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
    from physicsnemo.sym.models.fully_connected import FullyConnectedArch

    @physicsnemo.main(config_path="conf", config_name="config")
    def run(cfg: PhysicsNeMoConfig) -> None:
        model = FullyConnectedArch(
            input_keys=[Key("x"), Key("y")], 
            output_keys=[Key("u"), Key("v"), Key("p")],
            layer_size: int = 512,
            nr_layers: int = 6,
            ...
        )

    if __name__ == "__main__":
        run()

.. note::
    Both of these approaches yield the same model. The `instantiate_arch` approach allows the model architecture to be 
    controlled through the YAML file and CLI without loss of control. This can streamline the tuning of architecture hyperparameters.


Currently the architectures that are shipped internally in PhysicsNeMo Sym that have a configuration group include:

* ``fully_connected``: Fully connected neural network model 
* ``fourier_net``: Fourier neural network
* ``highway_fourier``: :ref:`highway_fn` - Fourier neural network with adaptive gating units 
* ``modified_fourier``:  :ref:`modified_fn` - Fourier neural network with two layers of Fourier features 
* ``multiplicative_fourier``: Fourier feature neural network with frequency connections
* ``multiscale_fourier``: :ref:`multiscale_fn` - Multi-scale Fourier feature neural network 
* ``siren``: :ref:`sirens` - Sinusoidal representation networks
* ``hash_net``: Neural network augmented by a multiresolution hash table
* ``fno``: :ref:`fno` - 1D, 2D, or 3D Fourier neural operator
* ``afno``: :ref:`afno` - Fourier neural operator based transformer model
* ``super_res``: :ref:`super_res` - Convolutional super resolution model
* ``pix2pix``: :ref:`pix2pix` - A pix2pix based convolutional encoder-decoder

Examples
~~~~~~~~
.. code-block:: python
    :caption: Initialization of fully-connected model with 5 layers of size 128

    # config.yaml
    defaults:
        - physicsnemo_default
        - arch:
            - fully_connected
        
    arch:
        fully_connected:
            layer_size: 512
            nr_layers: 6


    # Python code
    import physicsnemo.sym
    from physicsnemo.sym.hydra import instantiate_arch
    from physicsnemo.sym.hydra.config import PhysicsNeMoConfig

    @physicsnemo.main(config_path="conf", config_name="config")
    def run(cfg: PhysicsNeMoConfig) -> None:
        model = instantiate_arch(
            input_keys=[Key("x"), Key("y")],
            output_keys=[Key("u"), Key("v")],
            cfg=cfg.arch.fully_connected,
        )

    if __name__ == "__main__":
        run()

.. code-block:: python
    :caption: Initialization of modified fourier model and siren model

    # config.yaml
    defaults:
        - physicsnemo_default
        - arch:
            - modified_fourier
            - siren


    # Python code
    import physicsnemo.sym
    from physicsnemo.sym.hydra import instantiate_arch
    from physicsnemo.sym.hydra.config import PhysicsNeMoConfig

    @physicsnemo.main(config_path="conf", config_name="config")
    def run(cfg: PhysicsNeMoConfig) -> None:
        model_1 = instantiate_arch(
            input_keys=[Key("x"), Key("y")],
            output_keys=[Key("u"), Key("v")],
            frequencies=("axis,diagonal", [i / 2.0 for i in range(10)]),
            cfg=cfg.arch.modified_fourier,
        )

        model_2 = instantiate_arch(
            input_keys=[Key("x"), Key("y")],
            output_keys=[Key("u"), Key("v")],
            cfg=cfg.arch.siren,
        )


    if __name__ == "__main__":
        run()

.. warning::

    Not all model parameters are controllable through the configs. Parameters that are not supported can be specified through
    additional keyword arguments in the ``instantiate_arch`` method. Alternatively, the model can be initialized in the standard
    Pythonic approach.

Training
^^^^^^^^

The training config group contains parameters essential to the training process of the model.
This is set by default with `physicsnemo_default`, but many of the parameters contained in this group
are often essential to modify.


* ``default_training``: Default training parameters (set automatically)

Parameters
~~~~~~~~~~
Some essential parameters that you will find under the ``training`` config group include:

* ``max_steps``: Number of training iterations.
* ``grad_agg_freq``: Number of iterations to aggregate gradients over (default is 1). Effectively, setting ``grad_agg_freq=2`` will double the batch size per iteration, compared to a case with no gradient aggregation.
* ``rec_results_freq``: Frequency to record results. This value will be used as the default frequency for recording constraints, validators, inferencers and monitors. See :ref:`hydra_results` for more details.
* ``save_network_freq``: Frequency to save a network checkpoint.
* ``amp``: Use automatic mixed precision. This will set the precision for GPU operations to improve performance (default is ``'float16'`` set using ``amp_dtype``).
* ``ntk.use_ntk``: Use neural tangent kernel in training (default set to False)


Loss
^^^^
The loss config group is used to select different loss aggregations that are supported by PhysicsNeMo Sym.
A loss aggregation is the method used to combine the losses from different constraints.
Different methods can yield improved performance for some problems.


* ``sum``: Simple summation aggregation (default)
* ``grad_norm``: Gradient normalization for adaptive loss balancing
* ``homoscedastic``: :ref:`homoscedastic`
* ``lr_annealing``: :ref:`lr_annealing`
* ``soft_adapt``: Adaptive loss weighting
* ``relobralo``: Relative loss balancing with random lookback

Optimizer
^^^^^^^^^^
The loss optimizer group contains the supported optimizers that can be used in PhysicsNeMo Sym which includes ones that are built into `PyTorch <https://pytorch.org/docs/stable/optim.html#algorithms>`_ as well as from `Torch Optimizer <https://github.com/jettify/pytorch-optimizer>`_ package.
Some of the most commonly used optimizers include:

* ``adam``: ADAM optimizer
* ``sgd``: Standard stochastic gradient descent
* ``rmsprop``: The RMSProp algorithm
* ``adahessian``: Second order stochastic optimization algorithm
* ``bfgs``: L-BFGS iterative optimization method

as well as these more unique optimizers:
``a2grad_exp``, ``a2grad_inc``, ``a2grad_uni``, ``accsgd``, ``adabelief``, ``adabound``, 
``adadelta``, ``adafactor``, ``adagrad``, ``adamax``, ``adamod``, ``adamp``, ``adamw``, ``aggmo``, 
``apollo``, ``asgd``, ``diffgrad``, ``lamb``, ``madgrad``, ``nadam``, ``novograd``, ``pid``, ``qhadam``, ``qhm``, ``radam``, 
``ranger``, ``ranger_qh``, ``ranger_va``, ``rmsprop``, ``rprop``, ``sgdp``, ``sgdw``, ``shampoo``, ``sparse_adam``,  ``swats``, ``yogi``.


Scheduler
^^^^^^^^^^
The scheduler optimizer group contains the supported learning rate schedulers that can be used in PhysicsNeMo Sym.
By default none is specified for which a constant learning rate will be used.



* ``exponential_lr``: PyTorch exponential learning rate decay ``initial_learning_rate * gamma ^ (step)`` 
* ``tf_exponential_lr``: Tensorflow parameterization of exponential learning rate decay ``initial_learning_rate * decay_rate ^ (step / decay_steps)`` 


Command Line Interface
----------------------

As previously mentioned, a particular benefit using Hydra configs to control PhysicsNeMo Sym is that any of these parameters can be controlled
through CLI.
This can be particularly useful during hyperparameter tuning or queuing up multiple runs using `Hydra multirun <https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/>`_.
Here are a couple of examples which may be particularly useful when developing physics-informed models.

.. code-block:: bash
    :caption: Changing optimizer and learning rate

    $ python main.py optimizer=sgd optimizer.lr=0.01

.. code-block:: bash
    :caption: Hyperparameter search over architecture parameters using multirun

    $ python main.py -m arch.fully_connected.layer_size=128,256 arch.fully_connected.nr_layers=2,4,6

.. code-block:: bash
    :caption: Training for a different number of iterations

    $ python main.py training.max_steps=1000

.. note::
    Every parameter present in the config can be adjusted through CLI. For additional information please see the
    `Hydra documentation <https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/>`_.

Common Practices
----------------

.. _hydra_results:

Results Frequency
^^^^^^^^^^^^^^^^^

PhysicsNeMo Sym offers several different methods for recording the results of your training including recording validation, inference, batch, 
and monitor results. 
Each of these can be individually controlled in the training config group, however, typically it's preferred for each to have the same frequency.
In these instances, the ``rec_results_freq`` parameter can be used to control all of these parameters uniformly.
The following two config files are equivalent.

.. code-block:: yaml

    # config/config.yaml
    defaults:
        - physicsnemo_default
    
    training:
        rec_results_freq : 1000
        rec_constraint_freq: 2000

.. code-block:: yaml

    # config/config.yaml
    defaults:
        - physicsnemo_default
    
    training:
        rec_validation_freq: 1000
        rec_inference_freq: 1000
        rec_monitor_freq: 1000
        rec_constraint_freq: 2000


Changing Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Activations functions are one of the most important hyperparameters to test for any deep learning model.
While all of PhysicsNeMo Sym' networks have default activations functions that have been seen to provide the best performance,
specific activations may perform better than others on a case to case basis.
Changing a activation function is straight forward using the ``instantiate_arch`` method:

.. code-block:: python
    :caption: Initializing a fully-connect model with Tanh activation functions

    # Python code
    import physicsnemo.sym
    from physicsnemo.sym.hydra import instantiate_arch
    from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
    from physicsnemo.sym.models.layers import Activation

    @physicsnemo.main(config_path="conf", config_name="config")
    def run(cfg: PhysicsNeMoConfig) -> None:
        model_1 = instantiate_arch(
            input_keys=[Key("x"), Key("y")],
            output_keys=[Key("u"), Key("v")],
            cfg=cfg.arch.fully_connected,
            activation_fn=Activation.TANH,
        )

    if __name__ == "__main__":
        run()

.. warning::

    Activation functions are not currently supported in the config files. They must be set in the Python script.

Many of PhysicsNeMo Sym' models also include support for :ref:`adaptive_activations` which can be turned on in the config file or explicitly in the code:

.. code-block:: yaml

    # config/config.yaml
    defaults:
        - physicsnemo_default
        - arch:
            - fully_connected

    arch:
        fully_connected:
            adaptive_activations: true


Multiple Architectures
^^^^^^^^^^^^^^^^^^^^^^

For some problems, its better to have multiple models to learn the solution of different state variables.
This may require the use of models that are the `same` architecture with different hyperparameters.
We can have multiple neural network models with the same architecture using config group overrides in Hydra.
Here the ``arch_schema`` config group is used to access an architecture's structured config.

.. code-block:: yaml
    :caption: Extending configs with customized architectures

    # config/config.yaml
    defaults:
        - physicsnemo_default
        - /arch_schema/fully_connected@arch.model1
        - /arch_schema/fully_connected@arch.model2

    arch:
        model1:
            layer_size: 128
        model2:
            layer_size: 256


.. code-block:: python
    :caption: Initialization of two custom architectures

    # Python code
    import physicsnemo.sym
    from physicsnemo.sym.hydra import instantiate_arch
    from physicsnemo.sym.hydra.config import PhysicsNeMoConfig

    @physicsnemo.main(config_path="conf", config_name="config")
    def run(cfg: PhysicsNeMoConfig) -> None:
        model_1 = instantiate_arch(
            input_keys=[Key("x"), Key("y")],
            output_keys=[Key("u"), Key("v")],
            cfg=cfg.arch.model1,
        )

        model_2 = instantiate_arch(
            input_keys=[Key("x"), Key("y")],
            output_keys=[Key("u"), Key("v")],
            cfg=cfg.arch.model2,
        )


    if __name__ == "__main__":
        run()

Run Modes
^^^^^^^^^

PhysicsNeMo Sym has two different run modes available for training and evaluation:

* ``train``: Default run mode. Trains the neural network.

* ``eval``: Evaluates provided inferencers, monitors and validators using the last saved training checkpoint. Useful for post-processing after the training is complete. 

.. code-block:: yaml
   :caption: Changing run mode to evaluate
    
    # config/config.yaml
    defaults:
        - physicsnemo_default

    run_mode: 'eval'


Criterion Based Stopping
^^^^^^^^^^^^^^^^^^^^^^^^

PhysicsNeMo Sym supports early training termination, based on a user specified criterion, before the maximum number of iterations is reached.

* ``metric``: Metric to be monitored during the training. This can be the total loss, individual loss terms, validation metrics, or metrics in the monitor domain. For example, in the annular ring example, you can choose `loss`, `loss_continuity`, `momentum_imbalance`, or `l2_relative_error_u` as the metric. Note the use of `l2_relative_error_` for metrics from the validation domain, this is consistent with the tag used for tensorboard plots.

* ``min_delta``: Minimum required change in the metric to qualify as a training improvement.

* ``patience``: Number of training steps to wait for a training improvement to happen.

* ``mode``: Choose 'min' if the metric is to be minimized, or 'max' if the metric is to be maximized.

* ``freq``: Frequency of evaluating the stop criterion. Note that if using a metric from the validation or monitor domain, `freq` should be a multiplier of the `rec_validation_freq` or `rec_monitor_freq`.

* ``strict``: If True, raises an error in case the metric is not valid.


.. code-block:: yaml
   :caption: Defining a stopping criterion for training
    
    # config/config.yaml
    defaults:
        - physicsnemo_default

    stop_criterion:
        - metric: 'l2_relative_error_u'
        - min_delta: 0.1
        - patience: 5000
        - mode: 'min'
        - freq: 2000
        - strict: true

When using a metric from the validation domain, criterion based stopping can also serve as an early stopping regularization method for data-driven models.
