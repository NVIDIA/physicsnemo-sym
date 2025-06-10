.. _constraints_doc: 

Constraints
===========

PhysicsNeMo Sym uses constraints to define the objectives for neural network training. These house a set of nodes from which a computational graph is built for execution as well as loss function. 
Many physical problems require multiple training objectives/constraints to be defined in a well-posed manner. The constraints in PhysicsNeMo Sym are designed to provide the means for intuitively 
setting up multi-objective problems.

Several types of constraints are available within PhysicsNeMo Sym that allow you to quickly setup your AI training either in a physics-informed or data-informed fashion. 
At the core, the various constraints in PhysicsNeMo Sym sample a dataset, execute the computational nodes on the generated samples and compute the loss for each constraint. This individual loss is 
then combined with the losses of other user-defined constraints using a aggregator method selected. The combined loss is then passed to the optimizer for 
optimization. The different variants available in PhysicsNeMo Sym makes the definition of some common types of constraints easy so that you do not have to write a lot of boilerplate code
for sampling and evaluating. Each constraint is recorded in the ``Domain`` class which is input to the ``Solver``. 

Continuous Constraints
----------------------

The word "continuous" here is used primarily to indicate the constraints is applied on points sampled uniformly randomly in the continuous space or surface of the geometry. For a physics-informed training, it is very typical 
to apply the PDE constraints in the interior of the domain and boundary conditions on the domain boundaries. Several other constraints to apply integral losses are also available.

PointwiseBoundaryConstraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The boundary of a PhysicsNeMo Sym' geometry object can be sampled using ``PointwiseBoundaryConstraint`` class. 
This will sample the entire boundary of the geometry specified as input to the ``geometry`` parameter. 
In the case of 1D, the boundaries are the end points, for 2D, its the points along the perimeter, 
and for 3D its the points on the surface of the geometry. 

Mathematically the pointwise boundary constraint can be represented as

By default, all the boundaries will be sampled by this class and subsampling is possible using the ``criteria`` parameter. 
The ``outvar`` parameter is used for describing the constraint. The outvar dictionaries are used when 
unrolling the computational graph (specified using the ``nodes`` parameter) and computing the loss. 
The number of points to sample on each boundary are specified using the ``batch_size`` parameter. 
A detailed description of all the arguments can be found in the API documentation. 

Mathematically the pointwise boundary constraint can be represented as

.. math::

   L = \left| \int_{\partial \Omega} ( u_{net}(x,y,z) - \phi ) \right|^p = \left| \frac{S}{B} \sum_{i}(u_{net}(x_i, y_i, z_i) - \phi) \right|^p

Where :math:`L` is the loss, :math:`\partial \Omega` is the boundary, :math:`u_{net}(x,y,z)` is the network prediction for the keys in ``outvar``, 
:math:`\phi` is the value specified in the ``outvar`` and :math:`p` is the norm of the loss. :math:`S` and :math:`B` are the surface area/perimeter and batch size respectively.  


Below, a simple boundary condition definition is shown. Here the problem is trying to only satisfy the boundary.


.. code-block:: python
    :caption: Boundary constraint

    import numpy as np
    from sympy import Symbol, Function, Number, pi, sin
    
    import physicsnemo.sym
    from physicsnemo.sym.hydra import to_absolute_path, PhysicsNeMoConfig
    from physicsnemo.sym.solver import Solver
    from physicsnemo.sym.domain import Domain
    from physicsnemo.sym.geometry.primitives_1d import Point1D, Line1D
    from physicsnemo.sym.domain.constraint import (
        PointwiseBoundaryConstraint,
    )
    from physicsnemo.sym.key import Key
    from physicsnemo.sym.node import Node
    from physicsnemo.sym.models.fully_connected import FullyConnectedArch
    
    @physicsnemo.main(config_path="conf", config_name="config")
    def run(cfg: PhysicsNeMoConfig) -> None:
    
        # make list of nodes to unroll graph on
        u_net = FullyConnectedArch(
            input_keys=[Key("x")], output_keys=[Key("u")], nr_layers=3, layer_size=32
        )
    
        nodes = [u_net.make_node(name="u_network")]
    
        # add constraints to solver
        # make geometry
        x = Symbol("x")
        geo = Line1D(0, 1)
    
        # make domain
        domain = Domain()
    
        # bcs
        bc = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"u": 0},
            batch_size=2,
        )
        domain.add_constraint(bc, "bc")
    
        # make solver
        slv = Solver(cfg, domain)
    
        # start solver
        slv.solve()
    
    if __name__ == "__main__":
        run()

PointwiseInteriorConstraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The interior of a PhysicsNeMo Sym' geometry object can be sampled using ``PointwiseInteriorConstraint`` class. 
This will sample the entire interior of the geometry specified as input to the ``geometry`` parameter. 

Similar to boundary sampling, subsampling is possible using the ``criteria`` parameter. The ``outvar`` and ``batch_size`` parameters 
work in the same way as ``PointwiseBoundaryConstraint``.
A detailed description of all the arguments can be found in the API documentation. 

Mathematically the pointwise interior constraint can be represented as

.. math::

   L = \left| \int_{\Omega} ( u_{net}(x,y,z) - \phi ) \right|^p = \left| \frac{V}{B} \sum_{i}(u_{net}(x_i, y_i, z_i) - \phi) \right|^p

Where :math:`L` is the loss, :math:`\Omega` is the interior, :math:`u_{net}(x,y,z)` is the network prediction for the keys in ``outvar``, 
:math:`\phi` is the value specified in the ``outvar`` and :math:`p` is the norm of the loss. :math:`V` and :math:`B` are the volume/area and batch size respectively.  


Below, a simple interior constraint definition is shown.


.. code-block:: python
    :caption: Interior constraint

    import numpy as np
    from sympy import Symbol, Function, Number, pi, sin
    
    import physicsnemo.sym
    from physicsnemo.sym.hydra import to_absolute_path, PhysicsNeMoConfig
    from physicsnemo.sym.solver import Solver
    from physicsnemo.sym.domain import Domain
    from physicsnemo.sym.geometry.primitives_1d import Point1D, Line1D
    from physicsnemo.sym.domain.constraint import (
        PointwiseBoundaryConstraint,
        PointwiseInteriorConstraint,
    )
    from physicsnemo.sym.domain.inferencer import PointwiseInferencer
    from physicsnemo.sym.key import Key
    from physicsnemo.sym.node import Node
    from physicsnemo.sym.models.fully_connected import FullyConnectedArch
    from physicsnemo.sym.eq.pde import PDE
    
    class CustomPDE(PDE):
        def __init__(self, f=1.0):
            # coordinates
            x = Symbol("x")
    
            # make input variables
            input_variables = {"x": x}
    
            # make u function
            u = Function("u")(*input_variables)
    
            # source term
            if type(f) is str:
                f = Function(f)(*input_variables)
            elif type(f) in [float, int]:
                f = Number(f)
    
            # set equations
            self.equations = {}
            self.equations["custom_pde"] = (
                u.diff(x, 2) - f
            )  # "custom_pde" key name will be used in constraints
    
    
    @physicsnemo.main(config_path="conf", config_name="config")
    def run(cfg: PhysicsNeMoConfig) -> None:
    
        # make list of nodes to unroll graph on
        eq = CustomPDE(f=1.0)
        u_net = FullyConnectedArch(
            input_keys=[Key("x")], output_keys=[Key("u")], nr_layers=3, layer_size=32
        )
    
        nodes = eq.make_nodes() + [u_net.make_node(name="u_network")]
    
        # add constraints to solver
        # make geometry
        x = Symbol("x")
        geo = Line1D(0, 1)
    
        # make domain
        domain = Domain()
    
        # interior
        interior = PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"custom_pde": 0},
            batch_size=100,
            bounds={x: (0, 1)},
        )
        domain.add_constraint(interior, "interior")
    
        # make solver
        slv = Solver(cfg, domain)
    
        # start solver
        slv.solve()


    if __name__ == "__main__":
        run()



IntegralBoundaryConstraint
~~~~~~~~~~~~~~~~~~~~~~~~~~

This constraint samples points on the boundary of the geometry object similar to the ``PointwiseBoundaryConstraint``, but now instead of computing a pointwise loss, it computes monte-carlo integration of specified variable and then assigns the specified value to it to compute the loss. Mathematically this can be shown as below: 

.. math::

   L = \left| \int_{\partial \Omega} u_{net}(x,y,z) - \phi \right|^p = \left| \left(\frac{S}{B} \sum_{i}u_{net}(x_i, y_i, z_i)\right) - \phi \right|^p

Where :math:`L` is the loss, :math:`\partial \Omega` is the boundary, :math:`u_{net}(x,y,z)` is the network prediction for the keys in ``outvar``, 
:math:`\phi` is the value specified in the ``outvar`` and :math:`p` is the norm of the loss. :math:`S` and :math:`B` are the volume/area and batch size respectively.  

Please note that the ``batch_size`` has a slightly different meaning here. The ``batch_size`` parameter is used to define the number of instances of integrals to apply while
the ``integral_batch_size`` is the actual points sampled on the boundary. 

Below, a simple integral constraint definition is shown.


.. code-block:: python
    :caption: Integral constraint

    import numpy as np
    from sympy import Symbol, Function, Number, pi, sin
    
    import physicsnemo.sym
    from physicsnemo.sym.hydra import to_absolute_path, PhysicsNeMoConfig
    from physicsnemo.sym.solver import Solver
    from physicsnemo.sym.domain import Domain
    from physicsnemo.sym.geometry.primitives_1d import Point1D, Line1D
    from physicsnemo.sym.domain.constraint import (
        IntegralBoundaryConstraint,
    )
    from physicsnemo.sym.domain.inferencer import PointwiseInferencer
    from physicsnemo.sym.key import Key
    from physicsnemo.sym.node import Node
    from physicsnemo.sym.models.fully_connected import FullyConnectedArch
    from physicsnemo.sym.eq.pde import PDE
    
    
    @physicsnemo.main(config_path="conf", config_name="config")
    def run(cfg: PhysicsNeMoConfig) -> None:
    
        # make list of nodes to unroll graph on
        u_net = FullyConnectedArch(
            input_keys=[Key("x")], output_keys=[Key("u")], nr_layers=3, layer_size=32
        )
    
        nodes = [u_net.make_node(name="u_network")]
    
        # add constraints to solver
        # make geometry
        x = Symbol("x")
        geo = Line1D(0, 1)
    
        # make domain
        domain = Domain()
    
        # integral
        integral = IntegralBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"u": 0},
            batch_size=1,
            integral_batch_size=100,
        )
        domain.add_constraint(integral, "integral")
    
        # make solver
        slv = Solver(cfg, domain)
    
        # start solver
        slv.solve()
    
    
    if __name__ == "__main__":
        run()




Discrete Constraints
--------------------

For discrete constrains, the constraint is applied on a structure of fixed points taken from a discretized representation of the space. The simplest example of this is a uniform grid.


SupervisedGridConstraint
~~~~~~~~~~~~~~~~~~~~~~~~

This constraint performs standard supervised training on grid data. This constraint also supports the use of multiple workers, which are particularly important when using lazy loading. This constraint is primarily used for grid based models like Fourier Neural Operators. Losses computed in these constraint are pointwise similar to the above boundary and interior constraints. 

Below, a simple supervised grid constraint definition is shown.

.. code-block:: python
    :caption: Supervised Grid Constraint from the Darcy flow example

    import physicsnemo.sym
    from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
    from physicsnemo.sym.key import Key
    
    from physicsnemo.sym.solver import Solver
    from physicsnemo.sym.domain import Domain
    from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
    from physicsnemo.sym.dataset import HDF5GridDataset
    
    from physicsnemo.sym.utils.io.plotter import GridValidatorPlotter
    
    from utilities import download_FNO_dataset
    
    
    @physicsnemo.main(config_path="conf", config_name="config_FNO")
    def run(cfg: PhysicsNeMoConfig) -> None:
    
        # load training/ test data
        input_keys = [Key("coeff", scale=(7.48360e00, 4.49996e00))]
        output_keys = [Key("sol", scale=(5.74634e-03, 3.88433e-03))]
    
        download_FNO_dataset("Darcy_241", outdir="datasets/")
        train_path = to_absolute_path(
            "datasets/Darcy_241/piececonst_r241_N1024_smooth1.hdf5"
        )
        test_path = to_absolute_path(
            "datasets/Darcy_241/piececonst_r241_N1024_smooth2.hdf5"
        )
    
        # make datasets
        train_dataset = HDF5GridDataset(
            train_path, invar_keys=["coeff"], outvar_keys=["sol"], n_examples=1000
        )
        test_dataset = HDF5GridDataset(
            test_path, invar_keys=["coeff"], outvar_keys=["sol"], n_examples=100
        )
    
        # make list of nodes to unroll graph on
        model = instantiate_arch(
            input_keys=input_keys,
            output_keys=output_keys,
            cfg=cfg.arch.fno,
        )
        nodes = model.make_nodes(name="FNO", jit=cfg.jit)
    
        # make domain
        domain = Domain()
    
        # add constraints to domain
        supervised = SupervisedGridConstraint(
            nodes=nodes,
            dataset=train_dataset,
            batch_size=cfg.batch_size.grid,
            num_workers=4,  # number of parallel data loaders
        )
        domain.add_constraint(supervised, "supervised")
    
        # make solver
        slv = Solver(cfg, domain)
    
        # start solver
        slv.solve()
    
    
    if __name__ == "__main__":
        run()


Defining a custom constraint
----------------------------

User defined custom constraints can be implemented by inheriting from the ``Constraint`` class defined in ``physicsnemo/domain/constraint/constraint.py``. 
There are 3 methods you will need to specify to use your constraint, ``load_data``, ``loss`` and ``save_batch``. 
The ``load_data`` method is used to load a mini-batch of data from the internal dataloader. The ``loss`` method computes loss used when training. 
Lastly, the ``save_batch`` method specifies how to save a batch of for debugging or post processing. 
This structure is meant to be general and allows for many complex constraints to be formed such as those used in variational methods. 
For references on implementations of these methods please refer to any of the above base constraints.


