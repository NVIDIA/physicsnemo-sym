
Post Processing in PhysicsNeMo Sym
==================================


.. _tensorboard:

TensorBoard in PhysicsNeMo Sym
------------------------------

Introduction
^^^^^^^^^^^^

This section shows you how to visualize the outputs of your model as it trains, by adding (custom) plots to TensorBoard. These visualizations provide an easy way to qualitatively assess the performance of your model.
Plots can be made using PhysicsNeMo Sym validators (i.e. plotting the output of your model compared to some ground truth dataset) or inferencers (i.e. just plotting the output of your model given a set of inputs).
You can use the default plotter provided, or you can define your own custom plotter.
An example custom TensorBoard plot for the lid driven cavity example :ref:`Introductory Example` is shown here:

.. _fig-custom-plot:

.. figure:: ../../images/user_guide/tensorboard_custom.png
   :alt: Custom TensorBoard plots
   :width: 80.0%
   :align: center

   Example custom TensorBoard plots for the lid driven cavity example.

Workflow Overview
^^^^^^^^^^^^^^^^^^

Here is the overall workflow for adding plots to TensorBoard:

#. Instantiate either a ``ValidatorPlotter`` or a ``InferencerPlotter`` class from ``physicsnemo.utils.io.plotter``. For example, ``plotter = ValidatorPlotter()``.

#. Pass this plotter as an optional argument when creating a validator or inferencer object. For example, ``validator = PointwiseValidator(invar, true_outvar, nodes, plotter=plotter)``.

#. Add this validator or inferencer object to your domain / solver as you normally would.

PhysicsNeMo Sym handles the rest and at a certain number of training iterations, the plotter adds plots of the validator's or inferencer's inputs and outputs to TensorBoard.
To define a custom plotter, you can define your own ``Plotter`` class which inherits from either ``ValidatorPlotter`` or ``InferencerPlotter`` and overrides it's ``__call__`` method. More details are given in the lid driven cavity example below.

.. note:: 
    You can change the frequency at which these plots are added to TensorBoard by changing the values of `rec_validation_freq` and `rec_inference_freq` in your project's configuration file :ref:`config`. Plotting less frequently can avoid the creation of large TensorBoard event files.
    
    The plots can be found in the `Images` tab in TensorBoard.

Lid Driven Cavity Example
^^^^^^^^^^^^^^^^^^^^^^^^^

To show you how to use this workflow, an example of creating custom TensorBoard plots for the lid driven cavity  (:ref:`Introductory Example`) example is provided below.
First you define a custom ``ValidatorPlotter`` class, overriding its ``__call__`` methods with a custom plotting function:


.. code:: python

    import numpy as np
    import scipy.interpolate
    import matplotlib.pyplot as plt

    from physicsnemo.sym.utils.io.plotter import ValidatorPlotter

    # define custom class
    class CustomValidatorPlotter(ValidatorPlotter):

        def __call__(self, invar, true_outvar, pred_outvar):
            "Custom plotting function for validator"
            
            # get input variables
            x,y = invar["x"][:,0], invar["y"][:,0]
            extent = (x.min(), x.max(), y.min(), y.max())        
            
            # get and interpolate output variable
            u_true, u_pred = true_outvar["u"][:,0], pred_outvar["u"][:,0]
            u_true, u_pred = self.interpolate_output(x, y, 
                                                    [u_true, u_pred], 
                                                    extent,
            )
            
            # make plot
            f = plt.figure(figsize=(14,4), dpi=100)
            plt.suptitle("Lid driven cavity: PINN vs true solution")
            plt.subplot(1,3,1)
            plt.title("True solution (u)")
            plt.imshow(u_true.T, origin="lower", extent=extent, vmin=-0.2, vmax=1)
            plt.xlabel("x"); plt.ylabel("y")
            plt.colorbar()
            plt.vlines(-0.05, -0.05, 0.05, color="k", lw=10, label="No slip boundary")
            plt.vlines( 0.05, -0.05, 0.05, color="k", lw=10)
            plt.hlines(-0.05, -0.05, 0.05, color="k", lw=10)
            plt.legend(loc="lower right")
            plt.subplot(1,3,2)
            plt.title("PINN solution (u)")
            plt.imshow(u_pred.T, origin="lower", extent=extent, vmin=-0.2, vmax=1)
            plt.xlabel("x"); plt.ylabel("y")
            plt.colorbar()
            plt.subplot(1,3,3)
            plt.title("Difference")
            plt.imshow((u_true-u_pred).T, origin="lower", extent=extent, vmin=-0.2, vmax=1)
            plt.xlabel("x"); plt.ylabel("y")
            plt.colorbar()
            plt.tight_layout()
            
            return [(f, "custom_plot"),]
        
        @staticmethod
        def interpolate_output(x, y, us, extent):
            "Interpolates irregular points onto a mesh"
            
            # define mesh to interpolate onto
            xyi = np.meshgrid(
                np.linspace(extent[0], extent[1], 100),
                np.linspace(extent[2], extent[3], 100),
                indexing="ij",
            )
            
            # linearly interpolate points onto mesh
            us = [scipy.interpolate.griddata(
                (x, y), u, tuple(xyi)
                )
                for u in us]
            
            return us


.. note:: 
    The inputs to ``__call__`` are dictionaries of the model's inputs and output variables, as specified when you initialise the validator or inferencer object associated with the plotter. For ``ValidatorPlotter``, the ground truth output variables are also passed.
    The ``__call__`` function should return a list  of type ``[(Figure, "<name>"), ...]``, where ``Figure`` is a ``matplotlib`` figure and ``"<name>"`` is a name string assigned to each figure in TensorBoard.


Next, change the following lines in the example code:

.. code:: python

    openfoam_validator = PointwiseValidator(
        ...,
        plotter=CustomValidatorPlotter(),
    )

Finally, run the example code. You should automatically see your plots being added to TensorBoard in the `Images` tab as the model trains.



.. _vtk:

VTK Utilities in PhysicsNeMo Sym
--------------------------------

Introduction
^^^^^^^^^^^^

The primary output file format supported by PhysicsNeMo Sym are `Visualization Toolkit (VTK) <https://vtk.org/>`_ files which are widely used across multiple scientific domains.
A key benefit of VTK files is VTK's large library of filters one can use on the data as well as support from industry standard visualization software support such as `ParaView <https://www.paraview.org/>`_.
If you are unfamiliar with VTK and ParaView, you are encouraged to look over the `ParaView documentation <https://docs.paraview.org/en/latest/>`_ to help get started.
PhysicsNeMo Sym supports several VTK utilities to help make importing and exporting data effortless.

VTK outputs are selected by default in PhysicsNeMo Sym, which can be controlled using the ``save_filetypes`` parameter in the Hydra config.
PhysicsNeMo Sym supports several VTK data formats (legacy and XML versions) including:

.. list-table:: PhysicsNeMo Sym VTK Data Types
   :widths: 15 15 60 10
   :header-rows: 1

   * - VTK Class
     - PhysicsNeMo Sym Wrapper
     - Description
     - File extension
   * - ``vtkUniformGrid``
     - ``VTKUniformGrid``
     - Data stored on a uniform grid, such as an image.
     - ``.vti``
   * - ``vtkRectilinearGrid``
     - ``VTKRectilinearGrid``
     - Data stored on a rectilinear domain, such as a square domain with nonuniform mesh density.
     - ``.vtr``
   * - ``vtkStructuredGrid``
     - ``VTKStructuredGrid``
     - Data stored on a structured domain. This includes structured meshes with curved boundaries.
     - ``.vts``
   * - ``vtkUnstructuredGrid``
     - ``VTKUnstructuredGrid``
     - Data stored on an unstructured mesh domain.
     - ``.vtu``
   * - ``vtkPolyData``
     - ``VTKPolyData``
     - General polygon data. Can contain objects including points, lines, faces, cells, etc.
     - ``.vtp``

Generally speaking, these file types are listed most to least restrictive.
PhysicsNeMo Sym primarily will use ``vtkPolyData`` to output data given its flexibility, but other formats can offer significant memory savings if applicable.

.. warning::

    PhysicsNeMo Sym currently does not support multi-block VTK files.


Converting Variables to VTK Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The workhorses of PhysicsNeMo Sym' post-processing are the two functions ``var_to_polyvtk`` and  ``grid_to_vtk``, which are used for unstructured point data and grid data, respectively.
Both of these functions take dictionaries of numpy arrays and write them to VTK files.
When writing a custom constraint, inferencer or validator, using one of these functions will likely be needed to record your results.


.. _var_to_polyvtk:

`var_to_polyvtk`
~~~~~~~~~~~~~~~~
This function converts the dictionary, ``Dict[str: np.array]``, of variable data into a point cloud using a ``vtkPolyData`` dataset.
The number of data points in the first dimension of all arrays in the input dictionary *must* be consistent.
Additionally, the dictionary must include variables that represent the items' spatial location.
While not memory efficient, this function will ubiquitously work with all data as long as spatial coordinates are provided.

To better understand the conversion, consider the following minimal example for a 2D point cloud:

.. code-block:: python

    import numpy as np
    from physicsnemo.sym.utils.io.vtk import var_to_polyvtk

    n_points = 500
    save_var = {
        "U": np.random.randn(n_points, 2), # Different number of var dims supported
        "p": np.random.randn(n_points, 1), 
        "x": np.random.uniform(0, 1 ,size=(n_points, 1)),  # x coordinates
        "y": np.random.uniform(0, 1 ,size=(n_points, 1)), # y coordinates
        # PhysicsNeMo Sym will fill in z locations with zero
    }
    var_to_polyvtk(save_var, "./test_file")


.. figure:: ../../images/user_guide/vtk_poly_data.png
    :alt: `vtkPolyData` visualization example
    :width: 60.0%
    :align: center
    
    Visualization of `test_file.vtp` in ParaView


`grid_to_vtk`
~~~~~~~~~~~~~
This function converts a dictionary, ``Dict[str: np.array]``, of variable data into a uniform grid using a `vtkUniformGrid`` dataset.
``grid_to_vtk`` is built with image based data in mind, thus expects arrays to be of the form: ``[batch, D, xdim]``, ``[batch, D, xdim, ydim]`` or ``[batch, D, xdim, ydim, zdim]`` for 1D, 2D and 3D data, respectively.
Note that all spatial dimensions must be identical between dictionary entries.
Unlike ``var_to_polyvtk``, `no coordinates` are provided.
A good example of this function being used in a custom constraint is in the :ref:`turbulence_super_res` example.

The following minimal example will demonstrate this function for a 3D grid:

.. code-block:: python

    import numpy as np
    from physicsnemo.sym.utils.io.vtk import grid_to_vtk

    n_points = 20
    batch_size = 2
    save_var = {
        "U": np.random.randn(batch_size, 2, n_points, n_points, n_points),
        "p": np.random.randn(batch_size, 1, n_points, n_points, n_points),
    }
    # Export second example in batch
    grid_to_vtk(save_var, "./test_file", batch_index=1)


.. figure:: ../../images/user_guide/vtk_grid_data.png
    :alt: `vtkUniformGridData` visualization example
    :width: 60.0%
    :align: center
    
    Visualization of test_file.vti in ParaView


VTK Validator and Inferencer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PhysicsNeMo Sym also has a validator and inferencer node that builds from a VTK object directly called ``PointVTKValidator`` and ``PointVTKInferencer``.
These objects take one of PhysicsNeMo Sym built in VTK classes as an input and automatically queries the model at the point locations.
The advantage of these is that mesh data is kept in the validator/inferencer which is added into the output file.


Constructing VTK Objects from Scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first use case of this is to define your own VTK object from scratch in PhysicsNeMo Sym.
Consider adding a new inferencer to the :ref:`Introductory Example` example.
The example below defines a uniform mesh to conduct inference on:

.. code-block:: python

    from physicsnemo.sym.utils.io.vtk import VTKUniformGrid
    from physicsnemo.sym.domain.inferencer import PointVTKInferencer

    vtk_obj = VTKUniformGrid(
        bounds=[[-width / 2, width / 2], [-height / 2, height / 2]],
        npoints=[128, 128],
        export_map={"U": ["u", "v", None], "p": ["p"]},
    )
    grid_inference = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y"},
        output_names=["u", "v", "p"],
        requires_grad=False,
        batch_size=1024,
    )
    ldc_domain.add_inferencer(grid_inference, "vtk_inf")


``VTKUniformGrid`` is a PhysicsNeMo Sym wrapper for the ``vtkUniformGrid`` class and can be used to quickly define uniform domains.
The above example defines a square domain of resolution :math:`128\times 128`.
Adding this to your ``ldc_2d.py`` from :ref:`Introductory Example` will add an addition inferencer with and output file ``vtk_inf.vti`` which is visualized as a mesh rather than a point cloud.

.. figure:: ../../images/user_guide/vtk_ldc_grid_data.png
    :alt: `vtkUniformGridData` visualization LDC example
    :width: 60.0%
    :align: center
    
    Visualization of `vtk_inf.vti`` in ParaView from LDC inferencer

.. note::

    The ``export_map``, which is a dictionary, ``Dict[str, List[str]]`` used to map between VTK variable names and physicsnemo variable names.
    In this example the ``U`` field in the VTK file will contain PhysicsNeMo Sym variables ``u`` and ``v`` in the first and second dimension with zeros in the third.
 
.. note::

    ``input_vtk_map`` defines which parameters from the VTK object to use as model inputs. 
    This can be used to access point data arrays in the VTK file and also coordinates.


Reading VTK Objects from File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second and more powerful use case of these VTK inferencers/validators is the ability to load VTK meshes directly from file.
This means you can directly import testing data from a fluid simulation result and preserve the internal mesh data for visualization.
An example of reading in a OpenFOAM simulation file and using it for building a validator is shown below:

.. code-block:: python

    from physicsnemo.sym.utils.io.vtk import VTKFromFile
    from physicsnemo.sym.domain.validator import PointVTKValidator 

    vtk_obj = VTKFromFile(
        to_absolute_path("./openfoam/cavity_openfoam.vtk"), # Legacy VTK files supported
        export_map={"U_pred": ["u", "v", None]},
    )
    points = vtk_obj.get_points()
    points[:, 0] += -width / 2  # center OpenFoam data
    points[:, 1] += -height / 2  # center OpenFoam data
    vtk_obj.set_points(points)

    openfoam_validator = PointVTKValidator(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y"},
        true_vtk_map={"u": ["U:0"], "v": ["U:1"]},
        requires_grad=False,
        batch_size=1024,
    )
    ldc_domain.add_validator(openfoam_validator, "vtk_validator")

Since ``cavity_openfoam.vtk`` is an unstructured grid, the output from this validator would be ``vtk_validator.vtu`` and contain the same mesh structure.
Adding this code to your ``ldc_2d.py`` from :ref:`Introductory Example` will now produce a meshed validation result in ParaView.

.. figure:: ../../images/user_guide/vtk_ldc_validation_data.png
    :alt: `vtkUnstructuredGridData` visualization LDC example
    :width: 60.0%
    :align: center
    
    Visualization of `vtk_validator.vtu` in ParaView from LDC validator

.. note::

    The ``true_vtk_map`` tells PhysicsNeMo Sym what point fields to use as target values. 
    Here we are defining two target variables ``u`` and ``v`` which use the data in the first and second component of the field ``U`` in the VTK file.

.. warning::

    PhysicsNeMo Sym only supports the use of point data arrays in VTK objects.

This includes building validators/inferencers from more complex meshes as well. 
Even the results from a 2D system can be projected onto a 3D object using a VTK point inferencer. 
For example, you can download the `Stanford bunny <http://graphics.stanford.edu/data/3Dscanrep/>`_ and convert it into a VTK format in ParaView. This will allow you to then inference on this mesh.

.. code-block:: python

    from physicsnemo.sym.utils.io.vtk import VTKFromFile
    from physicsnemo.sym.domain.inferencer  import PointVTKInferencer 

    vtk_obj = VTKFromFile(
        to_absolute_path("./bunny.vtk"), # Legacy VTK files supported
        export_map={"U_pred": ["u", "v", None]},
    )

    openfoam_inferencer = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y"}, # Invariant to z location
        output_names=["u", "v", "p"],
        requires_grad=False,
        batch_size=1024,
    )
    ldc_domain.add_inferencer(openfoam_inferencer, "vtk_bunny")

With the VTK file ``bunny.vtk`` or any VTK unstructured mesh of your choosing, you can place this code into the lid driven cavity example.
The result is ``vtk_bunny.vtp``, shown below, which contains the result from querying the network at the mesh vertex points of the Stanford bunny.
While this is not a very practical result for the LDC flow, this illustrates how one can quickly load a predefined geometry and conduct inference on it.

.. figure:: ../../images/user_guide/vtk_ldc_bunny_data.png
    :alt: Bunny inference visualization LDC example
    :width: 60.0%
    :align: center
    
    Visualization of `vtk_bunny.vtp` in ParaView from LDC inferencer


Voxel Inferencer
^^^^^^^^^^^^^^^^

The ``VoxelInferencer`` is a unique class that can be particularly useful when you do not have a volume mesh of your geometry.
This includes cases when PhysicsNeMo Sym' geometry module is being used or you just have a mesh of the boundary.

The ``VoxelInferencer`` works by defining a uniform grid over a square domain.
A masking function, such as a SDF (Signed Distance Function), is provided which then flags which points lie inside the inference domain.
Masked points are set to ``NaN``, which can then be filtered out in ParaView. Below code shows how this can be used for the LDC example.

.. code-block:: python

    from physicsnemo.sym.domain.inferencer  import VoxelInferencer 

    # Define mask function, should be a callable with parameters being the variables
    mask_fn = lambda x, y: x**2 + y**2 > 0.001

    voxel_inferencer = VoxelInferencer(
        bounds = [[-width / 2, width / 2], [-height / 2, height / 2], [0, 0.1]],
        npoints = [128, 128, 128],
        nodes=nodes,
        output_names=["u", "v", "p"],
        export_map={"U": ["u", "v", None], "p": ["p"]},
        mask_fn = mask_fn,
        requires_grad=False,
        batch_size=1024,
    )
    ldc_domain.add_inferencer(voxel_inferencer, "vox_inf")

Here a unform grid of the resolution :math:`128\times 128\times 128` is used. 
The `mask_fn` defines which points should set to ``NaN`` and ignored during inference, in this case outside of a circle.
Adding this to ``ldc_2d.py`` will output the file ``vox_inf.vti``.
Initially upon loading this VTK file in ParaView, all masked and unmasked points will be shown.
Use the ``Threshold`` filter on the default settings to remove the masked points leaving a nice cylinder.

.. figure:: ../../images/user_guide/vtk_ldc_cylinder_data.png
    :alt: Voxel inference visualization LDC example
    :width: 60.0%
    :align: center
    
    Visualization of `vox_inf.vti` in ParaView from LDC inferencer


.. figure:: ../../images/user_guide/vtk_ldc_cylinder_masked_data.png
    :alt: Masked voxel inference visualization LDC example
    :width: 60.0%
    :align: center
    
    Visualization of `vox_inf.vti` with threshold filter in ParaView from LDC inferencer

.. note::

    ``PointVTKInferencer`` also supports the use of mask functions and can be combined with ``VTKUniformGrid`` to achieve the same result.
    Examples such as :ref:`stl` and :ref:`limerock` do this to inference their complex domains at a specific resolution.
