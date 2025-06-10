.. _geometry_and_tesselated:

Geometry Modules
================

Constructive Solid Geometry
---------------------------

PhysicsNeMo Sym provides several 1D, 2D and 3D primitives that can be used for sampling point clouds required for the physics-informed training. 
These primitives support the standard boolean operations* like Union (``+`` ), Intersection (``&`` ) and Subtraction (``-`` ). The boolean
operations work on the signed distance fields of the differnt primitives. 

Below example shows a simple CSG primitive being built using PhysicsNeMo Sym. 

.. code-block:: python
    :caption: Constructive solid geometry
        
    import numpy as np
    from physicsnemo.sym.geometry.primitives_3d import Box, Sphere, Cylinder
    from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
    
    # number of points to sample
    nr_points = 100000
    
    # make standard constructive solid geometry example
    # make primitives
    box = Box(point_1=(-1, -1, -1), point_2=(1, 1, 1))
    sphere = Sphere(center=(0, 0, 0), radius=1.2)
    cylinder_1 = Cylinder(center=(0, 0, 0), radius=0.5, height=2)
    cylinder_2 = cylinder_1.rotate(angle=float(np.pi / 2.0), axis="x")
    cylinder_3 = cylinder_1.rotate(angle=float(np.pi / 2.0), axis="y")
    
    # combine with boolean operations
    all_cylinders = cylinder_1 + cylinder_2 + cylinder_3
    box_minus_sphere = box & sphere
    geo = box_minus_sphere - all_cylinders
    
    # sample geometry for plotting in Paraview
    s = geo.sample_boundary(nr_points=nr_points)
    var_to_polyvtk(s, "boundary")
    print("Surface Area: {:.3f}".format(np.sum(s["area"])))
    s = geo.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
    var_to_polyvtk(s, "interior")
    print("Volume: {:.3f}".format(np.sum(s["area"])))
        

.. figure:: /images/user_guide/csg_demo.png
   :alt: Constructive Solid Geometry using PhysicsNeMo Sym primitives
   :width: 80.0%
   :align: center

   Constructive Solid Geometry using PhysicsNeMo Sym primitives

A complete list of primitives can be referred in ``physicsnemo.geometry.primitives_*`` 

.. note::
    While generating a complex primitive, it should be noted that the boolean operation are performed at the final stage, meaning it is invariant to 
    the order of the boolean operations. In other words, if you are subtracting a primitive from another primitive, and if you decide to add a different primitive
    in the area that is subtracted, you will not see the newly added primitive because the subtracted primitive.

The geometry objects created also support operations like ``translate``, ``scale``, ``rotate`` and ``repeat`` to further generate more complicated primitives

.. code-block:: python
    :caption: Transforms and Repeat

    # apply transformations
    geo = geo.scale(0.5)
    geo = geo.rotate(angle=np.pi / 4, axis="z")
    geo = geo.rotate(angle=np.pi / 4, axis="y")
    geo = geo.repeat(spacing=4.0, repeat_lower=(-1, -1, -1), repeat_higher=(1, 1, 1))
    
    # sample geometry for plotting in Paraview
    s = geo.sample_boundary(nr_points=nr_points)
    var_to_polyvtk(s, "repeated_boundary")
    print("Repeated Surface Area: {:.3f}".format(np.sum(s["area"])))
    s = geo.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
    var_to_polyvtk(s, "repeated_interior")
    print("Repeated Volume: {:.3f}".format(np.sum(s["area"])))

.. figure:: /images/user_guide/csg_repeat_demo.png
   :alt: Geometry transforms
   :width: 80.0%
   :align: center

   Geometry transforms


The CSG objects can be easily parameterized using sympy. An example of this is used in :ref:`ParameterizedSim`

.. code-block:: python
    :caption: Parameterized geometry

    from physicsnemo.sym.geometry.primitives_2d import Rectangle, Circle
    from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
    from physicsnemo.sym.geometry.parameterization import Parameterization, Parameter
    
    # make plate with parameterized hole
    # make parameterized primitives
    plate = Rectangle(point_1=(-1, -1), point_2=(1, 1))
    y_pos = Parameter("y_pos")
    parameterization = Parameterization({y_pos: (-1, 1)})
    circle = Circle(center=(0, y_pos), radius=0.3, parameterization=parameterization)
    geo = plate - circle
    
    # sample geometry over entire parameter range
    s = geo.sample_boundary(nr_points=100000)
    var_to_polyvtk(s, "parameterized_boundary")
    s = geo.sample_interior(nr_points=100000)
    var_to_polyvtk(s, "parameterized_interior")
    
    # sample specific parameter
    s = geo.sample_boundary(
        nr_points=100000, parameterization=Parameterization({y_pos: 0})
    )
    var_to_polyvtk(s, "y_pos_zero_boundary")
    s = geo.sample_interior(
        nr_points=100000, parameterization=Parameterization({y_pos: 0})
    )
    var_to_polyvtk(s, "y_pos_zero_interior")
    

.. figure:: /images/user_guide/csg_parameterized_demo.png
   :alt: Parameterized Constructive Solid Geometry using PhysicsNeMo Sym primitives
   :width: 80.0%
   :align: center

   Parameterized Constructive Solid Geometry using PhysicsNeMo Sym primitives


Some interesting shapes generated using PhysicsNeMo Sym' geometry module are presented below

.. figure:: /images/user_guide/naca_demo.png
   :alt: NACA airfoil using ``Polygon`` primitive
   :width: 80.0%
   :align: center

   NACA airfoil using ``Polygon`` primitive. (script at ``/examples/geometry/naca_airfoil.py``)

.. figure:: /images/user_guide/ahmed_demo.png
   :alt: Ahmed body
   :width: 80.0%
   :align: center

   Ahmed body

.. figure:: /images/user_guide/industrial_geo_demo.png
   :alt: Industrial heatsink geometry
   :width: 80.0%
   :align: center

   Industrial heatsink geometry


Defining Custom Primitives
--------------------------

If you don't find a primitive defined for your application, it is easy to setup using the base classes from PhysicsNeMo Sym. All you need to do is come up with and appropriate
expression for the signed distance field and the surfaces of the geometry. An example is shown below. 

.. code-block:: python
    :caption: Custom Primitive

    from sympy import Symbol, pi, sin, cos, sqrt, Min, Max, Abs
    
    from physicsnemo.sym.geometry.geometry import Geometry, csg_curve_naming
    from physicsnemo.sym.geometry.helper import _sympy_sdf_to_sdf
    from physicsnemo.sym.geometry.curve import SympyCurve, Curve
    from physicsnemo.sym.geometry.parameterization import Parameterization, Parameter, Bounds
    from physicsnemo.sym.geometry.primitives_3d import Cylinder
    from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
    
    class InfiniteCylinder(Geometry):
        """
        3D Infinite Cylinder
        Axis parallel to z-axis, no caps on ends
    
        Parameters
        ----------
        center : tuple with 3 ints or floats
            center of cylinder
        radius : int or float
            radius of cylinder
        height : int or float
            height of cylinder
        parameterization : Parameterization
            Parameterization of geometry.
        """
    
        def __init__(self, center, radius, height, parameterization=Parameterization()):
            # make sympy symbols to use
            x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
            h, r = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
            theta = Symbol(csg_curve_naming(2))
    
            # surface of the cylinder
            curve_parameterization = Parameterization(
                {h: (-1, 1), r: (0, 1), theta: (0, 2 * pi)}
            )
            curve_parameterization = Parameterization.combine(
                curve_parameterization, parameterization
            )
            curve_1 = SympyCurve(
                functions={
                    "x": center[0] + radius * cos(theta),
                    "y": center[1] + radius * sin(theta),
                    "z": center[2] + 0.5 * h * height,
                    "normal_x": 1 * cos(theta),
                    "normal_y": 1 * sin(theta),
                    "normal_z": 0,
                },
                parameterization=curve_parameterization,
                area=height * 2 * pi * radius,
            )
            curves = [curve_1]
    
            # calculate SDF
            r_dist = sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            sdf = radius - r_dist
    
            # calculate bounds
            bounds = Bounds(
                {
                    Parameter("x"): (center[0] - radius, center[0] + radius),
                    Parameter("y"): (center[1] - radius, center[1] + radius),
                    Parameter("z"): (center[2] - height / 2, center[2] + height / 2),
                },
                parameterization=parameterization,
            )
    
            # initialize Infinite Cylinder
            super().__init__(
                curves,
                _sympy_sdf_to_sdf(sdf),
                dims=3,
                bounds=bounds,
                parameterization=parameterization,
            )
    
    
    nr_points = 100000
    
    cylinder_1 = Cylinder(center=(0, 0, 0), radius=0.5, height=2)
    cylinder_2 = InfiniteCylinder(center=(0, 0, 0), radius=0.5, height=2)
    
    s = cylinder_1.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
    var_to_polyvtk(s, "interior_cylinder")
    
    s = cylinder_2.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
    var_to_polyvtk(s, "interior_infinite_cylinder")
        

.. figure:: /images/user_guide/custom_primitive_demo.png
   :alt: Custom primitive in PhysicsNeMo Sym. The cylinders are sliced to visualize the interior SDF
   :width: 80.0%
   :align: center

   Custom primitive in PhysicsNeMo Sym. The cylinders are sliced to visualize the interior SDF

   
Tesselated Geometry
-------------------

For more complicated shapes, PhysicsNeMo Sym allows geometries to be imported in the STL format. The module uses ray tracing to compute SDF and its derivatives.
The module also gives surface normals of the geometry for surface sampling. Once the geometry is imported, the point cloud can be used for training. An
example of this can be found in :ref:`stl`.

Tesselated geometries can also be combined with the primitives 

.. code-block:: python
    :caption: Tesselated Geometry

    import numpy as np
    from physicsnemo.sym.geometry.tessellation import Tessellation
    from physicsnemo.sym.geometry.primitives_3d import Plane
    from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
    
    # number of points to sample
    nr_points = 100000
    
    # make tesselated geometry from stl file
    geo = Tessellation.from_stl("./stl_files/tessellated_example.stl")
    
    # tesselated geometries can be combined with primitives
    cut_plane = Plane((0, -1, -1), (0, 1, 1))
    geo = geo & cut_plane
    
    # sample geometry for plotting in Paraview
    s = geo.sample_boundary(nr_points=nr_points)
    var_to_polyvtk(s, "tessellated_boundary")
    print("Repeated Surface Area: {:.3f}".format(np.sum(s["area"])))
    s = geo.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
    var_to_polyvtk(s, "tessellated_interior")
    print("Repeated Volume: {:.3f}".format(np.sum(s["area"])))
        

.. figure:: /images/user_guide/stl_demo_1.png
   :alt: Tesselated Geometry sampling using PhysicsNeMo Sym 
   :width: 80.0%
   :align: center

   Tesselated Geometry sampling using PhysicsNeMo Sym

.. figure:: /images/user_guide/stl_demo_2.png
   :alt: Tesselated Geometry sampling using PhysicsNeMo Sym: Stanford bunny
   :width: 80.0%
   :align: center

   Tesselated Geometry sampling using PhysicsNeMo Sym: Stanford bunny


Signed Distance Fields (SDF) of Geometry objects
-------------------------------------------------

Mathematically, signed distance field or signed distance function (SDF) is defined as
the orthonogal distance of a given point to the boundary / surface of a geometric shape.
It is widely used to describe the geometry in mathematics, rendering, and similar
applications. In physics-informed learning, it is also used to represent
`geometric inputs to the neural networks <https://www.research.autodesk.com/app/uploads/2023/03/convolutional-neural-networks-for.pdf_rectr0tDKzFYVAAJe.pdf>`_.

PhysicsNeMo' geometry module (CSG and Tesselation) computes the SDF (and its derivatives) on
points sampled in the interior for use in the training pipelines.
Additionally, the SDF can be computed on custom points using the ``.sdf`` attribute.
 
PhysicsNeMo also provides a utility to recover the STL geometry from the SDF using marching
cubes algorithm. For more details refer
`here <https://github.com/NVIDIA/physicsnemo/blob/main/physicsnemo/utils/mesh/generate_stl.py#L25>`_.

Below example shows the use of these utilities for a CSG geometry.

.. code-block:: python
    :caption: Computing SDF for CSG geometry and reconstructing the STL

    import numpy as np
    from physicsnemo.sym.geometry.primitives_3d import Box, Sphere, Cylinder

    from physicsnemo.utils.mesh import sdf_to_stl

    # make standard constructive solid geometry example
    box = Box(point_1=(-1, -1, -1), point_2=(1, 1, 1))
    sphere = Sphere(center=(0, 0, 0), radius=1.2)
    cylinder_1 = Cylinder(center=(0, 0, 0), radius=0.5, height=2)
    cylinder_2 = cylinder_1.rotate(angle=float(np.pi / 2.0), axis="x")
    cylinder_3 = cylinder_1.rotate(angle=float(np.pi / 2.0), axis="y")

    all_cylinders = cylinder_1 + cylinder_2 + cylinder_3
    box_minus_sphere = box & sphere
    geo = box_minus_sphere - all_cylinders

    # generate cartesian grid to compute SDF
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    z = np.linspace(-2, 2, 100)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # compute the SDF on the grid points
    sdf = geo.sdf(
            {
                "x": xx.reshape(-1, 1), 
                "y": yy.reshape(-1, 1), 
                "z": zz.reshape(-1, 1),
            }, 
        params={}
    )["sdf"]

    # reconstruct the STL from SDF
    sdf_to_stl(
        sdf.reshape(100, 100, 100),    # sdf field in [nx, ny, nz] shape
        threshold=0.0,
        backend="skimage",  # skimage backend works better for CSG geometry
        filename="csg_reconstruction.stl"
    )


.. figure:: /images/user_guide/sdf_to_stl_csg.png
   :alt: Computing SDF for CSG geometry and reconstructing the STL
   :width: 80.0%
   :align: center

   Computing SDF for CSG geometry and reconstructing the STL


Below example shows the use of these utilities for a Tesselation geometry.

.. code-block:: python
    :caption: Computing SDF for Tesselation geometry and reconstructing the STL

    import numpy as np
    from physicsnemo.sym.geometry.tessellation import Tessellation

    from physicsnemo.utils.mesh import sdf_to_stl

    # read the Stanford Bunny stl
    geo = Tessellation.from_stl("./Stanford_Bunny.stl")

    # generate cartesian grid to compute SDF
    bounds = {str(k): v for k, v in geo.bounds.bound_ranges.items()}
    x = np.linspace(bounds["x"][0] - 1, bounds["x"][1] + 1, 100)
    y = np.linspace(bounds["y"][0] - 1, bounds["y"][1] + 1, 100)
    z = np.linspace(bounds["z"][0] - 1, bounds["z"][1] + 1, 100)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # compute the SDF on the grid points
    sdf = geo.sdf(
            {
                "x": xx.reshape(-1, 1), 
                "y": yy.reshape(-1, 1), 
                "z": zz.reshape(-1, 1),
            }, 
        params={}
    )["sdf"]
    
    # reconstruct the STL from SDF
    sdf_to_stl(
        sdf.reshape(100, 100, 100),    # sdf field in [nx, ny, nz] shape
        threshold=0.0,
        backend="warp",  # warp backend works better and is faster for STL geometry
        filename="stl_reconstruction.stl"
    )


.. figure:: /images/user_guide/sdf_to_stl_tesselation.png
   :alt: Computing SDF for Tesselation geometry and reconstructing the STL
   :width: 80.0%
   :align: center

   Computing SDF for Tesselation geometry and reconstructing the STL
