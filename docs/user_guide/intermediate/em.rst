Electromagnetics: Frequency Domain Maxwell's Equation
=======================================================

Introduction
------------

This tutorial demonstrates how to use PhysicsNeMo Sym to do the
electromagnetic (EM) simulation. Currently, PhysicsNeMo Sym offers the following
features for frequency domain EM simulation:

#. Frequency domain Maxwell's equation in scalar form. This is same to
   Helmholtz equation. (This is available for 1D, 2D, and 3D. Only real
   form is available now.)

#. Frequency domain Maxwell's equation in vector form. (This is
   available for 3D case and only real form is available now.)

#. Perfect electronic conductor (PEC) boundary conditions for 2D and 3D
   cases.

#. Radiation boundary condition (or, absorbing boundary condition) for
   3D.

#. 1D waveguide port solver for 2D waveguide source.

Two electromagnetic problems are solved in this tutorial. All the
simulations are appropriately nondimensionalized.

.. note::
   This tutorial assumes that you have completed the tutorial :ref:`Introductory Example` and are
   familiar with PhysicsNeMo Sym APIs

   All the scripts referred in this tutorial can be found in ``examples/waveguide/``.

Problem 1: 2D Waveguide Cavity
------------------------------

Consider a 2D domain :math:`\Omega=[0, 2]\times [0, 2]` as shown
below. The whole domain is vacuum. Say,
relative permittivity :math:`\epsilon_r = 1`. The left boundary is a
waveguide port while the right boundary is absorbing boundary (or ABC).
The top and the bottom is PEC.

.. figure:: ../../images/user_guide/2Dwaveguide.png
   :alt: Domain of 2D waveguide
   :name: fig:2Dwaveguide
   :width: 30.0%
   :align: center

   Domain of 2D waveguide

In this example, the waveguide problem is solved by
transverse magnetic (:math:`TM_z`) mode, so that the unknown variable is
:math:`E_z(x,y)`. The governing equation in :math:`\Omega` is

.. math:: \Delta E_z(x,y) + k^2E_z(x,y) = 0,

where :math:`k` is the wavenumber. Notice in 2D scalar case, the PEC and
ABC will be simplified in the following form, respectively:

.. math:: E_z(x,y)=0\mbox{ on top and bottom boundaries, }\frac{\partial E_z}{\partial y}=0\mbox{ on right boundary.}

Case Setup
~~~~~~~~~~

This subsection shows how to use PhysicsNeMo Sym to setup the EM
solver. Similar to the previous tutorials, you will first import the necessary
libraries.

.. literalinclude:: ../../../examples/waveguide/cavity_2D/waveguide2D_TMz.py
   :language: python
   :lines: 15-35

Then, define the variables for ``sympy`` symbolic calculation and parameters for geometry.
Also, before you define the main classes for PhysicsNeMo Sym, you need to compute
the eigenmode for waveguide solver. Since the material is uniform
(vacuum), the closed form of the eigenmode is of the form
:math:`\sin(\frac{k\pi y}{L})`, where :math:`L` is the length of the
port, and :math:`k = 1, 2,\cdots`. Then define the waveguide port
directly by using ``sympy`` function. The code for the geometry and computing eigenmode can be found below.

.. literalinclude:: ../../../examples/waveguide/cavity_2D/waveguide2D_TMz.py
   :language: python
   :lines: 42-53


For wave simulation, since the result is always periodic, Fourier
feature will be greatly helpful for the convergence and accuracy. The
frequency of the Fourier feature can be implied by the wavenumber. This
block of code shows the solver setup.
Also, define the normal gradient for the boundary conditions. Finally, make the domain for training.

.. literalinclude:: ../../../examples/waveguide/cavity_2D/waveguide2D_TMz.py
   :language: python
   :lines: 54-74

Now, define the constraints for PDEs and boundary conditions. The BCs are defined based on the
explanations provided above. In the interior domain,
the weights of the PDE is ``1.0/wave_number**2``. This is because when the
wavenumber is large, the PDE loss will be very large in the beginning
and will potentially break the training. Using this
weighting method you can eliminate this phenomenon.

.. literalinclude:: ../../../examples/waveguide/cavity_2D/waveguide2D_TMz.py
   :language: python
   :lines: 76-117

To validate the result, you can import the ``csv`` files for the validation
domain below.

.. literalinclude:: ../../../examples/waveguide/cavity_2D/waveguide2D_TMz.py
   :language: python
   :lines: 122-138

Inferencer has been implemented to plot the results.

.. literalinclude:: ../../../examples/waveguide/cavity_2D/waveguide2D_TMz.py
   :language: python
   :lines: 140-148

Results
~~~~~~~

The full code of this example can be found in
``examples/waveguide/cavity_2D/waveguide2D_TMz.py``. The
simulation with wavenumber equals :math:`32`. The solution from comercial solver, PhysicsNeMo Sym prediction, and their difference
are shown below.

.. figure:: ../../images/user_guide/2Dwaveguide_physicsnemo.png
   :alt: PhysicsNeMo Sym, wavenumber=\ :math:`32`
   :name: fig:2Dwaveguide_physicsnemo
   :width: 100.0%
   :align: center

   PhysicsNeMo Sym, wavenumber=\ :math:`32`

Problem 2: 2D Dielectric slab waveguide
---------------------------------------

This section, demonstrates using the 2D waveguide simulation with a dielectric
slab. The problem setup is almost same as before except there is a
horizontal dielectric slab in the middle of the domain. The domain is
shown below.

.. figure:: ../../images/user_guide/2Dslab_geo.png
   :alt: Domain of 2D Dielectric slab waveguide
   :name: fig:2Dslab_geo
   :width: 30.0%
   :align: center

   Domain of 2D Dielectric slab waveguide

In the dielectric, set the relative permittivity
:math:`\epsilon_r=2`. That is,

.. math::

   \epsilon_r = 
   \begin{cases}
   2   & \mbox{ in dielectric slab,}\\
   1   & \mbox{ otherwise.}
   \end{cases}

All the other settings are kept the same to the previous example.

Case setup
~~~~~~~~~~

Here, for simplicity, only the parts of the code that are
different than the previous example are shown. The main difference is the
spatially dependent permittivity. First, compute the
eigenfunctions on the left boundary.

.. literalinclude:: ../../../examples/waveguide/slab_2D/slab_2D.py
   :language: python
   :lines: 81-86

For the geometry part, you will need to define the slab and corresponding
permittivity function. There is a square root in ``eps_sympy`` because in the
``HelmholtzEquation``, the wavenumber will be squared. Next, based on
the permittivity function, use the eigensolver to get the numerical
waveguide port.


.. literalinclude:: ../../../examples/waveguide/slab_2D/slab_2D.py
   :language: python
   :lines: 61-80

In the definition of the PDEs and neural network's structure,
set the ``k`` in ``HelmholtzEquation`` as the product of wavenumber and
permittivity function. Also, update the frequency for the Fourier
features to suit the problem.

.. literalinclude:: ../../../examples/waveguide/slab_2D/slab_2D.py
   :language: python
   :lines: 93-108


Now, define the constraints. The only difference here is the
left boundary, which will be given by a ``numpy`` array. Only
the modified BC is shown below:

.. literalinclude:: ../../../examples/waveguide/slab_2D/slab_2D.py
   :language: python
   :lines: 123-130

Results
~~~~~~~

The full code of this example can be found in
``examples/waveguide/slab_2D/slab_2D.py``. We do the
simulation with wavenumber equals :math:`16` and :math:`32`,
respectively. The results are shown in figure below

.. figure:: ../../images/user_guide/2Dslab_16.png
   :alt: PhysicsNeMo Sym, wavenumber=\ :math:`16`
   :name: fig:2Dslab
   :width: 50.0%
   :align: center

   PhysicsNeMo Sym, wavenumber=\ :math:`16`


Problem 3: 3D waveguide cavity
------------------------------

This example, shows how to setup a 3D waveguide simulation in
PhysicsNeMo Sym. Unlike the previous examples, the features in PhysicsNeMo Sym
to define the boundary condition are used. The geometry is
:math:`\Omega = [0,2]^3`, as shown below.

Problem setup
~~~~~~~~~~~~~

.. figure:: ../../images/user_guide/3Dwaveguide_geo.png
   :alt: 3D waveguide geometry
   :name: fig:3Dwaveguide_geo
   :width: 50.0%
   :align: center

   3D waveguide geometry

We will solve the 3D frequency domain Maxwell's equation for electronic
field :math:`\mathbf{E}=(E_x, E_y, E_z)`:

.. math:: \nabla\times \nabla\times \mathbf{E}+\epsilon_rk^2\mathbf{E} = 0,

where :math:`\epsilon_r` is the permittivity, and the :math:`k` is the
wavenumber. Note that, currently PhysicsNeMo Sym only support real permittivity
and wavenumber. For the sake of simplicity, assume the permeability
:math:`\mu_r=1`. As before, waveguide port has been applied on the left.
We apply absorbing boundary condition on the right side and PEC for the
rest. In 3D, the absorbing boundary condition for real form reads

.. math:: \mathbf{n}\times\nabla\times \mathbf{E} = 0,

while the PEC is

.. math:: \mathbf{n}\times \mathbf{E} = 0.

Case setup
~~~~~~~~~~

This section shows how to use PhysicsNeMo Sym to setup the 3D
frequency EM solver, especially for the boundary conditions.

First import the necessary libraries.

.. literalinclude:: ../../../examples/waveguide/cavity_3D/waveguide3D.py
   :language: python
   :lines: 15-31

Define ``sympy`` variables, geometry and waveguide
function.

.. literalinclude:: ../../../examples/waveguide/cavity_3D/waveguide3D.py
   :language: python
   :lines: 38-50

Define the PDE class and neural network structure.

.. literalinclude:: ../../../examples/waveguide/cavity_3D/waveguide3D.py
   :language: python
   :lines: 51-70

Then, define the constraints for PDEs and boundary conditions, and all boundary
conditions. The 3D Maxwell's equations has been implemented in
``Maxwell_Freq_real_3D``, PEC has been implemented in ``PEC_3D``, and
absorbing boundary condition has been implemented in
``SommerfeldBC_real_3D``. We may use these features directly to apply
the corresponding constraints.

.. literalinclude:: ../../../examples/waveguide/cavity_3D/waveguide3D.py
   :language: python
   :lines: 72-130


Note that this is done in 3D, so the PDEs, PEC and absorbing
boundaries have three output components.

Inferencer domain has been defined to check the result.

.. literalinclude:: ../../../examples/waveguide/cavity_3D/waveguide3D.py
   :language: python
   :lines: 132-143


Results
~~~~~~~

The full code of this example can be found in
``examples/waveguide/waveguide3D.py``. The wavenumber equals
:math:`32` and use second eigenmode for :math:`y` and :math:`z`. The
slices of the three components are shown in below.

.. figure:: ../../images/user_guide/3Dwaveguide_Ex.png
   :alt: 3D waveguide, :math:`E_x`
   :name: fig:3Dwaveguide_ex
   :width: 50.0%
   :align: center

   3D waveguide, :math:`E_x`

.. figure:: ../../images/user_guide/3Dwaveguide_Ey.png
   :alt: 3D waveguide, :math:`E_y`
   :name: fig:3Dwaveguide_ey
   :width: 50.0%
   :align: center

   3D waveguide, :math:`E_y`

.. figure:: ../../images/user_guide/3Dwaveguide_Ez.png
   :alt: 3D waveguide, :math:`E_z`
   :name: fig:3Dwaveguide_ez
   :width: 50.0%
   :align: center

   3D waveguide, :math:`E_z`

Problem 4: 3D Dielectric slab waveguide
---------------------------------------

This example, shows a 3D dielectric slab waveguide. In this
case, consider a unit cube :math:`[0,1]^3` with a dielectric slab
centered in the middle along :math:`y` axis. The length of the slab is
:math:`0.2`. :numref:`fig-3Dslab_geo` and :numref:`fig-3Dslab_geo_cross` shows the
whole geometry and an :math:`xz` cross-section that shows the dielectric
slab.

.. _fig-3Dslab_geo:

.. figure:: ../../images/user_guide/3Dslab_geo.png
   :alt: 3D view with BCs
   :name: fig:3Dslab_geo
   :width: 80.0%
   :align: center

   Geometry for 3D dielectric slab waveguide

.. _fig-3Dslab_geo_cross:

.. figure:: ../../images/user_guide/3Dslab_geo_xz.png
   :alt: :math:`xz` cross-sectional view
   :name: fig:3Dslab_geo_cross
   :width: 50.0%
   :align: center

   :math:`xz` cross-sectional view

The permittivity is defined as follows

.. math::

   \epsilon_r = 
   \begin{cases}
   1.5   & \mbox{ in dielectric slab,}\\
   1   & \mbox{ otherwise.}
   \end{cases}


Case setup
~~~~~~~~~~

For the sake of simplicity, only the differences compared to the
last example are covered here. The main difference for this simulation is that you
need to import the waveguide result from a ``csv`` file and then use
that as the waveguide port boundary condition.

First define the geometry and the ``sympy`` permittivity
function. To define the piecewise ``sympy`` functions, use
``Heaviside`` instead of ``Piecewise`` as the later cannot be complied
in PhysicsNeMo Sym for the time being. The waveguide data can also be imported using the ``csv_to_dict()`` function.


.. literalinclude:: ../../../examples/waveguide/slab_3D/slab_3D.py
   :language: python
   :lines: 42-79


In ``validation/2Dwaveguideport.csv``, there are six eigenmodes. You may
try different modes to explore more interesting results.

Next, define the PDEs classes and neural network structure.

.. literalinclude:: ../../../examples/waveguide/slab_3D/slab_3D.py
   :language: python
   :lines: 80-102

Then, define the constraints. Here imported
data is used as the waveguide port boundary condition.

.. literalinclude:: ../../../examples/waveguide/slab_3D/slab_3D.py
   :language: python
   :lines: 104-162

Finally, define the inferencer. These are
same as the previous example except the ``bounds`` for the domain.

.. literalinclude:: ../../../examples/waveguide/slab_3D/slab_3D.py
   :language: python
   :lines: 164-171

Results
~~~~~~~

The full code of this example can be found in
``examples/waveguide/slab_3D/slab_3D.py``. Here the
simulation for different wavenumbers is done. Below figures show the result for wavenumber equals :math:`16`.


.. figure:: ../../images/user_guide/3Dslab_16_Ex.png
   :alt: 3D dielectric slab, :math:`E_x`
   :name: fig:3Dslab_16ex
   :width: 50.0%
   :align: center

   3D dielectric slab, :math:`E_x`

.. figure:: ../../images/user_guide/3Dslab_16_Ey.png
   :alt: 3D dielectric slab, :math:`E_y`
   :name: fig:3Dslab_16ey
   :width: 50.0%
   :align: center

   3D dielectric slab, :math:`E_y`

.. figure:: ../../images/user_guide/3Dslab_16_Ez.png
   :alt: 3D dielectric slab, :math:`E_z`
   :name: fig:3Dslab_16ez
   :width: 50.0%
   :align: center

   3D dielectric slab, :math:`E_z`

Also the results of higher wavenumber :math:`32` are shown below.


.. figure:: ../../images/user_guide/3Dslab_32_Ex.png
   :alt: 3D dielectric slab, :math:`E_x`
   :name: fig:3Dslab_32ex
   :width: 50.0%
   :align: center

   3D dielectric slab, :math:`E_x`

.. figure:: ../../images/user_guide/3Dslab_32_Ey.png
   :alt: 3D dielectric slab, :math:`E_y`
   :name: fig:3Dslab_32ey
   :width: 50.0%
   :align: center

   3D dielectric slab, :math:`E_y`

.. figure:: ../../images/user_guide/3Dslab_32_Ez.png
   :alt: 3D dielectric slab, :math:`E_z`
   :name: fig:3Dslab_32ez
   :width: 50.0%
   :align: center

   3D dielectric slab, :math:`E_z`
