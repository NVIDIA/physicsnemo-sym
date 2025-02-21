********************************************
NVIDIA PhysicsNeMo Sym (Latest Release)
********************************************
..
   TODO: add conf.py and root_doc

NVIDIA PhysicsNeMo Sym is a deep learning framework that blends the power of 
physics and partial differential equations (PDEs) with AI to build more 
robust models for better analysis. 

There is a plethora of ways in which ML/NN models can be applied for 
physics-based systems. These can depend based on the availability of 
observational data and the extent of understanding of underlying physics. 
Based on these aspects, the ML/NN based methodologies can be broadly 
classified into forward (physics-driven), data-driven and hybrid approaches 
that involve both the physics and data assimilation. 

.. figure:: /images/user_guide/ai_in_computational_sciences_spectrum.png
   :alt: AI in computational sciences
   :width: 80.0%
   :align: center

With NVIDIA PhysicsNeMo Sym, we aim to provide researchers and industry specialists, 
various tools that will help accelerate your development of such models for the 
scientific discipline of your need. Experienced users can start with exploring the 
PhysicsNeMo Sym APIs and building the models while beginners can use this User Guide 
as a portal to explore the possibilities of AI in the domain of scientific 
computation. The User Guide comes in with several examples that will help 
you jumpstart your development of AI driven models.

For Beginners
-------------
New to PhysicsNeMo Sym? No worries, we will get you up and running with physics driven AI in a few minutes. 
Check out the following tutorials to get started. 

.. raw:: html
   :file: _static/beginner_landing_buttons.html

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :name: Getting Started
   :hidden:

   Table of Contents <user_guide/getting_started/toc.rst>

.. toctree::
   :maxdepth: 2
   :caption: Learn the Basics
   :name: Learn the Basics
   :hidden:

   Overview <user_guide/basics/physicsnemo_overview.rst>
   Introductory Example <user_guide/basics/lid_driven_cavity_flow.rst>

.. toctree::
   :maxdepth: 1
   :caption: Theory
   :name: Theory
   :hidden:

   Physics-Informed Learning <user_guide/theory/phys_informed.rst>
   Architectures <user_guide/theory/architectures.rst>
   Advanced Schemes <user_guide/theory/advanced_schemes.rst>
   Recommended Practices <user_guide/theory/recommended_practices.rst>
   Miscellaneous Concepts <user_guide/theory/miscellaneous.rst>

.. toctree::
   :maxdepth: 1
   :caption: PhysicsNeMo Sym Features
   :name: PhysicsNeMo Sym Features
   :hidden:
   
   Geometry and Tesselation Modules <user_guide/features/csg_and_tessellated_module.rst>
   Computational Graph, Nodes and Architectures <user_guide/features/nodes.rst> 
   Constraints <user_guide/features/constraints.rst>
   Configuration <user_guide/features/configuration.rst>
   Post Processing <user_guide/features/post_processing.rst>
   Performance <user_guide/features/performance.rst>

.. toctree::
   :maxdepth: 1
   :caption: Physics-Informed Foundations
   :name: Physics-Informed Foundations
   :hidden:
 
   1D Wave Equation <user_guide/foundational/1d_wave_equation.rst>
   2D Wave Equation <user_guide/foundational/2d_wave_equation.rst>
   Spring Mass ODE <user_guide/foundational/ode_spring_mass.rst>
   Zero Equation Turbulence <user_guide/foundational/zero_eq_turbulence.rst>
   Scalar Transport <user_guide/foundational/scalar_transport.rst>
   Linear Elasticity <user_guide/foundational/linear_elasticity.rst>
   Inverse Problem <user_guide/foundational/inverse_problem.rst>

.. toctree::
   :maxdepth: 1
   :caption: Neural Operators
   :name: Neural Operators
   :hidden:
 
   Fourier <user_guide/neural_operators/darcy_fno.rst>
   Adaptive Fourier <user_guide/neural_operators/darcy_afno.rst>
   Physics-Informed <user_guide/neural_operators/darcy_pino.rst>
   Deep-O Nets <user_guide/neural_operators/deeponet.rst>
   FourCastNet <user_guide/neural_operators/fourcastnet.rst>
   
.. toctree::
   :maxdepth: 1
   :caption: Intermediate Case Studies
   :name: Intermediate Case Studies
   :hidden:
 
   Variational Examples <user_guide/intermediate/variational_example.rst>
   Geometry from STL Files <user_guide/intermediate/adding_stl_files.rst>
   Time Window Training <user_guide/intermediate/moving_time_window.rst>
   Electromagnetics <user_guide/intermediate/em.rst>
   2D Turbulent Channel <user_guide/intermediate/two_equation_turbulent_channel.rst>
   Turbulence Super Resolution <user_guide/intermediate/turbulence_super_resolution.rst>

.. toctree::
   :maxdepth: 1
   :caption: Advanced Case Studies
   :name: Advanced Case Studies
   :hidden:
 
   Conjugate Heat Transfer <user_guide/advanced/conjugate_heat_transfer.rst>
   3D Fin Parameterization <user_guide/advanced/parametrized_simulations.rst>
   Heat Transfer with High Conductivity <user_guide/advanced/2d_heat_transfer.rst>
   FPGA <user_guide/advanced/fpga.rst>
   Industrial Heat Sink <user_guide/advanced/industrial_heat_sink.rst>


.. toctree::
   :maxdepth: 2
   :caption: PhysicsNeMo Sym API
   :name: PhysicsNeMo Sym API
   :hidden:
   
   api/api_index
