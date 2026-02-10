.. _2d_heat:

Heat Transfer with High Thermal Conductivity
============================================

Introduction
------------

This tutorial discusses strategies that can be employed for handling conjugate heat transfer problems with higher thermal conductivities that represent more realistic materials. 
The :ref:`cht` tutorial introduced how you can setup a simple conjugate heat transfer problem in PhysicsNeMo Sym. However, the thermal properties in that example do not represent realistic material properties 
used to manufacture a heatsink or to cool one. Usually the heatsinks are made of a highly conductive material like Aluminum/Copper, and for air cooled cases, the fluid surrounding the heatsink is air. 
The conductivities of these materials are orders of magnitude different. This causes sharp gradients at the interface and makes the neural network training very complex. 
This tutorial shows how such properties and scenarios can be handled via appropriate scaling and architecture choices. 

This tutorial presents two scenarios, one where both the materials are solids but the conductivity ratio between the two is :math:`10^4` and a second scenario where there is conjugate heat transfer between a solid and a fluid where the solid is copper and fluid is air. The scripts used in this problem can be found in ``examples/chip_2d/`` directory. 


2D Solid-Solid Heat Transfer
----------------------------

The geometry of the problem is a simple composite 2D geometry with different material conductivities. The heat source is placed in inside the material of higher conductivity to replicate the actual heatsink and scenario. The objective of this case is to mimic the orders of magnitude difference between Copper :math:`(k=385 \text{ } W/m-K)` and Air :math:`(k=0.0261 \text{ } W/m-K)`. Therefore, set the conductivity of the heatsink and surrounding solid to 100 and 0.01 respectively. 


.. figure:: ../../images/user_guide/2d-solid-solid-geo.png
   :alt: Geometry for 2D solid-solid case
   :width: 60.0%
   :align: center

   Geometry for 2D solid-solid case

For this problem, it was observed that using the Modified Fourier Networks with Gaussian frequencies led to the best results. 
Also, for this problem you can predict the temperatures directly in :math:`(^{\circ} C)`. 
To achieve this, rescale the network outputs according to the rough range of the target solution, which typically requires some domain knowledge of the problem. 
It turns out that this simple strategy not only greatly accelerates the training convergence, but also effectively improves the model performance and predictions. 
The code to setup this problem is shown here: 

.. literalinclude:: ../../../examples/chip_2d/chip_2d_solid_solid_heat_transfer.py
   :language: python

You can monitor the Tensorboard plots to see the convergence of the simulation. The following table summarizes a comparison of the peak temperature achieved by 
the heat sink between the Commercial solver and PhysicsNeMo Sym results. 


.. list-table:: Comparison of the peak temperature with the reference values
   :widths: 30 30 30
   :header-rows: 0
   :align: center

   * - Property
     - OpenFOAM (Reference)
     - PhysicsNeMo Sym (Predicted)
   * - Peak temperature :math:`(^{\circ} C)`
     - :math:`180.24` 
     - :math:`180.28`


This figure visualizes the solution. 

.. figure:: ../../images/user_guide/2d_solid_solid_results.png
   :alt: Results for 2D solid-solid case
   :width: 80.0%
   :align: center

   Results for 2D solid-solid case


2D Solid-Fluid Heat Transfer
----------------------------

The geometry of the problem is very similar to the earlier case, except that now you have a fluid surrounding the solid chip and the dimensions of the 
geometry are more representative of a real heatsink geometry scales. The real properties for air and copper will also be used in this example. This example is also a 
good demonstrator for nondimensionalizing the properties/geometries to improve the neural network training. This figure shows the geometry and measurements for this problem. 

.. figure:: ../../images/user_guide/solid_fluid_geo.png
   :alt: Geometry for 2D solid-fluid case
   :width: 60.0%
   :align: center

   Geometry for 2D solid-fluid case

For this problem you can employ the same strategies for the heat solution that was used in the solid-solid case and use the Modified Fourier Networks with Gaussian frequencies. 
The flow solution and heat solution is one way coupled. Use the same multi-phase training approach that was introduced in :ref:`cht`. A similar approach with rescaling of the 
network outputs is taken to improve the performance of the model. 
 
The code for solving the flow is very similar to the other examples such as :ref:`advection-diffusion` and :ref:`cht`. 
The heat setup is also similar to the solid-solid case that was covered earlier. 
See ``examples/chip_2d/chip_2d_solid_fluid_heat_transfer_flow.py`` and ``examples/chip_2d/chip_2d_solid_fluid_heat_transfer_heat.py`` for more details on the 
definitions of flow/heat constraints and boundary conditions. 

The figure below visualizes the thermal solution in solid and fluid. You can observe that PhysicsNeMo Sym prediction does a much better job in predicting the temperature continuity
at the interface when compared to the commercial solution. We believe these differences in the solver results are due to the discretization errors and can be potentially fixed 
by improving the grid resolution at the interface. PhysicsNeMo Sym prediction however does not suffer from such errors and the physical constraints are respected to a better degree of accuracy. 

.. figure:: ../../images/user_guide/2d_solid_fluid_results.png
   :alt: Results for 2D solid-fluid case
   :width: 99.0%
   :align: center

   Results for 2D solid-fluid case







