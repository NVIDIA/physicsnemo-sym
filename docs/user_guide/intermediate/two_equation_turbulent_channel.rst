.. _two_equation_turbulent_channel:

Fully Developed Turbulent Channel Flow
======================================

Introduction
------------

This tutorial demonstrates the use of PINNs to solve a canonical 
turbulent flow in a 2D channel using two equation turbulence models and wall functions, 
without using any training data. 

This tutorial also demonstrates how lookup tables can be created which are used in some 
forms of turbulence modeling using neural networks. 

.. note:: Solving turbulence using PINNs and Neural networks in 
   general, is an active field of research. While making a generalizable turbulence model implementation in 
   PINNs is an ambitious goal, with this example, the aim is to provide a few potential ideas 
   that you can use to solve turbulent systems using two equation models and wall functions.

.. note:: This chapter is in continuous development and will be updated with new methods and improvements
   over time.

Problem Description
-------------------

This tutorial tests the turbulence implementation and the wall functions on a fully 
developed channel flow case. The flow is set between two flat plates separated by a distance
2H in y-direction and is driven by a constant negative pressure gradient in the streamwise direction. 
The flow in streamwise direction is periodic and flow is homogenous in the z-direction. 

.. figure:: /images/user_guide/turbulent_channel_geometry.png
   :alt: Turbulent flow in a 2D channel 
   :width: 50.0%
   :align: center

   Turbulent flow in a 2D channel

Solve the problem for a friction Reynolds number of 590. The friction Reynolds number is defined as

.. math::
   Re_{\tau} = \frac{u_{\tau}H}{\nu}

The pressure gradient is determined from the friction velocity as 

.. math::
   \frac{\partial p}{\partial x} = \frac{u_{\tau}^2}{H}


Governing Equations
-------------------

This example implements the high Reynolds number version of the standard k epsilon model with wall functions.
The goal is to solve the equations beyond a certain distance from the wall so that the viscous 
sublayer is not resolved. The turbulence equations for the k-epsilon model is shown here: 

.. math::

   \frac{\partial k}{\partial t} + U \cdot \nabla k = \nabla \cdot \left[ \left( \nu + \frac{\nu_t}{\sigma_k} \right) \nabla k \right] + P_k - \varepsilon 

.. math::

   \frac{\partial \varepsilon}{\partial t} + U \cdot \nabla \varepsilon = \nabla \cdot \left[ \left( \nu + \frac{\nu_t}{\sigma_{\varepsilon}} \right) \nabla \varepsilon \right] + (C_{\varepsilon 1}P_k - C_{\varepsilon 2} \varepsilon)\frac{\varepsilon}{k}

Next, this tutorial focuses on wall functions. :numref:`fig-point-sampling` shows the wall modeling approach followed in this example. As seen in the figure, 
the equation loses are applied beyond a particular wall height that is chosen a priori. The points below this chosen wall height are discarded. 
The wall function relations that are shown in the following sections are then applied at this wall height. The wall height is chosen such that the 
:math:`y^+` is :math:`~30` which ensures that the inertial sublayer assumption is valid. 

.. _fig-point-sampling:

.. figure:: /images/user_guide/turbulent_channel_sampling.png
   :alt: Turbulent channel sampling 
   :width: 50.0%
   :align: center

   Sampling for interior and wall function points

Not all of the important equations used in the formulation are presented here. For a more detailed description of the wall functions, see [#bredberg2000wall]_ [#lacasse2004judicious]_. 


Standard Wall Functions
~~~~~~~~~~~~~~~~~~~~~~~

Assuming the closest point to the wall (:math:`P`) is within the logarithmic layer, the tangential velocity can be set as 

.. math::
   :label: log-law

   \begin{align}
   U &= \frac{u_{\tau}}{\kappa} \ln{(E y^+)} 
   \end{align}

Where, :math:`\kappa=0.4187` and :math:`E=9.793`. 

Here, since the :math:`U` and :math:`u_{\tau}` are related implicitly, you can use a lookup table approach to solve for :math:`u_{\tau}` given :math:`U` and the wall distance :math:`y`.
This approach is similar to the one discussed here [#kalitzin2005near]_. 

The :math:`k` and :math:`\varepsilon` can be set using the below relations

.. math::
   \begin{align}
   k &= \frac{u_{\tau}^2}{\sqrt{C_{\mu}}} \\
   \varepsilon &= \frac{C_{\mu} ^ {3 / 4} k ^ {3 / 2}}{\kappa y}
   \end{align}

The total shear stress :math:`\tau` is the sum of laminar and turbulent shear stresses, i.e. :math:`\tau=\tau_l+\tau_t`. 
At the wall, the laminar shear dominates and turbulent shear stress drops to zero, while in the log layer, the turbulent shear 
stress dominates. For setting the shear stress appropriately, it is assumed that the total shear stress at wall is equal to the
turbulent shear stress in the log layer. For :math:`y^+` values closer to 30, this assumption does not lead to large inaccuracies. [#moser1999direct]_ [#bredberg2000wall]_

Therefore, the wall friction can be set using the log-law assumption and the approximation above as shown here:

.. math::
   \tau_w \equiv \mu \frac{\partial U}{\partial y} \Bigg|_{w} \approx (\mu + \mu_t) \frac{\partial U}{\partial y} \Bigg|_{P} = \frac{\rho u_{\tau}U \kappa}{\ln{(E y^+)}}


Launder Spalding Wall Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The relations for Launder-Spalding wall functions formulation are similar to standard one, except now the friction velocity can directly be computed 
from the turbulent kinetic energy as shown below:

.. math::
   \begin{align}
   u_{\tau} &= C_{\mu} ^ {1 / 4} k ^ {1 / 2} \\
   U &= \frac{C_{\mu} ^ {1 / 4} k ^ {1 / 2}}{\kappa} \ln{(E y^+)} 
   \end{align}


With this formulation, it was found that an explicit boundary condition for :math:`k` is not required as the log-law, relation for :math:`\varepsilon` and :math:`\tau_w` are sufficient to define all the variables at point :math:`P`. 

.. math::
   \begin{align}
   \varepsilon &= \frac{C_{\mu} ^ {3 / 4} k ^ {3 / 2}}{\kappa y} \\
   \tau_w \equiv \mu \frac{\partial U}{\partial y} \Bigg|_{w} \approx (\mu + \mu_t) \frac{\partial U}{\partial y} \Bigg|_{P} &= \frac{\rho C_{\mu} ^ {1 / 4} k ^ {1 / 2} U \kappa}{\ln{(E y^+)}}
   \end{align}


.. note::

   The python code for this problem can be found in the directory ``examples/turbulent_channel/``. 

Case Setup - Standard Wall Functions
------------------------------------

The equations are symbolically defined using the custom PDEs (for details on 
setting up custom PDEs in PhysicsNeMo Sym, please see :ref:`transient`). For these problems, the convergence behavior is greatly influenced by the initialization of the network. 
Therefore, this example starts by training the network to predict constant profiles for turbulent flow quantities and then slowly ramps up the equation losses while ramping down the initialization losses. To do this, a :ref:`custom_aggregator` is defined. 

For the Standard wall function approach, the equation :eq:`log-law` needs to be satisfied exactly. So, train a neural network that can lookup values of :math:`u_{\tau}`. This is done by solving the log law for a range of :math:`u` and
:math:`y` values and using a Newton-Raphson method to find the zeros. Once you have the table ready, you can train a network to assimilate these values. 

Now let's walk through each of the sections one by one. 

This code shows the script to generate the lookup table and assimilate the values in a neural network. 

.. literalinclude:: ../../../examples/turbulent_channel/2d_std_wf/u_tau_lookup.py
   :language: python


The code below shows the definition of custom PDEs. Define separate PDEs for the initialization losses, equation losses and wall function losses. 
For this, use the initialization strategy that is used for CFD problems discussed `here <https://www.cfd-online.com/Wiki/Turbulence_free-stream_boundary_conditions>`_ .

.. literalinclude:: ../../../examples/turbulent_channel/2d_std_wf/custom_k_ep.py
   :language: python

.. _custom_aggregator:

Custom Aggregator
~~~~~~~~~~~~~~~~~

The aggregator in PhysicsNeMo Sym combines different loss terms together to form a global loss that is used by the network during optimization. The default behavior of the aggregator is to sum all the weighted losses. However, this example varies which losses are applied in the begining of the model training (initialization losses) and which ones are applied later (equations and wall function relations). For this, a :math:`tanh` function is used to smoothly ramp up and ramp down the different losses. The code for the following is shown here: 

.. literalinclude:: ../../../examples/turbulent_channel/2d_std_wf/custom_aggregator.py
   :language: python

The custom aggregator is then called in the config files under the ``loss`` config group as shown here:

.. literalinclude:: ../../../examples/turbulent_channel/2d_std_wf/conf_re590_k_ep/config.yaml
   :language: yaml

Once all the prerequisites are defined, you can start solving the turbulent channel flow. Notice that the config file imports the ``u_tau_network`` from a previous tutorial by setting the appropriate initialization directory. Set the ``optimize``
parameter for that network to ``False`` so that the weights and biases from that network do not change during the training. Also, to setup the 
periodicity in the domain, create symbolic nodes that transform :math:`x` to :math:`sin(x)` which will become the input to the neural network. 

.. literalinclude:: ../../../examples/turbulent_channel/2d_std_wf/re590_k_ep.py
   :language: python


Case Setup - Launder Spalding Wall Functions
--------------------------------------------

The case setup for this problem is very similar to the earlier one, except now that you can avoid training of the lookup network as the :math:`u_{\tau}` is explicitly defined using the turbulent
kinetic energy. The process to define the custom PDEs is similar as before. The custom PDEs for the Launder Spalding wall functions can be found in ``examples/turbulent_channel/2d/custom_k_ep_ls.py``

Use the same custom aggregator you defined earlier as a similar initialization strategy will work for this problem as well. Once you have the PDEs and the aggregator defined, you can setup 
the script to solve the problem as shown below. There is no seperate network for the lookup table and the losses at the wall are different from the standard wall function formulation. 

.. literalinclude:: ../../../examples/turbulent_channel/2d/re590_k_ep_LS.py
   :language: python


Post-processing, Results and Discussion
---------------------------------------

The PhysicsNeMo Sym results are shown in the figure below along with the DNS data [#moser1999direct]_ and Solver data [#gistanford]_. Observe that the the nondimensionalized velocity and turbulent kinetic energy profiles match very well with the DNS and solver data. Also, the k-omega models are able to predict the friction velocity with better accuracy when compared to the k-epsilon models. 

.. figure:: /images/user_guide/turbulent_channel_results.png
   :alt: Turbulent flow in a 2D channel 
   :width: 100.0%
   :align: center

   Turbulent channel flow results

The scripts for generating the plots shown above as well as the scripts for k-omega models can be found the the examples directory at ``examples/turbulent_channel/``. 


.. rubric:: References

.. [#bredberg2000wall] Bredberg, Jonas. "On the wall boundary condition for turbulence models." Chalmers University of Technology, Department of Thermo and Fluid Dynamics. Internal Report 00/4. G oteborg (2000): 8-16.

.. [#lacasse2004judicious] Lacasse, David, Eric Turgeon, and Dominique Pelletier. "On the judicious use of the k–ε model, wall functions and adaptivity." International Journal of Thermal Sciences 43.10 (2004): 925-938.

.. [#moser1999direct] Moser, Robert D., John Kim, and Nagi N. Mansour. "Direct numerical simulation of turbulent channel flow up to Re τ= 590." Physics of fluids 11.4 (1999): 943-945.

.. [#kalitzin2005near] Kalitzin, Georgi, et al. "Near-wall behavior of RANS turbulence models and implications for wall functions." Journal of Computational Physics 204.1 (2005): 265-291.

.. [#gistanford] Gianluca Iaccarino Lecture handouts: Computational Methods in Fluid Dynamics using commercial CFD codes. https://web.stanford.edu/class/me469b/handouts/turbulence.pdf
