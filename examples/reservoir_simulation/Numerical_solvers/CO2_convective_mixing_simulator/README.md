# GPU Based CO<sub>2</sub>-Brin sequestration 
![NVIDIA-Energy](https://www.dgidocs.info/slider/images/media/resources_reservoirsim.jpg)


## **Forward problem**


The forward problem for a CO<sub>2</sub>-Brine system is given by. 

```math
\begin{equation}
\phi \frac{\partial}{\partial t} \left( \sum_{l} \rho_l y_{cl} S_l \right) - \nabla \cdot \mathbf{k} \left( \sum_{l} \rho_l y_{cl} \lambda_l u_l \right) - \sum_{l} \rho_l y_{cl} q_l = 0

\end{equation}
```

```math
\begin{equation}
u_l = -k\lambda_l \nabla\Theta_l = -k\lambda_l \left( \nabla(p - P_{cl}) - \rho_l g\nabla z \right)
\end{equation}
```

```math
\begin{equation}
\lambda_l = \frac{K_{rl}}{\mu_l}
\end{equation}
```

```math
\begin{equation}
\phi \frac{\partial}{\partial t} \left( \sum_{l} \rho_l y_{cl} S_l \right) - \nabla \cdot k \left( \sum_{l} \rho_l y_{cl} \lambda_l \nabla\Theta_l \right) - \sum_{l} \rho_l y_{cl} q_l = 0
\end{equation}
```

Where: $`l`$ is the phase (brine/gas), $`k`$ is the rock absolute permeability, $`\lambda_l`$ is the phase mobility ratio, $`\mu_l`$ is the phase viscosity, $`K_{rl}`$ is the phase relative permeability, $`S_l`$ is the phase saturation, $`u_l`$ is the phase Darcy velocity, $`g`$ is the acceleration due to gravity, $`z`$ is the depth, $`y_{c,l}`$ is the mass fraction of component $`c`$ in phase $`l`$, $`t`$ is time, and $`p`$ is the pressure.

The system is closed by assuming,
```math
\begin{equation}
\sum_{l} S_l = 1, \quad Z_c = \frac{\rho_c}{\rho_T}, \quad \rho_T = \sum_{c} \rho_c
\end{equation}
```


```math
\begin{equation}
S_l = v_l \frac{\rho_T}{\rho_l}, \quad S_g = v_g \frac{\rho_T}{\rho_g}
\end{equation}
```

### **Thermodynamic Equations** 
The CO$`_2`$-brine model includes two components (CO$`_2`$ and H$`_2`$O) that are transported by one or two fluid phases (the brine phase and the CO$`_2`$ phase). We refer to the brine phase with the subscript $`l`$ and to the CO$`_2`$ phase with the subscript $`g`$ (although the CO$`_2`$ phase can be in supercritical, liquid, or gas state). The water component is only present in the brine phase, while the CO$`_2`$ component can be present in the CO$`_2`$ phase as well as in the brine phase. Thus, considering the molar phase component fractions, $`y_{c,p}`$ (i.e., the fraction of the molar mass of phase $`p`$ represented by component $`c`$).

The update of the fluid properties is done in two steps:

1) The phase fractions ($`v_p`$) and phase component fractions ($`y_{c,p}`$) are computed as a function of pressure ($`p`$), temperature ($`T`$), component fractions ($`z_c`$), and a constant salinity.
2) The phase densities ($`\rho_p`$) and phase viscosities ($`\mu_p`$) are computed as a function of pressure, temperature, the updated phase component fractions, and a constant salinity.


Note that the current implementation of the flow solver is isothermal and that the derivatives to temperature are therefore discarded.

The models that are used in steps 1) and 2) are reviewed in more detail below.


#### **Computation of the phase fractions and phase component fractions (flash)** 
We compute the values of CO$`_2`$ solubility in brine as a function of pressure, temperature, and a constant salinity. We note the pressure ($`p`$) and temperature ($`T`$):

Note that the pressures are in Pascal, temperatures are in Kelvin, and the salinity is a molality (moles of NaCl per kg of brine). The temperature must be between 283.15 and 623.15 Kelvin. We solve the following nonlinear CO$`_2$ equation of state (Duan and Sun, 2003) for each pair to obtain the reduced volume as,

```math
\begin{equation}
Z = \left( \frac{p_r V_r}{T_r} \right) = 1 + \frac{ (a_1 + \frac{a_2}{T_r^2} + \frac{a_3}{T_r^3})}{V_r} + \frac{(a_4 + \frac{a_5}{T_r^2} + \frac{a_6}{T_r^3})}{(V_r^2)} + \frac{(a_7 + \frac{a_8}{T_r^2} + \frac{a_9}{T_r^3})}{(V_r^4)} + \frac{(a_{10} + \frac{a_{11}}{T_r^2} + \frac{a_{12}}{T_r^3})}{(V_r^5)} + \frac{a_{13}}{(T_r^3 V_r^2)} \left( a_{14} + a_{15} \left( \frac{1}{V_r^2} \right) \right) \exp \left( -a_{15} \left( \frac{1}{V_r^2} \right) \right)
\end{equation}
```

Where $`p_r = \frac{p}{p_{crit}}`$ is the reduced pressure and the reduced temperature $`T_r = \frac{T}{T_{crit}}`$. The coefficients $`a_1, a_2, \ldots, a_{15}`$ are given as:

$`a_1 = 8.99288497 \times 10^{-2}`$, $`a_2 = -4.94783127 \times 10^{-1}`$, $`a_3 = 4.77922245 \times 10^{-2}`$, $`a_4 = 1.03808883 \times 10^{-2}`$, $`a_5 = -2.82516861 \times 10^{-2}`$, $`a_6 = 9.49887563 \times 10^{-2}`$, $`a_7 = 5.20600880 \times 10^{-4}`$, $`a_8 = -2.93540971 \times 10^{-4}`$, $`a_9 = -1.77265112 \times 10^{-3}`$, $`a_{10} = -2.51101973 \times 10^{-5}`$, $`a_{11} = 8.93353441 \times 10^{-5}`$, $`a_{12} = 7.88998563 \times 10^{-5}`$, $`a_{13} = -1.66727022 \times 10^{-2}`$, $`a_{14} = 1.39800000`$, $`a_{15} = 2.96000000 \times 10^{-2}`$.

Using the reduced volume, $`V_r`$, we compute the fugacity coefficient of CO$`_2`$,

```math
\begin{equation}
\ln \phi (T,P) = Z - 1 - \ln Z + \frac{(a_1 + \frac{a_2}{T_r^2} + \frac{a_3}{T_r^3})}{V_r} + \frac{(a_4 + \frac{a_5}{T_r^2} + \frac{a_6}{T_r^3})}{(2V_r^2)} + \frac{(a_7 + \frac{a_8}{T_r^2} + \frac{a_9}{T_r^3})}{(4V_r^4)} + \frac{(a_{10} + \frac{a_{11}}{T_r^2} + \frac{a_{12}}{T_r^3})}{(5V_r^5)} + \frac{a_{13}}{(2T_r^3 V_r^2)} \left[ a_{14} + 1 - \left( a_{14} + 1 + \frac{a_{15}}{(V_r^2)} \right) \right] \exp \left( -\frac{a_{15}}{(V_r^2)} \right)
\end{equation}
```

To conclude, we use the fugacity coefficient of CO$`_2`$ to compute and store the solubility of CO$`_2`$ in brine, $`s_{\text{CO}_2}`$,

```math
\begin{equation}
\ln \frac{y_{\text{CO}_2}}{s_{\text{CO}_2}} P = \frac{\Phi_{\text{CO}_2}}{RT} - \ln \phi (T,P) + \sum_{c} 2\lambda_c m + \sum_{a} 2\lambda_a m + \sum_{(a,c)} \xi_{(a,c)} m^2
\end{equation}
```

Where $`\Phi_{\text{CO}_2}`$ is the chemical potential of the CO$`_2`$ component, $`R`$ is the gas constant, and $`m`$ is the salinity. The mole fraction of CO$`_2`$ in the vapor phase, $`y_{\text{CO}_2} = \left( \frac{p - p_{\text{H}_2\text{O}}}{p} \right)`$.

Then, we compute the phase fractions as:
```math
\begin{equation}
v_l = \frac{(1 + s_{\text{CO}_2})}{\left( 1 + \frac{z_{\text{CO}_2}}{(1-z_{\text{CO}_2})} \right)}

\quad v_g = 1 - v_l

\end{equation}
```

We conclude by computing the phase component fractions as:
```math
\begin{equation}

y_{\text{CO}_2,l} = \frac{s_{\text{CO}_2}}{(1 + s_{\text{CO}_2})}, \quad y_{\text{H}_2\text{O},l} = 1 - y_{\text{CO}_2,l}

\quad y_{\text{CO}_2,g} = 1, \quad y_{\text{H}_2\text{O},g} = 0

\end{equation}
```


#### **Computation of the phase densities and phase viscosities**

**<u>CO<sub>2</sub></u> phase density and viscosity**


The nonlinear Helmholtz energy equation yields, 

```math
\begin{equation}
\frac{P}{(RT\rho_g)} = 1 + \psi \phi_\psi^r (\psi,\tau)
\end{equation}
```

Where, 
```math
\begin{equation}
\psi = \frac{\rho_g}{\rho_{crit}}, \quad \tau = \frac{T_{crit}}{T}
\end{equation}
```

```math
\begin{equation}
\mu_g = \mu_o (T) + \mu_{excess} (\rho_g,T)
\end{equation}
```

```math
\begin{equation}
\mu_{excess} (\rho_g,T) = d_1 \rho_g + d_2 \rho_g^2 + \frac{d_3 \rho_g^6}{T^3} + d_4 \rho_g^8 + \frac{d_5 \rho_g^8}{T}
\end{equation}
```
Where,
$`d_1 = 0.4071119 \times 10^{-2}`$ , $`d_2 = 0.7198037 \times 10^{-4}`$ , $`d_3 = 0.2411697 \times 10^{-16}`$ , $`d_4 = 0.2971072 \times 10^{-22}`$ , $`d_5 = -0.1627888 \times 10^{-22}`$ 

```math
\begin{equation}
\mu_o (T) = \frac{1.00697T^{0.5}}{B^*(T^*)}, \quad \ln B^*(T^*) = \sum_{i=0}^{4} x_i (\ln T^*)^i, \quad T^* = \omega T, \quad \omega = \frac{1}{251.196 \text{ K}}
\end{equation}
```
Where,
$`x_0 = 0.235156`$ , $`x_1 = -0.491266`$ , $`x_2 = 5.211155 \times 10^{-2}`$ , $`x_3 = 5.347906 \times 10^{-2}`$ , $`x_4 = -1.537102 \times 10^{-2}`$ 

**<u>Brine phase density and viscosity</u>**


```math
\begin{equation}
\rho_{(l,table)} = A_1 + A_2 x + A_3 x^2 + A_4 x^3
\end{equation}
```

```math
\begin{equation}
x = c_1 \exp(a_1 m) + c_2 \exp(a_2 T) + c_3 \exp(a_3 P)
\end{equation}
```

```math
\begin{equation}
\rho_l = \rho_{(l,table)} + M_{\text{CO}_2} C_{\text{CO}_2} - C_{\text{CO}_2} \rho_{(l,table)} V_\phi
\end{equation}
```

```math
\begin{equation}
C_{\text{CO}_2} = \frac{y_{\text{CO}_2,l} \rho_{(l,table)}}{M_{\text{H}_2\text{O}} (1-y_{\text{CO}_2,l})}
\end{equation}
```

```math
\begin{equation}
V_\phi = 37.51 - (T \times 9.585 \times 10^{-2}) + (T^2 \times 8.740 \times 10^{-4}) - (T^3 \times 5.044 \times 10^{-7})
\end{equation}
```

```math
\begin{equation}
\mu_l = a_z T + b_z
\end{equation}
```

```math
\begin{equation}
a_z = \mu_w (T) \times 0.000629(1 - \exp(-0.7m))
\end{equation}
```

```math
\begin{equation}
b_z = \mu_w (T)(1 + 0.0816m + 0.0122m^2 + 0.000128m^3)
\end{equation}
```

## Important Dependencies & Prerequisites:
- CUDA 11.8 : [link](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- CuPy : [link](https://github.com/cupy/cupy.git)
- Python 3.8 upwards
```bash
# enable shell script execution.
sudo chmod +x ./scripts/docker/docker-build.sh
# Build docker image
./scripts/docker/docker-build.sh

# enable shell script execution.
sudo chmod +x ./scripts/docker/docker-run.sh
# Run docker container (also enables X server for docker)
./scripts/docker/docker-run.sh
```

## Features
* CO<sub>2</sub> sequestration with diffusion and convective mixing phase equilibra
* Creates an animation at the end

## Getting Started:
- These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
- The code is developed in a Linux environment.

![alt text](Evolution.gif)**Figure 1**: CO2 dissolution in brine over some epoch iterations 


- From terminal, install (missing) dependencies in 'requirements.txt' in the conda enviroment **MDLO**
- Follow instructions to install CuPy from : [link](https://github.com/cupy/cupy.git)

### Run


#### Forward problem


- Navigate to the code base root directory - **work folder** via terminal.

##### Bare Metal alone
```bash
cd **work folder**
```
- where **work folder** is the location you downloaded the code base to.

- Download the supplemental material.


##### Bare Metal
```bash
conda activate MDLO 
cd src
python Convective_Mixing.py


cd ..
conda deactivate
```
## Author:
- Clement Etienam- Solution Architect-Energy @NVIDIA  Email: cetienam@nvidia.com

