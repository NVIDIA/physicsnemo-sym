![NVIDIA-Energy](../pino_modeling/3D/Forward_problem_results/FNO/TRUE/Evolution_water_3D.gif)

# [GPU accelerated Reservoir Simulator](https://gitlab-master.nvidia.com/GlobalEnergyTeam/simulation/gpu_black_oil_reservoir_simulation)


### Forward problem
**Black oil model**;

 - Our simplified model for two-phase flow in porous media for reservoir simulation is given as [7].
```math
\begin{equation}
φ\frac{∂S_w}{∂t} - \nabla .[T_{w} (\nabla.p_{w} + ρ_{w}gk)]= Q_{w} 
\end{equation}
```
```math
\begin{equation}
φ\frac{∂S_o}{∂t}- \nabla .[T_{o} (\nabla.p_{o} + ρ_{o} gk)]= Q_{o}     
\end{equation}
```
$`φ(x)`$ stands for the porosity,subscript $`w`$ stands for water and subscript $`o`$ stands for oil. $`T_{o},T_{w},T`$ stands for the transmissibilities, which are known functions of the permeability $`K`$ and the water saturation $`S_w`$. The system is closed by adding two additional equations

```math
\begin{equation}
P_{cwo} = p_o- p_w  ; S_w+ S_o=1. 
\end{equation}
``` 
This gives four unknowns
```math
\begin{equation}
p_{o}, p_{w}, S_{w}, S_{o}
\end{equation}
``` 
Gravity effects are considered by the terms 
```math
\begin{equation}
ρ_{w}gk,ρ_{o}gk\quad Ω \subset R^{n}(n = 2, 3) 
\end{equation}
``` 
The subsequent water, oil and overall transmissibilities is given by,
```math
\begin{equation} 
T_w = \frac{K(x) K_{rw}S_w}{μ_w}  ;T_o = \frac{K(x) K_{ro}S_o}{μ_o}   ;T= T_w + T_o          
\end{equation}
``` 
The relative permeabilities $`K_{rw} (S_w ),K_{ro} (S_w )`$ are available as tabulated functions, and $`μ_w,μ_o`$ denote the viscosities of each phase. 

We define the oil flow, the water flow, and the total flow, respectively, which are measured at the well position as; 
```math
\begin{equation} 
Q = Q_{o}  + Q_{w }
\end{equation}
``` 

The final pressure and saturation equations for a two-phase oil-water flow is
```math
\begin{equation} 
- ∇ .[T∇p]=Q   
\end{equation}
```  
```math
\begin{equation}                                               
φ\frac{∂S_w}{∂t} - \nabla .[T_{w} (\nabla.p_{w}]= Q_{w} 
\end{equation}
```  


## Features
* Simulates 2 / 3-phase flow Black oil forward Modeling with GPU enhanced linear inverse problem solution ansatz
* cuSOLVER integrated workflow for matrix algebra computation and sparse linear solver
* Configures from a Yaml file
* Creates an animation at the end

## Important Dependencies & Prerequisites:
- NVIDIA's in-house GPU based black oil reservoir simulator - **NVRS**
- CUDA 11.8 : [link](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- CuPy : [link](https://github.com/cupy/cupy.git)
- Python 3.8 upwards

## Getting Started:
- These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
- The code is developed in a Linux environment.

## Installation

- From terminal create a new conda enviroment named **MDLO** (check if conda is installed) .
```bash
conda create --name MDLO python=3.8
```

Clone this code base repository in a dedicated **work folder**.
```bash
cd **work folder**
conda activate MDLO
git clone https://gitlab-master.nvidia.com/GlobalEnergyTeam/simulation/gpu_black_oil_reservoir_simulation.git

```
- From terminal, install missing dependencies in `requirements.txt` inside the **MDLO** environment.
- Follow instructions to install CuPy from : [link](https://github.com/cupy/cupy.git)

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

### Run
**NVRS** is a fully GPU based Black oil reservoir simulator.
Solvers include;
1) Left-Preconditioned GMRES [link](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.gmres.html)
2) LSQR [link](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.lsqr.html)
3) Left-Preconditioned Conjugate Gradient [link](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.cg.html)
4) Constrained Pressure Residual (CPR)  (V-cycle two-stage AMG for the pressure solve and left-preconditioned GMRES with ILU(0) for the saturation solve) [link](https://doi.org/10.2118/96809-MS)
6) Spsolve [link](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.spsolve.html)


**CPR is the default solver**.
1) Pressure solver: V cycle 2 stage AMG with Aggregation and Level scheduling for the coarsening/restriction operations
2) Saturation solver:  Left preconditoned GMRES with ILU(0) as preconditioner
3) Smoothers include : Jacobi, Gauss-Seidel, SOR

## How to run

```
python NVRS/simulator.py Run.yaml
```
## Results
| Reservoir 1 (Nx =33, Ny =33, Nz =1) | Reservoir 2 (Nx = 40, Ny = 40, Nz =3)| Reservoir 3  (Nx= 40, Ny = 60 ,Nz =5)|
| --------------------|---------------------|---------------------|
| ![Image 1][img1]     | ![Image 2][img2]     | ![Image 3][img3]     |
| **Figure(a) -Permeability**| **Figure (b) - Permeability**| **Figure (c) - Permeability**|
| --------------------|---------------------|---------------------|
| ![Image 4][img4]     | ![Image 5][img5]     | ![Image 6][img6]     |
| **Figure (d) -Pressure**| **Figure (e) -Pressure**| **Figure (f) - Pressure** |
| --------------------|---------------------|---------------------|
| ![Image 7][img7]     | ![Image 8][img8]     | ![Image 9][img9]     |
| **Figure (g) -Water saturation**| **Figure (h) - Water saturation** | **Figure (i) - Water saturation** |
| --------------------|---------------------|---------------------|
| ![Image 10][img10]  | ![Image 11][img11]   | ![Image 12][img12]   |
| **Figure (j) - Oil saturation** | **Figure (k) - Oil saturation** | **Figure (l) - Oil saturation**|
| --------------------|---------------------|---------------------|
| ![Image 13][img13]  | ![Image 14][img14]   | ![Image 15][img15]   |
| **Figure (m) - P- wave impedance** | **Figure (n) - P- wave impedance** | **Figure (o) - P - wave impedance**|




[img1]: RESULTS/33331/Permeability.png "Permeability Field ( 33 by 33 by 1)"
[img2]: RESULTS/40403/Permeability.png "Permeability Field ( 40 by 40 by 3)"
[img3]: RESULTS/40605/Permeability.png "Permeability Field ( 40 by 60 by 5)"
[img4]: RESULTS/33331/Evolution_pressure_3D.gif "Pressure Field (33 by 33 by 1)"
[img5]: RESULTS/40403/Evolution_pressure_3D.gif "Pressure Field (40 by 40 by 3)"
[img6]: RESULTS/40605/Evolution_pressure_3D.gif "Pressure Field (40 by 60 by 5)"
[img7]: RESULTS/33331/Evolution_water_3D.gif "Water Field (33 by 33 by 1)"
[img8]: RESULTS/40403/Evolution_water_3D.gif "Water Field (40 by 40 by 3)"
[img9]: RESULTS/40605/Evolution_water_3D.gif "Water Field (40 by 60 by 5)"
[img10]: RESULTS/33331/Evolution_oil_3D.gif "Oil Field (33 by 33 by 1)"
[img11]: RESULTS/40403/Evolution_oil_3D.gif "Oil Field (40 by 40 by 3)"
[img12]: RESULTS/40605/Evolution_oil_3D.gif "Oil Field (40 by 60 by 5)"
[img13]: RESULTS/33331/Evolution2_impededance.gif "Impedance Field (33 by 33 by 1)"
[img14]: RESULTS/40403/Evolution2_impededance.gif "Impedance Field (40 by 40 by 3)"
[img15]: RESULTS/40605/Evolution2_impededance.gif "Impedance Field (40 by 60 by 5)"



### Contacts and Authors

Clement Etienam cetienam@nvidia.com

Oleg Ovcharenko oovcharenko@nvidia.com

Issam Said isaid@nvidia.com
