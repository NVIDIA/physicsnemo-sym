# Automatic History Matching with a Weighted adaptive Regularised Ensemble Kalman Inversion ($`\alpha`$REKI) and a GPU based black oil forward model
![NVIDIA-Energy](https://www.dgidocs.info/slider/images/media/resources_reservoirsim.jpg)

## An Automatic History Matching Workflow with a GPU based forward solver:
Calibration of subsurface structures is an important step to forecast fluid dynamics and behaviours in a plethora of geoenvironments such as petroleum and geothermal reservoirs. History matching is an ill-posed inverse process to find reservoir model parameters honouring observations by integration of static (e.g., core, logging, and seismic) and dynamic data (e.g., oil and gas rate, water cut, bottomhole pressure, and subsidence/uplift) .In recent developments, the ensemble Kalman filter (EnKF) (Evensen 1994, Law et al 2012, Stuart 2010), ensemble smoother (ES) (van Leeuwen & Evensen 1996) ES with multiple data assimilation (ES-MDA) (Emerick & Reynolds 2013) and Iterative variants of ES (Chen & Oliver 2011, Iglesias 2013) have all been utilised for history matching/data assimilation problems. However, the ensemble-based data assimilation approaches have limited capability in preserving non-Gaussian distributions of model parameters such as facies (Aanonsen et al.,2009, Villegas et al, 2018, Lorentzen et al, 2013). In the ensemble-based data assimilation techniques, model parameters lose the non- Gaussianity of their original distributions (generated from a geostatistical software) that are initially constrained and constructed to available hard data and the distributions of the model parameters tend towards Gaussian ones (Evensen & Eikrem 2018, Kim et al 2018). Some methods utilised for this parametrisation technique in reservoir characterization literature is, the discrete cosine transform (DCT) (Jafarpour & McLaughlin, 2007)  (Liu & Jafarpour, 2013), level set (Moreno & Aanonsen 2007, Dorn & Villegas 2008, Villegas et al 2018, Chang et al 2010, Lorentzen et al., 2013) and sparse geologic dictionaries (Etienam et al, 2019, Kim et al, 2018, Khaninezhad et al, 2012). In particular, Fourier transform-based methods such as DCT are capable of capturing essential traits such as main shapes and patterns of a facies channel reservoir (Khaninezhad, et al., 2012) but reveal a deficiency in describing a crisp contrast among different facies because of data loss from inverse transformation (Kim et al 2018, Khaninezhad et al, 2012, Tarrahi & Afra,2016).

Reservoir model calibration is applicable and relevant for locating new hydrocarbon deposits and for CCUS stratigraphic trapping initiatives in many energy companies. Energy companies are aiming to accelerate their forward simulation codes for precise and detailed subsurface/dynamic/Petro-physical mappings of their assets/plays.A recently developed weighted adaptive regularised ensemble kalman inversion method is used for solving the inverse problem.


The aim of this project is to develop an integrated workflow, where the finite volume fully/adaptive implicit black oil reservoir simulator now used in an inverse problem methodology. Two methods are developed for the inverse problem, a newly developed adaptive regularised ensemble alman inversion method (with various forms of exotic priors). This approach is well suited for forward and inverse uncertainty quantifiication.


## Methods for the forward and inverse problem (In the weeds):


### Inverse problem 

**Weighted-$`\alpha`$REKI**

The aim and target of history matching can be formulated as;

```math
\begin{equation}
\min Φ(m;d^{obs})
\end{equation}
```
where the cost functional for the history matching inverse problem is posed as [24,29,30,31,35,36,19,18,14,15,6,1].
```math
\begin{equation}
Φ(m;d^{obs})≡ \frac{1}{2}\left\VertΓ^{\frac{-1}{2}}(d^{obs}-G(m))\right\Vert^2
\end{equation}
```
Where $`‖.‖`$ represents the $`M`$ dimensional Euclidean Norm.
                                                       
in Eqn.2 $`Φ`$ is the objective function of history matching and m is the state vector composed of reservoir variables (e.g., permeability/porosity and facies) in this case. The typical expression of $`Φ(m;d^{obs})`$for ensemble-based history matching problems is presented as (Emerick & Reynolds, 2013, Oliver et al 2008, Tarantola 2005);
```math
\begin{equation}
Φ(m;d^{obs}) = (m-m^b)^T B^{-1} (m-m^b ) + (d^{obs}-d)^T Γ^{-1} (d^{obs}-d)
\end{equation}
```
In Eqn.3, $`\,`$ $`m^b`$ denotes state vector before update and the superscript b denotes the background; $`B`$ denotes the covariance matrix of $`m^b`$; $`d^{obs}`$ denotes the observed responses; $`d = G(m)`$ is the dynamic vector composed of simulated responses constructed by running a reservoir simulator or the surrogate simulator $`G`$ for the state vector $`m`$; and $`Γ`$ denotes the covariance matrix of observation error. The right-hand side of Eqn.3 is the addition of background and observation error terms (Emerick & Reynolds, 2013).  $`m`$  in principle can contain any unknown variables such as facies indexes, permeability/porosity field, coefficients of discrete cosine functions or sparse coefficients depending.

$`\frac{∂J(m)} {∂m} = 0`$ can be used to derive the minimum of the cost function and update equation for $`m`$ as (Emerick and Reynolds, 2013) 
```math
\begin{equation}
m_i = m_i^{b} + C_{md} (C_{dd} +α_p C_d )^{-1} (d_i^{pert}-d_i )\quad  for \quad i = 1…N_{ens}
\end{equation}
```
                               

In Eqn.4 $`\,`$ the subscript $`i`$ denotes the $`i^{th}`$ ensemble member; $`C_{md}`$ denotes the cross-covariance matrix of $`m`$ and $`d`$; $`C_{dd}`$ denotes the autocovariance matrix of $`d`$; $`α_p`$ is the coefficient to inflate $`C_d`$, which denotes the covariance matrix of the observed data measurement error (Emerick and Reynolds, 2013); $`d^{pert}`$ denotes the observation data perturbed by the inflated observed data measurement error; and $`N_{ens}`$ is the ensemble size (i.e., number of reservoir models as columns of the matrix in the ensemble). Conventionally, ensemble-based history matching updates $`N_{ens}`$ reservoir models simultaneously. In Eqn.4, $`\frac {C_{md}}{(C_{dd}+α_pC_d )}`$ denotes the Kalman gain $`K`$, which is normally computed with regularization by SVD using 99.9% of the total energy in singular values (Emerick and Reynolds, 2013, Oliver et al 2008, Law et al 2012) 

The main difference between ES and ES-MDA is the update process of the state vector $`m`$. ES updates the state vector of each ensemble member using observation data measured at all time steps (Emerick and Reynolds, 2013). Compared to the single assimilation of ES, ES-MDA assimilates every state vector $`N_a`$ times using an inflated covariance matrix of measurement error (Emerick and Reynolds, 2013). In this case, $`N_a`$ is the number of assimilations in ES-MDA. 
We define $`C_{md}`$ and $`C_{dd}`$  as follows:
```math
\begin{equation}
C_{md} = \frac{1}{(N_{ens}-1)}  \sum_{n=1}^{N_{ens}} (m_i- \bar{m})(d_i-\bar{d})^T 
\end{equation}
```

```math
\begin{equation}
C_{dd} = \frac{1}{(N_{ens}-1)}  \sum_{n=1}^{N_{ens}} (d_i- \bar{d})(d_i-\bar{d})^T 
\end{equation}
```
```math
\textrm{where}\quad\bar{m}\quad\textrm{denotes the mean of the parameters and }\quad\bar{d}\quad\textrm{denotes the mean of predicted values}
```
```math
\textrm{In ES-MDA}, α_p\quad\textrm{is normally constrained to}\quad\sum_{n=1}^{N_{a}} \frac{1}{a_p} = 1  
```

```math
\begin{equation}
d_{i}^{pert} = d^{obs} + a_p^{0.5} C_d^{0.5} z_{d,i} \quad   for \quad i=1…N_{ens} 
\end{equation}
```


The second term on the right-hand side of Eqn.7 $`\,`$ is known as the perturbation term. It reflects the uncertainty associated with data measurement and processing. The stochastic characteristics of $`C_d`$ are reflected by $`z_d \sim \mathcal{N}(0,I_{N_d } )`$  $`z_d`$ is the random error matrix to observations, which is constructed with a mean of zero and a standard deviation of $`I_{N_d}`$, where $`N_d`$ is the number of time steps found in the observations (Emerick and Reynolds 2013).

The newly developed $`\alpha`$REKI constructs the dampening parameter $`α_p`$ via the discrepancy principle and jeffrey's divergence

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
### Forward modelling


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
3) Left Preconditioned Conjugate Gradient [link](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.cg.html)
4) Constrained Pressure Residual (CPR) (V-cycle two-stage AMG for the pressure solve and left-preconditioned GMRES with ILU(0) for the saturation solve) [link](https://doi.org/10.2118/96809-MS)
6) Spsolve [link](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.spsolve.html)
7) AMGx suite of solvers

**CPR is the default solver**.
1) Pressure solver: V cycle 2 stage AMG with Aggregation and Level scheduling for the coarsening/restriction operations
2) Saturation solver:  Left preconditioned GMRES with ILU(0) as preconditioner
3) Smoothers include : Jacobi, Gauss-Seidel, SOR


#### Inverse problem

**Parametrisation methods (non-Gaussian exotic priors) for the petrophysical properties includes** :
1) Generative adversarial network (GAN)
2) Variational Convolution autoencoder (VCAE)
3) Cluster Classify Regress (CCR) introduced in Etienam et al, 2020 (MoE)
4) Normal score transformation
5) Diffusion
6) K-means
7) PCA/KPCA
8) Mixture of Experts method introduced in Etienam et al, 2019 (MoE)
9) Convolution autoencoder for parametrising the permeability field
10) Denoising convolution autoencoder for parametrising the permeability field, 
11) DCT( discrete cosing transform) introduced in Jafarpour et al 2009
12) KSVD approach and orthogonal matching pursuit introduced in Etienam, 2019
13) Level set approach introduced in Villegas et al, 2018
14) Normal score transformation
 
- Run the inverse problem workflow from the **src** folder

The aim of parametrisation methods is to reduce the complexity of the Kalman gain matrix in Eqn.4, make the inverse problem less ill-posed and enforce prior non-Gaussian densities during the reconstruction.



##### Bare Metal
```bash
conda activate MDLO
python simulator_ensemble.py
conda deactivate
```


- Inverse problem solution results are found in the root directory folder **HM_RESULTS**

## Pretrained models

- Pre-trained models and all necessary files are provided in the script for rapid prototyping & reproduction

## Results
### Summary of Numerical Model




| Column 1             | Column 2             |Column 3             |
| --------------------|---------------------|---------------------|
| ![Image 1][img1]     | ![Image 2][img2]     |![Image 3][img3]     |
| **Figure 1(a) - permeability field reconstruction (bottom-right) True Model (top-left) Prior, (top-right) Posterior – aREKI + Convolution autoencoder, (bottm-left) MAP estimate**  | **Figure 1(b) - Cost function evolution. (Top -left) prior ensemble RMSE cost, (Top-right) posterior ensemble RMSE cost, (Bottom-left) RMSE cost evolutio betyween the MAP model (blue) and the MLE model (green)** | **Figure 1(c) - Production profile comparison for posterior ensemble. (red) True model, (grey) Ensemble. First row is for the bottom-hole-pressure of well injectors (I1-I4), second row is for the oil rate production for the well producers (P1-P4), third row is for the water rate production for the well producers (P1-P4) and the last row is for the water cut ratio of the 4 well producers (P1-P4). Left of the dash vertical line is used for assimilation while right of this line is used for prediction. Notice the decrease in the spread of the ensemble members signifying a reduction in uncertainty** |
| --------------------|---------------------|---------------------|
| ![Image 4][img4]     | ![Image 5][img5] |![Image 6][img6] |
| **Figure 1(d) - Production profile comparison for prior ensemble. (red) True model, (grey) Ensemble. First row is for the bottom-hole-pressure of well injectors (I1-I4), second row is for the oil rate production for the well producers (P1-P4), third row is for the water rate production for the well producers (P1-P4) and the last row is for the water cut ratio of the 4 well producers (P1-P4). Left of the dash vertical line is used for assimilation while right of this line is used for prediction.**  |**Figure 1(e) - Production profile comparison for prior ensemble. (red) True model, (grey) Ensemble. First row is for the bottom-hole-pressure of well injectors (I1-I4), second row is for the oil rate production for the well producers (P1-P4), third row is for the water rate production for the well producers (P1-P4) and the last row is for the water cut ratio of the 4 well producers (P1-P4). Left of the dash vertical line is used for assimilation while right of this line is used for prediction.**  |  |

[img1]: HM_RESULTS/Comparison.png "Numerical implementation of Reservoir forward simulation. PINO based reservoir forwarding showing the 3D water saturation evolution with well locations"
[img2]: HM_RESULTS/Cost_Function.png "Numerical implementation of Reservoir forward simulation. PINO based reservoir forwarding showing the 3D oil saturation evolution with well locations"
[img3]: HM_RESULTS/Final.png "Numerical implementation of Reservoir forward simulation. PINO based reservoir forwarding shwoing the field pressure evolution with well locations."
[img4]: HM_RESULTS/Initial.png "Numerical implementation of Reservoir forward simulation. PINO based reservoir forwarding shwoing the field pressure evolution with well locations."
[img5]: HM_RESULTS/P10_P50_P90.png "Numerical implementation of Reservoir forward simulation. PINO based reservoir forwarding shwoing the field pressure evolution with well locations."


- We now compare the MLE model, MAP model and P10 model.

| MLE         | MAP         | P10         |
| ---------------- | ---------------- | ---------------- |
| ![Image 1][img1a] | ![Image 2][img2a] | ![Image 3][img3a] |
| **Figure 2(a) - Production profile comparison**         | **Figure 2(b) - Production profile comparison**        | **Figure 2(c) - Production profile comparison**        |
| ![Image 4][img4a] | ![Image 5][img5a] | ![Image 6][img6] |
| **Figure 2(d) - Oil evolution**        | **Figure 2(e) - Oil evolution**         |**Figure 2(f) - Oil evolution**         |
| ![Image 7][img7] | ![Image 8][img8] | ![Image 9][img9] |
| **Figure 2(g) - Water evolution**        | **Figure 2(h) - Water evolution**        | **Figure 2(I) - water evolution**         |
| ![Image 10][img10] | ![Image 11][img11] | ![Image 12][img12] |
| **Figure 2(j) - Pressure evolution**           | **Figure 2(k) - Pressure evolution**           | **Figure 2(l) - Pressure evolution**          |
| ![Image 13][img13] | ![Image 14][img14] | ![Image 15][img15] |
| **Figure 2(m) - Permeability field**           | **Figure 2(n) - Permeability field**          | **Figure 2(o) - Permeability field**           |

[img1a]: HM_RESULTS/ADAPT_REKI/Compare.png
[img2a]: HM_RESULTS/BEST_RESERVOIR_MODEL/Compare.png
[img3a]: HM_RESULTS/MEAN_RESERVOIR_MODEL/Compare.png
[img4a]: HM_RESULTS/ADAPT_REKI/Evolution_oil_3D.gif
[img5a]: HM_RESULTS/BEST_RESERVOIR_MODEL/Evolution_oil_3D.gif
[img6]: HM_RESULTS/MEAN_RESERVOIR_MODEL/Evolution_oil_3D.gif
[img7]: HM_RESULTS/ADAPT_REKI/Evolution_water_3D.gif
[img8]: HM_RESULTS/BEST_RESERVOIR_MODEL/Evolution_water_3D.gif
[img9]: HM_RESULTS/MEAN_RESERVOIR_MODEL/Evolution_water_3D.gif
[img10]: HM_RESULTS/ADAPT_REKI/Evolution_pressure_3D.gif
[img11]: HM_RESULTS/BEST_RESERVOIR_MODEL/Evolution_pressure_3D.gif
[img12]: HM_RESULTS/MEAN_RESERVOIR_MODEL/Evolution_pressure_3D.gif
[img13]: HM_RESULTS/ADAPT_REKI/Permeability.png
[img14]: HM_RESULTS/BEST_RESERVOIR_MODEL/Permeability.png
[img15]: HM_RESULTS/MEAN_RESERVOIR_MODEL/Permeability.png


## Release Notes



**23.01**
* First release 

## End User License Agreement (EULA)
Refer to the included Energy SDK License Agreement in **Energy_SDK_License_Agreement.pdf** for guidance.

## Author:
- Clement Etienam- Solution Architect-Energy @NVIDIA  Email: cetienam@nvidia.com



## References:
[1] Aanonsen, S., Oliver, D., Reynolds, A. & Valles, B., 2009. The Ensemble Kalman Filter in Reservoir Engineering--a Review. SPE Journal, 14(3), pp. 393-412.

[2] Aharon, M., Elad, M. & Bruckstein, A., 2006. K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation. IEEE Transactions on Signal Processing, 54(11), pp. 4311-4322.

[3] Bishop. Christopher M 2006. Pattern Recognition and Machine Learning (Information Science and Statistics). Springer-Verlag, Berlin, Heidelberg.

[4] Candes, E. J., Romberg, J. & Tao, T., 2006. Robust uncertainty principles: Exact signal reconstruction from high incomplete frequency information. Information Theory, IEEE Transactions on, 52(2), pp. 489-509.

[5] Chang, H., Zhang, D. & Lu, Z., 2010. History matching of facies distribution with the EnKF and level set parameterization. Journal of Computational Physics, 229(1), pp. 8011-8030.

[6] Chen, Y. and Oliver,D., .2011 Ensemble randomized maximum likelihood method as an iterative ensemble smoother, Mathematical Geosciences, Online First.


[7] Dorn, O. & Villegas, R., 2008. History matching of petroleum reservoirs using a level set technique. Inverse problems, p. 24.


[8] Elsheikh, A., Wheeler, M. & Hoteit, I., 2013. Sparse calibration of subsurface flow models using nonlinear orthogonal matching pursuit and an iterative stochastic ensemble method. Advances in Water Resources, 56(1), pp. 14-26.

[9] Emerick, A. A. and Reynolds A. C., 2013 Ensemble smoother with multiple data assimilation, Computers & Geosciences.

[10] Etienam, C., Mahmood, I. & Villegas, R., 2017. History matching of Reservoirs by Updating Fault Properties Using 4D Seismic Results and Ensemble Kalman Filter. Paris, SPE.

[11] Etienam, C., Villegas, R., Babaei, M & Dorn,O 2018. An Improved Reservoir Model Calibration through Sparsity Promoting ES-MDA. ECMI conference proceeding. 20th European Conference on Mathematics for Industry (ECMI) 18-22 June 2018, Budapest, Hungary 

[12] Etienam C. 4D Seismic History Matching Incorporating Unsupervised Learning, Society of Petroleum Engineers B (2019 June), 10.2118/195500-MS

[13] Etienam C., Law, J.H L., Wade, S., Ultra-fast Deep Mixtures of Gaussian Process Experts. https://arxiv.org/abs/2006.13309.

[14] Evensen, G., 2003. The Ensemble Kalman Filter: Theoretical formulation and practical implementation. Ocean Dynamics, 53(4), pp. 343-367.

[15] Evensen, G.,2009. The ensemble Kalman filter for combined state and parameter estimation, IEEE Control Systems Magazine, pp. 83-104.

[16] Evensen, G., Eikrem, K, S 2018. Conditioning reservoir models on rate data using ensemble smoothers.Computational Geosciences,22(5), pp.1251-1270.

[17] Haibin, C. & Zhang, D., 2015. Jointly updating the mean size and spatial distribution of facies in reservoir history matching. Computational Geosciences.

[18] Hanke, M.,1997. A regularizing Levenberg-Marquardt scheme, with applications to inverse groundwater filtration problems. Inverse problems 13(1), pp. 79–95 

[19] Iglesias, M.A., Dawson, C.2013. The regularizing Levenberg-Marquardt scheme for history matching of petroleum reservoirs. Computational Geosciences 17, pp.1033–1053 

[20] Jafarpour, B., 2011. Wavelet reconstruction of geologic facies from nonlinear dynamic flow measurements. Geosciences and Remote sensing, IEEE Transactions, 49(5), pp. 1520-1535.

[21] Jafarpour, B. & McLaughlin, D. B., 2007. History matching with an ensemble Kalman filter and discrete cosine parametrization. Anaheim, California, SPE.

[22] Khaninezhad, M. M., Jafarpour, B. & Li, L., 2012. Sparse geologic dictionaries for subsurface flow model calibration: Part 1, Inversion formulation. Advances in Water Resources, 39(1), pp. 106-121.

[23] Kim S, Min B, Lee K,& Jeong H, 2018 Integration of an Iterative Update of Sparse Geologic Dictionaries with ES-MDA for History Matching of Channelized Reservoirs. Geofluids, Volume 2018, Article ID 1532868, 21 pages.

[24] Law K J H & Stuart A M, 2012 Evaluating Data Assimilation Algorithms Mon.  Weather  Rev 140 37-57

[25] Liu, E. & Jafarpour, B., 2013. Learning sparse geologic dictionaries from low-rank representations of facies connectivity for flow model calibration. Water resources, Volume 49, pp. 7088-7101.


[26] Lorentzen, R. J., Flornes, M. K. & Naevdal, G., 2012. History matching Channelized Reservoirs Using the Ensemble Kalman Filter. Society of Petroleum Engineers, 17(1).


[27] Luo X, Bhakta T, Jakobsen M, Naevdal G, 2018. Efficient big data assimilation through sparse representation: A 3D benchmark case study in petroleum engineering. PLoS ONE 13(7): e0198586. https://doi.org/10.1371/journal.pone.0198586


[28] Moreno, D. L., 2011. Continuous Facies Updating Using the Ensemble Kalman Filter and the Level set method. s.l., Mathematical Geosciences.

[29] Nocedal, J. and Wright, S.J, 1999.Numerical Optimization,  Springer, New York.

[30] Oliver, D. S. & Chen, Y., 2010. Recent progress on reservoir history matching: a review. s.l.: Computational Geoscience - Springer Science.

[31] Oliver, D. S., Reynolds, A. C. & Liu, N., 2008. Inverse Theory for Petroleum Reservoir Characterization and History Matching. s.l.:Cambridge University Press.


[32] Sana, F, Katterbauer, K , Al-Naffouri, T.Y and Hoteit, I.,2016 Orthogonal matching pursuit for enhanced recovery of sparse geological structures with the ensemble Kalman filter, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 9, no. 4, pp. 1710–1724

[33] Sarma, P., L. J. Durlofsky, and K. Aziz,.2008. Kernel principal component analysis for efficient differentiable parameterization of multipoint geostatistics, Mathematical Geosciences, 40, 3-32, 2008.


[34] Sherman, J., Morrison, W.J.1950. Adjustment of an inverse matrix corresponding to a change in one element of a given matrix. The Annals of Mathematical Statistics 21, 124–127.


[35] Stuart A M ,2010 Inverse problems:  A Bayesian perspective Acta Numerica 19 451-559.

[36] Tarantola, 2005. Inverse problem theory and methods for model parameter estimation. 1 ed. Philadelphia: SIAM.

[37] Tarrahi, M. & Afra, S., 2016. Improved Geological model calibration through Sparsity-promoting Ensemble Kalman Filter. Offshore Technology Conference.

[38] Tropp, J. A. & Gilbert, C. A., 2007. Signal recovery from random measurements via orthogonal matching pursuit. Information Theory, IEEE Transactions on, 53(12), pp. 4655-4666.


[39] Villegas, R., Etienam, C., Dorn, O., & Babaei, M.,2018. The shape and Distributed Parameter Estimation for History Matching using a Modified Ensemble Kalman Filter and Level Sets. ;  Inverse problems Science and Engineering.


[40] William P, Mitchell J,1993 JPEG: Still Image Data Compression Standard, Van Nostrand Reinhold, 

[41] Wu, J., Boucher, A. & Journel, G. A., 2006. A 3D code for mp simulation of continuous and categorical variables: FILTERSIM. SPE.

[42] Zhou, H,   Li, L and. Gómez-Hernández, J.,2012, “Characterizing curvilinear features using the localized normal-score ensemble Kalman filter,” Abstract and Applied Analysis, vol. 2012, Article ID 805707, 18 pages.

[43] L. Yang, D. Zhang, G. E. Karniadakis, Physics-informed generative adversarial networks for stochastic differential equations, arXiv preprint arXiv:1811.02033

[43] J. Adler, O. Oktem, solving ill-posed inverse problems using iterative ¨ deep neural networks, Inverse Problems 33 (12) (2017) 124007. URL http://stacks.iop.org/0266-5611/33/i=12/a=124007

[44] J.-Y. Zhu, R. Zhang, D. Pathak, T. Darrell, A. A. Efros, O. Wang, E. Shechtman, Toward multimodal image-to-image translation, in Advances in Neural Information Processing Systems, 2017, pp. 465–476.

[45] S. Rojas, J. Koplik, Nonlinear flow in porous media, Phys. Rev. E 58 (1998) 4776–4782.doi:10.1103/PhysRevE.58.4776.URL,https://link.aps.org/doi/10.1103/PhysRevE.58.4776

[46] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, Generative adversarial nets, in: Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, K. Q. Weinberger (Eds.), Advances in Neural Information Processing Systems 27, Curran Associates, Inc., 2014, pp. 2672–2680. URL http://papers.nips.cc/paper/5423-generative-adversarial-nets. pdf

[47] M. Raissi, P. Perdikaris, G. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics 378 (2019) 686 – 707. doi:https://doi.org/10.1016/j.jcp.2018.10.045.URL,http://www.sciencedirect.com/science/article/pii/ S0021999118307125

[48] M. Raissi, Forward-backward stochastic neural networks: Deep learning of high-dimensional partial differential equations, arXiv preprint arXiv:1804.07010

[49] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics Informed Deep Learning (Part I): Data-driven solutions of nonlinear partial differential equations, arXiv preprint arXiv:1711.10561

[50] Bishop. Christopher M 2006. Pattern Recognition and Machine Learning (Information Science and Statistics). Springer-Verlag, Berlin, Heidelberg.

[51] Dorn O & Villegas R, 2008 History matching of petroleum reservoirs using a level set technique, Inverse problems Volume: 24   Issue: 3 Article Number: 035015 

[52] Hansen, T.M., Vu. L.T., and Bach, T. 2016. MPSLIB: A C++ class for sequential simulation of multiple-point statistical models, in Softwar X, doi:10.1016/j.softx.2016.07.001. [pdf,www].

[53] David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park, Kody J. H. Law, and Clement Etienam. Cluster, classify, regress: A general method for learning discontinuous functions. Foundations of Data Science, 1(2639-8001-2019-4-491):491, 2019.

[54] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar. Fourier Neural Operator for Parametric Partial Differential Equations. https://doi.org/10.48550/arXiv.2010.08895

[55] Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, Anima Anandkumar.Physics-Informed Neural Operator for Learning Partial Differential Equations. https://arxiv.org/pdf/2111.03794.pdf

