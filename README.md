# OF-DFT with Continuous Normalizing Flows

This repository contains the original implementation of the experiments for ["Orbital-Free Density Functional Theory with Continuous Normalizing Flows"](archive).

--------------------

## Sketch of the algorithm

In orbital free density functional theory, the ground-state density is found by solving a constrained optimization problem,
    $$\min_{\rho(\mathbf{x})}  E[\rho(\mathbf{x})] - \mu \left(\int \rho(\mathbf{x}) \mathrm{d} \mathbf{x} - N_{e} \right ) \ \text{s.t. } \rho(\mathbf{x}) \geq 0,$$ 
where $\mu$ acts as the Lagrange multiplier associated with the normalization constraint on the total number of particles $\left(N_{e}\right)$. These constraints, which enforce both positivity and normalization, 
ensure the attainment of physically valid solutions.

In this work, we present an alternative constraint-free approach to solve for the ground-state density by a continuous-time normalizing flow (NF) ansatz, allowing us to reframe the OF-DFT variational problem as a Lagrangian-free optimization problem for molecular densities in real space,
     $$\min_{\rho(\mathbf{x})}  E[\rho(\mathbf{x})] \cancel{- \mu \left(\int \rho(\mathbf{x}) \mathrm{d} \mathbf{x} - N_{e} \right )} \ \text{s.t. } \rho(\mathbf{x}) \geq 0.$$ 


## Results

We successfully replicate the electronic density for the one-dimensional Lithium hydride molecule with varying interatomic distances, as well as comprehensive simulations of hydrogen and water molecules, all conducted in
Cartesian space.

# 1-D 
|Ground state electronic density of $\texttt{LiH}$ for various inter-atomic distances.|Transformation of our base distribution into the target distribution.|
|:--:|:--:|
|![](https://github.com/RodrigoAVargasHdz/ofdft_normflows/blob/ml4phys2023/Assets/Figure_1.png)|![](https://github.com/RodrigoAVargasHdz/ofdft_normflows/blob/ml4phys2023/Assets/neural_ode_2_gif.gif)|

