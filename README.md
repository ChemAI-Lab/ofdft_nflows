# OF-DFT with Continuous-time Normalizing Flows

This repository contains the original implementation of the experiments for ["Orbital-Free Density Functional Theory with Continuous Normalizing Flows"](archive).

--------------------

## Sketch of the algorithm

In orbital-free density functional theory, the ground-state density is found by solving a constrained optimization problem,
    $$\min_{\rho_\mathcal{M}}  E[\rho_\mathcal{M}] - \mu \left(\int \rho_\mathcal{M} \mathrm{d} \mathbf{x} - N_{e} \right ) \ \text{s.t. } \rho_\mathcal{M} \geq 0,$$ 
where $\mu$ acts as the Lagrange multiplier associated with the normalization constraint on the total number of particles $\left(N_{e}\right)$. These constraints, which enforce both positivity and normalization, 
ensure the attainment of physically valid solutions.

In this work, we present an alternative constraint-free approach to solve for the ground-state density by a continuous-time normalizing flow (NF) ansatz, allowing us to reframe the OF-DFT variational problem as a Lagrangian-free optimization problem for molecular densities in real space,
     $$\min_{\rho_\mathcal{M}}  E[\rho_\mathcal{M}] \cancel{- \mu \left(\int \rho_\mathcal{M} \mathrm{d} \mathbf{x} - N_{e} \right )} \ \text{s.t. } \rho_\mathcal{M} \geq 0.$$ 


We parameterize the electron density $\rho_\mathcal{M} := N_{e}  \rho_{\phi}(\mathbf{x})$, where $\rho_{\phi}$ is a NF, this form is also referred to as the *shape factor*. The samples are drawn from the base distribution $\rho_0$ and transformed by, $$\mathcal{x} = T_\phi(\mathcal{z}) := \mathcal{z} + \int_{t_{0}}^{T} g_\phi(\mathcal{z}(t),t) \mathrm{d}t.$$

## Results

We successfully replicate the electronic density for the one-dimensional Lithium hydride molecule with varying interatomic distances, as well as comprehensive simulations of hydrogen and water molecules, all conducted in
Cartesian space.

## Running the code 

# 1-D 
For Lithium hydride ($\texttt{LiH}$) molecule, simulations can be run in the following way, 
```
python LiH.py
                --epochs <number of iterations>
                --bs <batch size>
                --N <number of valence electrons>
                --sched <learning rate schedule>
                --R <interatomic distances>
                --Z <atomic number> 
```
The default functionals can be found in the directory [ofdft_normflows](https://github.com/RodrigoAVargasHdz/ofdft_normflows/tree/ml4phys2023/ofdft_normflows#readme)

|$\rho_{{\cal M}}$ of $\texttt{LiH}$ for various inter-atomic distances.|The change of $\rho_{{\cal M}}$ and $T_\phi(\mathcal{z})$ during training.|
|:--:|:--:|
|![](https://github.com/RodrigoAVargasHdz/ofdft_normflows/blob/ml4phys2023/Assets/Figure_1.png)|![](https://github.com/RodrigoAVargasHdz/ofdft_normflows/blob/ml4phys2023/Assets/neural_ode_2_gif.gif)|

# 3-D 
For water ($\texttt{H2O}$) and hydrogen ($\texttt{H2}$) molecules, simulations can be run in the following way,
```
python H2_mol_ofdft_min.py
                            --epochs <number of iterations>
                            --bs <batch size>
                            --lr <initial learning rate>
                            --sched <learning rate schedule>          
```
The default kinetic energy functional is the sum of the [Thomas-Fermi and Weizsäcker](https://github.com/RodrigoAVargasHdz/ofdft_normflows/tree/ml4phys2023/ofdft_normflows#readme), however, ``` --kin <name> ``` could be used to select others. 

## Dependencies

1. [DeepMind JAX Ecosystem]([jax.readthedocs.io/](https://deepmind.google/discover/blog/using-jax-to-accelerate-our-research/))
2. [Flax](flax.readthedocs.io/)
6. [PySCF](pyscf.org/)
