# Overview of ofdft_normflows 

This `ofdft_normflows` directory contains clean up code regarding the usage of the ODE solver regarding 

# Energy Functionals 

The `functionals.py` file contains the codes regarding the total energy functional,
    $$E[\rho_{\mathcal{M}}] = T[\rho_{\mathcal{M}}] + V_{\text{H}}[\rho_{\mathcal{M}}] +  V_{\text{e-N}}[\rho_{\mathcal{M}}]  + E_{X}[\rho_{\mathcal{M}}],$$

$\rho_{\mathcal{M}}(\mathbf{x})$ is already define [here](https://github.com/RodrigoAVargasHdz/ofdft_normflows/tree/ml4phys2023/ofdft_normflows#readme). There are differences between the functionals in the 1-D and the 3-D case. 

## 1-D Case
    
Considering a one-dimensional model for diatomic molecules, the total kinetic energy is estimated by the sum of the Thomas-Fermi ($T_{\text{TF}}$) and  Weizsäcker ($T_{\text{W}}$)  functionals; $T[\rho_{\mathcal{M}}] = T_{\text{TF}}[\rho_{\mathcal{M}}] + T_{\text{W}}[\rho_{\mathcal{M}}]$,

$$T_{\text{TF}}[\rho_{\mathcal{M}}] = \frac{\pi^2}{24} \int \left(\\rho_{\mathcal{M}}(x) \right)^{3} \mathrm{d}x.$$

$$T_{\text{W}}[\rho_{\mathcal{M}}] = \frac{\lambda}{8} \int \frac{(\nabla \rho_{\mathcal{M}}(x))^2}{\rho_{\mathcal{M}}} \mathrm{d}x, $$

where the phenomenological parameter $\lambda$ was set to 0.2. 

We rewrite the Weizsäcker functional in terms of the score function, 
    $$T_{\text{W}}[\rho_{\mathcal{M}}] = \frac{\lambda}{8} \int  \left(\nabla \log \\rho_{\mathcal{M}}(x) \right)^2  \rho_{\mathcal{M}}(x) \mathrm{d}x,$$
in order to use use memory-efficient gradients for optmizing $T_{\text{W}}[\rho_{\mathcal{M}}]$. 

The Hartree ($V_{\text{H}}[\rho_{\mathcal{M}}]$) potential and the external potential ($V_{\text{e-N}}[\rho_{\mathcal{M}}]$) functionals both are defined by a soft version,

   $$ V_{\text{H}}[\rho_{\mathcal{M}}] = \int \int v_{\text{H}}(x) \rho_{\mathcal{M}}(x)\rho_{\mathcal{M}}(x')\mathrm{d}x \mathrm{d}x' = \int \int \frac{ \rho_{\mathcal{M}}(x)\rho_{\mathcal{M}}(x')}{\sqrt{1 + |x - x'|^2}} \mathrm{d}x \mathrm{d}x',$$
   
   $$V_{\text{e-N}}[\rho_{\mathcal{M}}] = \int v_{\text{e-N}}(x) \rho_{\mathcal{M}}(x) \mathrm{d}x = -\int  \left  ( \frac{Z_\alpha}{\sqrt{1 + | x - R/2 |^2}} + \frac{Z_\beta}{\sqrt{1 + | x + R/2 |^2}} \right )\rho_{\mathcal{M}}(x) \mathrm{d}x,$$

where the atomic numbers $Z_\alpha$ and $Z_\beta$ are the atomic numbers, $N_e$ is the number of valence electrons and $R$ is the interatomic distance. 

And we consider the Dirac exchange functional, 
    $$E_{\text{X}}[\rho_{\mathcal{M}}] = -\frac{3}{4} \left(\frac{3}{\pi} \right)^{1/3} \int \rho_{\mathcal{M}}(x)^{4/3} \mathrm{d}x.$$

## 3-D Case 

To demonstrate the capability to use CNF for real-space simulations, we considered the optimization of $H_2$ and $H_{2}O$. For both chemical systems, we considered the same total energy functional where the differences are in the Hartree-Potential, $$v_{\text{e-N}}(\mathcal{x}) = -\sum_i \frac{Z_i}{\|\mathcal{x} - \mathbf{R}_i\|},$$ where no soft form approximation was used and $Z_i$ is the atomic number of the $i$ atom.

And in the Thomas-Fermi ($T_{\text{TF}}$) functional, 

$$T_{\text{TF}}[\rho_{\mathcal{M}}] = \frac{3}{10}(3\pi^2)^{2/3} \int \left(\rho_{\mathcal{M}}(x) \right)^{5/3} \mathrm{d}x.$$


