# Overview of ofdft_normflows 

This `ofdft_normflows` directory contains clean up code regarding the energy functionals and the usage of the ODE solver for our 1-D and 3-D case. 

# Energy Functionals 

The `functionals.py` file contains the codes regarding the total energy functional,
    $$E[\rho_{\mathcal{M}}] = T[\rho_{\mathcal{M}}] + V_{\text{H}}[\rho_{\mathcal{M}}] +  V_{\text{e-N}}[\rho_{\mathcal{M}}]  + E_{\text{XC}}[\rho_{\mathcal{M}}],$$

where $\rho_{\mathcal{M}}(\mathbf{x})$ is already defined [here](https://github.com/RodrigoAVargasHdz/ofdft_normflows/blob/ml4phys2023/README.md). The total energy functional ($E[\rho_{\mathcal{M}}]$) is composed with the total kinetic energy ($T[\rho_{\mathcal{M}}]$), the Hartree potential ($V_{\text{H}}[\rho_{\mathcal{M}}]$), the external potential ($V_{\text{e-N}}[\rho_{\mathcal{M}}]$) and the Dirac exchange ($E_{X}[\rho_{\mathcal{M}}]$) functionals. The differences between the 1-D and 3-D case will be presented in the next sections. 

## 1-D Case
    
Considering a one-dimensional model for diatomic molecules, the total kinetic energy is estimated by the sum of the Thomas-Fermi ($T_{\text{TF}}$) and  Weizsäcker ($T_{\text{W}}$)  functionals; $T[\rho_{\mathcal{M}}] = T_{\text{TF}}[\rho_{\mathcal{M}}] + T_{\text{W}}[\rho_{\mathcal{M}}]$,

$$T_{\text{TF}}[\rho_{\mathcal{M}}] = \frac{\pi^2}{24} \int \left(\\rho_{\mathcal{M}}(x) \right)^{3} \mathrm{d}x.$$

$$T_{\text{W}}[\rho_{\mathcal{M}}] = \frac{\lambda_0}{8} \int \frac{(\nabla \rho_{\mathcal{M}}(x))^2}{\rho_{\mathcal{M}}} \mathrm{d}x, $$

where the phenomenological parameter $\lambda_0$ was set to 0.2. 

We rewrite the Weizsäcker functional in terms of the score function, 
    $$T_{\text{W}}[\rho_{\mathcal{M}}] = \frac{\lambda_0}{8} \int  \left(\nabla \log \\rho_{\mathcal{M}}(x) \right)^2  \rho_{\mathcal{M}}(x) \mathrm{d}x,$$
in order to use use memory-efficient gradients for optmizing $T_{\text{W}}[\rho_{\mathcal{M}}]$. 

The Hartree ($V_{\text{H}}[\rho_{\mathcal{M}}]$) potential and the external potential ($V_{\text{e-N}}[\rho_{\mathcal{M}}]$) functionals both are defined by a soft version,

   $$ V_{\text{H}}[\rho_{\mathcal{M}}] = \int \int v_{\text{H}}(x) \rho_{\mathcal{M}}(x)\rho_{\mathcal{M}}(x')\mathrm{d}x \mathrm{d}x' = \int \int \frac{ \rho_{\mathcal{M}}(x)\rho_{\mathcal{M}}(x')}{\sqrt{1 + |x - x'|^2}} \mathrm{d}x \mathrm{d}x',$$
   
   $$V_{\text{e-N}}[\rho_{\mathcal{M}}] = \int v_{\text{e-N}}(x) \rho_{\mathcal{M}}(x) \mathrm{d}x = -\int  \left  ( \frac{Z_\alpha}{\sqrt{1 + | x - R/2 |^2}} + \frac{Z_\beta}{\sqrt{1 + | x + R/2 |^2}} \right )\rho_{\mathcal{M}}(x) \mathrm{d}x,$$

where the atomic numbers $Z_\alpha$ and $Z_\beta$ are the atomic numbers, $N_e$ is the number of valence electrons and $R$ is the interatomic distance. 

And the exchange-correlation functional $E_{\text{XC}}[\rho_{\mathcal{M}}]$ is given by, 
    $$E_{\text{XC}}[\rho_{\mathcal{M}}] = \int  \epsilon_{\text{XC}} \rho_{\mathcal{M}}$$,
where $\epsilon_{\text{XC}}$ is, 

$$\epsilon_{\text{XC}} (r_{\text{s}},\zeta) = \frac{a_{\zeta} + b_{\zeta}r_{\text{s}} + c_{\zeta} r_{\text{s}}^{2}}{1 + d_{\zeta} r_{\text{s}} + e_{\zeta} r_{\text{s}}^2 + f_{\zeta} r_{\text{s}}^3} + \frac{g_{\zeta} r_{\text{s}} \ln[{r_{\text{s}} + \alpha_{\zeta} r_{\text{s}}^{\beta_{\zeta}} }]}{1 + h_{\zeta} r_{\text{s}}^2}.$$

For all one-dimensional simulations, we used, the Wigner-Seitz radius $r_{\text{s}} = \tfrac{1}{2\rho_{\mathcal{M}}}$ and $\zeta = 0$ (unpolarized density). 

## 3-D Case 

To demonstrate the capability to use CNF for real-space simulations, we considered the optimization of $H_2$ and $H_{2}O$. For both chemical systems, we considered the same total energy functional where the differences are in the Hartree-Potential, $$v_{\text{e-N}}(\mathcal{x}) = -\sum_i \frac{Z_i}{\|(\boldsymbol{x}) - \mathbf{R}_i\|},$$ where no soft form approximation was used and $Z_i$ is the atomic number of the $i$ atom.

 The Thomas-Fermi ($T_{\text{TF}}$) functional, 

$$T_{\text{TF}}[\rho_{\mathcal{M}}] = \frac{3}{10}(3\pi^2)^{2/3} \int \left(\rho_{\mathcal{M}}(\boldsymbol{x}) \right)^{5/3} \mathrm{d}x.$$

And in the exchange-correlation functional that is composed of the sum of the exchange (X) and correlation (C) terms,
$$E_{\text{XC}}[\rho_{\mathcal{M}}] = \int  \epsilon_{\text{XC}} \rho_{\mathcal{M}}(\boldsymbol{x}) \mathrm{d}\boldsymbol{x} = \int  \epsilon_{\text{X}} \rho_{\mathcal{M}}(\boldsymbol{x}) \mathrm{d}\boldsymbol{x} + \int  \epsilon_{\text{C}} \rho_{\mathcal{M}}(\boldsymbol{x})  \mathrm{d}\boldsymbol{x}.$$

We report all different $\epsilon_{\text{X}}$ and $\epsilon_{\text{C}}$ used in the simulations,
$$\epsilon_{\text{X}}^{\text{LDA}} = -\frac{3}{4} \left(\frac{3}{\pi}\right)^{1/3} \rho_{\mathcal{M}}(\boldsymbol{x})^{1/3}$$
$$\epsilon_{\text{X}}^{\text{B88}} = -\beta \frac{X^2}{\left(1 + 6 \beta X \sinh^{-1}(X) \right)} (\boldsymbol{x})^{1/3}$$
$$\epsilon_{\text{C}}^{\text{VWN}} = \frac{A}{2} \left[ \ln\left(\frac{y^2}{Y(y)}\right) + \frac{2b}{Q} \tan^{-1} \left(\frac{Q}{2y + b}\right) \frac{b y_0}{Y(y_0)} \left[\ln\left(\frac{(y-y_0)^2}{Y(y)}\right) + \frac{2(b+2y_0)}{Q}\tan^{-1}  \left(\frac{Q}{2y+b}\right) \right] \right]$$
$$\epsilon_{\text{C}}^{\text{PW92}} = -2A(1+\alpha_1 r_{\text{s}}) \ln \left[{1 + \frac{1}{2A(\beta_1 r_{\text{s}}^{1/2} + \beta_2 r_{\text{s}} + \beta_3 r_{\text{s}}^{3/2} + \beta_4 r_{\text{s}}^{2}}}\right]$$

where $r_{\text{s}} = \left ( \frac{3}{4\pi \rho_{\mathcal{M}}}  \right)^{\frac{1}{3}}$. 
For $\epsilon_{\text{C}}^{\text{VWN}}$, $y = r_{\text{s}}^{1/2}$, $Y(y) = y^2 + by + c$, $Q = \sqrt{4c - b^2}$, and the constants $b$, $c$ and $y_0$ are given in the SI of the paper. 


