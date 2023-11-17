## Energy Functionals 

We first considered a one-dimensional model for diatomic molecules where the total energy functional is defined as,
    $$E[\rho_{\mathcal{M}}] = T[\rho_{\mathcal{M}}] + V_{\text{H}}[\rho_{\mathcal{M}}] +  V_{\text{e-N}}[\rho_{\mathcal{M}}]  + E_{X}[\rho_{\mathcal{M}}].$$ 
    
The total kinetic energy is estimated by the sum of the Thomas-Fermi ($T_{\text{TF}}$) and  Weizsäcker ($T_{\text{W}}$)  functionals; $T[\rho_{\mathcal{M}}] = T_{\text{TF}}[\rho_{\mathcal{M}}] + T_{\text{W}}[\rho_{\mathcal{M}}]$. 

$$T_{\text{TF}}[\rho_{\mathcal{M}}] = \frac{\pi^2}{24} \int \left(\\rho_{\mathcal{M}}(x) \right)^{3} \mathrm{d}x,$$

$$T_{\text{W}}[\rho_{\mathcal{M}}] = \frac{\lambda}{8} \int \frac{(\nabla \rho_{\mathcal{M}}(x))^2}{\rho_{\mathcal{M}}} \mathrm{d}x, $$

where the phenomenological parameter $\lambda$ was set to 0.2. We can rewrite the Weizsäcker functional in terms of the score function, 
    $$T_{\text{W}}[\rho_{\mathcal{M}}] = \frac{\lambda}{8} \int  \left(\nabla \log \\rho_{\mathcal{M}}(x) \right)^2  \rho_{\mathcal{M}}(x) \mathrm{d}x.$$

The Hartree ($V_{\text{H}}[\rho_{\mathcal{M}}]$) potential and the external potential ($V_{\text{e-N}}[\rho_{\mathcal{M}}]$) functionals both are defined by a soft version,

   $$ V_{\text{H}}[\rho_{\mathcal{M}}] = \int \int v_{\text{H}}(x) \rho_{\mathcal{M}}(x)\rho_{\mathcal{M}}(x')\mathrm{d}x \mathrm{d}x' = \int \int \frac{ \rho_{\mathcal{M}}(x)\rho_{\mathcal{M}}(x')}{\sqrt{1 + |x - x'|^2}} \mathrm{d}x \mathrm{d}x',$$
   
   $$V_{\text{e-N}}[\rho_{\mathcal{M}}] = \int v_{\text{e-N}}(x) \rho_{\mathcal{M}}(x) \mathrm{d}x = -\int  \left  ( \frac{Z_\alpha}{\sqrt{1 + | x - R/2 |^2}} + \frac{Z_\beta}{\sqrt{1 + | x + R/2 |^2}} \right )\rho_{\mathcal{M}}(x) \mathrm{d}x,$$

where the atomic numbers $Z_\alpha$ and $Z_\beta$ are the atomic numbers, $N_e$ is the number of valence electrons and $R$ is the interatomic distance. 

We only consider the Dirac exchange functional, 
    $$E_{\text{X}}[\rho_{\mathcal{M}}] = -\frac{3}{4} \left(\frac{3}{\pi} \right)^{1/3} \int \rho_{\mathcal{M}}(x)^{4/3} \mathrm{d}x.$$
