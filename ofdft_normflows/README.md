## Energy Functionals 

We first considered a one-dimensional model for diatomic molecules where the total energy functional is defined as,
    $$E[\rho_{\mathcal{m}}] = T[\rho_{\mathcal{m}}] + V_{\text{H}}[\rho_{\mathcal{m}}] +  V_{\text{e-N}}[\rho_{\mathcal{m}}]  + E_{X}[\rho_{\mathcal{m}}]$$. 
    
The total kinetic energy is estimated by the sum of the Thomas-Fermi ($T_{\text{TF}}$) and  Weizsäcker ($T_{\text{W}}$)  functionals; $T[\rho_{\mathcal{m}}] = T_{\text{TF}}[\rho_{\mathcal{m}}] + T_{\text{W}}[\rho_{\mathcal{m}}]$. 

$$T_{\text{TF}}[\rho_{\mathcal{m}}] = \frac{\pi^2}{24} \int \left(\\rho_{\mathcal{m}}(x) \right)^{3} \mathrm{d}x,$$

$$T_{\text{W}}[\rho_{\mathcal{m}}] = \frac{\lambda}{8} \int \frac{(\nabla \rho_{\mathcal{m}}(x))^2}{\rho_{\mathcal{m}}} \mathrm{d}x, $$

where the phenomenological parameter $\lambda$ was set to 0.2. We can rewrite the Weizsäcker functional in terms of the score function, 
    $$T_{\text{W}}[\rho_{\mathcal{m}}] = \frac{\lambda}{8} \int  \left(\nabla \log \\rho_{\mathcal{m}}(x) \right)^2  \rhom(x) \mathrm{d}x.$$

The Hartree ($V_{\text{H}}[\rho_{\mathcal{m}}]$) potential and the external potential ($V_{\text{e-N}}[\rho_{\mathcal{m}}]$) functionals both are defined by a soft version,
   $$ V_{\text{H}}[\rho_{\mathcal{m}}] = \int \int v_{\text{H}}(x) \rho_{\mathcal{m}}(x)\rho_{\mathcal{m}}(x')\mathrm{d}x \mathrm{d}x' = \int \int \frac{ \rho_{\mathcal{m}}(x)\rho_{\mathcal{m}}(x')}{\sqrt{1 + |x - x'|^2}} \mathrm{d}x \mathrm{d}x',$$
    $$V_{\text{e-N}}[\rho_{\mathcal{m}}] = \int v_{\text{e-N}}(x) \rho_{\mathcal{m}}(x) \mathrm{d}x = -\int  \left  ( \frac{Z_\alpha}{\sqrt{1 + | x - R/2 |^2}} + \frac{Z_\beta}{\sqrt{1 + | x + R/2 |^2}} \right )\rhom(x) \mathrm{d}x,$$

where the atomic numbers $Z_\alpha$ and $Z_\beta$ are the atomic numbers, $N_e$ is the number of valence electrons and $R$ is the interatomic distance. 

We only consider the Dirac exchange functional, 
    $$E_{\text{X}}[\rho_{\mathcal{m}}] = -\frac{3}{4} \left( \frac{3}{\pi} \right)^{1/3} \int \rho_{\mathcal{m}}(x)^{4/3} \ \ \mathrm{d}x.$$
