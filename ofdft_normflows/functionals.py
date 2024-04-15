from typing import Any, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, hessian, jacrev, lax

#from utils import *
from ofdft_normflows.utils import *


Array = jax.Array
BHOR = 1.  # 1.8897259886  # 1AA to BHOR


# ------------------------------------------------------------------------------------------------------------
# KINETIC FUNCTIONALS
# ------------------------------------------------------------------------------------------------------------

def _kinetic(name: str = 'TF'):
    if name.lower() == 'tf' or name.lower() == 'thomas_fermi':
        def wrapper(*args):
            return thomas_fermi(*args)
    elif name.lower() == 'tf1d' or name.lower() == 'thomas_fermi_1d':
        def wrapper(*args):
            return thomas_fermi_1D(*args)
    elif name.lower() == 'w' or name.lower() == 'weizsacker':
        def wrapper(*args):
            return weizsacker(*args)
    elif name.lower() == 'tf-w' or name.lower() == 'thomas_fermi_weizsacker':
        def wrapper(*args):
            return thomas_fermi(*args) + weizsacker(*args)
    elif name.lower() == 'w1d' or name.lower() == 'weizsacker1d':
        def wrapper(*args):
            return weizsacker(*args)
    elif name.lower() == 'tfw_1d' or name.lower() == 'thomas_fermi_weizsacker_1d':
        def wrapper(*args): 
            return thomas_fermi_1D(*args) + weizsacker(*args)
    elif name.lower() == 'k' or name.lower() == 'kinetic':
        def wrapper(*args):
            return kinetic(*args)
    return wrapper


# @ partial(jit,  static_argnums=(3,))
@jit
def kinetic(den: Any, lap_sqrt_den: Any, Ne: int) -> jax.Array: 
    rho_val = 1./(den+1E-4)**0.5  # for numerical stability
    return -0.5*jnp.multiply(rho_val, lap_sqrt_den)


@jit
def weizsacker(den: Array, score: Array, Ne: int) -> jax.Array: # type: ignore
    """    
    l = 0.2 (W Stich, EKU Gross., Physik A Atoms and Nuclei, 309(1):511, 1982.)
    T_{\text{Weizsacker}}[\rho] &=& \frac{\lambda}{8} \int \frac{(\nabla \rho)^2}{\rho} dr = 
                            &=&    \frac{\lambda}{8} \int  \rho \left(\frac{(\nabla \rho)}{\rho}\right)^2 dr\\
    T_{\text{Weizsacker}}[\rho] = \mathbb{E}_{\rho} \left[ \left(\frac{(\nabla \rho)}{\rho}\right)^2 \right]


    Args:
        score (Array): _description_
        Ne (int): _description_
        l (Any, optional): _description_. Defaults to .2.

    Returns:
        jax.Array: _description_
    """
    l = .2
    score_sqr = jnp.einsum('ij,ij->i', score, score)
    return (l*Ne/8.)*lax.expand_dims(score_sqr, (1,))



@jit
def thomas_fermi(den: Array, score: Array, Ne: int) -> jax.Array:
    """
    T_{\text{TF}}[\rho] &=& \frac{3}{10}(3\pi^2)^{2/3} \int ( \rho)^{5/3} dr \\
    T_{\text{TF}}[\rho] = \mathbb{E}_{\rho} \left[ ( \rho)^{2/3} \right]


    Args:
        params (Any): _description_
        den (Array): _description_
        Ne (int): _description_

    Returns:
        jax.Array: _description_
    """

    val = (den)**(2/3)
    l = (3./10.)*(3.*jnp.pi**2)**(2/3)
    return l*(Ne**(5/3))*val


# @partial(jit,  static_argnums=(3,))
@jit
def thomas_fermi_1D(den: Array, score: Array, Ne: int) -> jax.Array:
    """
    T_{\text{TF}}[\rho] &=& \frac{\pi^2}{12}\int ( \rho)^{3} dr \\
    T_{\text{TF}}[\rho] = \frac{\pi^2}{12}\mathbb{E}_{\rho} \left[ ( \rho)^{2} \right]

    Args:
        params (Any): _description_
        u (Array): _description_
        Ne (int): _description_
        fun (callable): _description_

    Returns:
        jax.Array: _description_
    """
    
    den_sqr = den*den
    l = (jnp.pi*jnp.pi)/24.
    return l*(Ne**3)*den_sqr
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# EXCHANGE-CORRELATION FUNCTIONALS 
# ------------------------------------------------------------------------------------------------------------
# def _correlation(name: str = 'correlation'): 
#     if name.lower() == 'correlation': 
#         def wrapper(*args): 
#             return uniform_gas(*args)

def _exchange_correlation(name: str = 'XC'):
    if name.lower() == 'dirac' or name.lower() == 'dirac_exchange':
        def wrapper(*args):
            return Dirac_exchange(*args) 
    if name.lower() == 'xc_1d' or name.lower() == 'exchange_correlation_1d':
        def wrapper(*args):
            return exchange_correlation_one_dimensional(*args)
    if name.lower() == 'vwn_c_e' or name.lower() == 'correlation_vwn_c_e': 
        def wrapper(*args):
            return correlation_vwn_c_e(*args) 
    if name.lower() == 'pw92_c_e' or name.lower() == 'correlation_pw92_c_e': 
        def wrapper(*args):
            return correlation_pw92_c_e(*args) 
    if name.lower() == 'b88_x_e': 
        def wrapper(*args):
            return b88_x_e(*args)
    if name.lower() == 'dirac_b88_x_e': 
        def wrapper(*args): 
            return Dirac_exchange(*args) + b88_x_e(*args)
    return wrapper

@jit 
def correlation_vwn_c_e(den: Array, Ne:int):
    """
    VWN correlation functional
    See original paper eq 4.4 in https://cdnsciencepub.com/doi/abs/10.1139/p80-159
    See also text after eq 8.9.6.1 in https://www.theoretical-physics.com/dev/quantum/dft.html

    Parameters
    ----------
    rho : Float[Array, "grid spin"]
    clip_cte : float, optional
        small clip to prevent numerical instabilities.

    Returns
    -------
    Float[Array, "grid"]

    In our case the "grip spin" is equal to zero. 
    """
    
    A = 0.0621814
    b = 3.72744
    c = 12.9352
    x0 = -0.10498
    clip_cte = 1e-30
    den = jnp.where(den > clip_cte, den, 0.0)
    log_den = jnp.log2(den)
    log_den = jnp.log2(jnp.clip(den, a_min=clip_cte))
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_den / 3.0
    log_x = log_rs / 2
    rs = 2.**log_rs
    x = 2.**log_x

    X = 2. ** (2. * log_x) + 2. ** (log_x + jnp.log2(b)) + c
    X0 = x0**2 + b * x0 + c

    Q = jnp.sqrt(4 * c - b**2)

    # check eq with https://github.com/ElectronicStructureLibrary/libxc/blob/master/maple/vwn.mpl
    e_PF = (
        A
        / 2.
        * (
            2. * jnp.log(x)
            - jnp.log(X)
            + 2. * b / Q * jnp.arctan(Q / (2. * x + b))
            - b
            * x0
            / X0
            * (jnp.log((x - x0) ** 2. / X) + 2. * (2. * x0 + b) / Q * jnp.arctan(Q / (2. * x + b)))
        )
    ) 

    e_correlation = Ne*e_PF  

    return e_correlation

@jit 
def correlation_pw92_c_e(den: Array, Ne:int):
    """
    Eq 10 in
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.45.13244

    Parameters
    ----------
    rho : Float[Array, "grid spin"]
    clip_cte : float, optional
        small clip to prevent numerical instabilities.

    Returns
    -------
    """
    clip_cte: float = 1e-30
    A_ = 0.031091
    alpha1 = 0.21370
    beta1 = 7.5957
    beta2 =3.5876
    beta3 = 1.6382
    beta4 = 0.49294

    log_den = jnp.log2(jnp.clip(den.sum(axis=1, keepdims=True), a_min=clip_cte))
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_den / 3.0
    brs_1_2 = 2 ** (log_rs / 2 + jnp.log2(beta1))
    ars = 2 ** (log_rs + jnp.log2(alpha1))
    brs = 2 ** (log_rs + jnp.log2(beta2))
    brs_3_2 = 2 ** (3 * log_rs / 2 + jnp.log2(beta3))
    brs2 = 2 ** (2 * log_rs + jnp.log2(beta4))

    e_PF = -2 * A_ * (1 + ars) * jnp.log(1 + (1 / (2 * A_)) / (brs_1_2 + brs + brs_3_2 + brs2))
    
    e_correlation = Ne*e_PF 

    return e_correlation

@jit
def exchange_correlation_one_dimensional(den:Array, Ne:int): 
    rs = 1/(2*Ne*den)
    a0 = -0.8862269
    b0 = -2.1414101
    c0 = 0.4721355
    d0 = 2.81423
    e0 = 0.529891
    f0 = 0.458513
    g0 = -0.202642
    h0 = 0.470876
    alpha0 = 0.104435
    beta0 = 4.11613
    n1 = a0 + b0*rs + c0*rs**2 
    d1 = 1 + d0*rs + e0*rs**2 + f0*rs**3 
    f1 = n1/d1 
    n2 = g0*rs*jnp.log(rs + alpha0*rs**beta0)
    d2 = 1 + h0*rs**2 
    f2 = n2/d2 
    return Ne*(f1 + f2)

@jit
def pw92_c_e(den: Array, Ne: int):
    """
    Eq 10 in
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.45.13244

    Parameters
    ----------
    rho : Float[Array, "grid spin"]
    clip_cte : float, optional
        small clip to prevent numerical instabilities.

    Returns
    -------
    """
    clip_cte: float = 1e-30
    A_ = 0.031091
    alpha1 = 0.21370
    beta1 = 7.5957
    beta2 =3.5876
    beta3 = 1.6382
    beta4 = 0.49294

    log_den = jnp.log2(jnp.clip(den.sum(axis=1, keepdims=True), a_min=clip_cte))
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_den / 3.0
    brs_1_2 = 2 ** (log_rs / 2 + jnp.log2(beta1))
    ars = 2 ** (log_rs + jnp.log2(alpha1))
    brs = 2 ** (log_rs + jnp.log2(beta2))
    brs_3_2 = 2 ** (3 * log_rs / 2 + jnp.log2(beta3))
    brs2 = 2 ** (2 * log_rs + jnp.log2(beta4))

    e_PF = -2 * A_ * (1 + ars) * jnp.log(1 + (1 / (2 * A_)) / (brs_1_2 + brs + brs_3_2 + brs2))

    return e_PF * Ne

@jit
def vwn_c_e(den: Array, Ne:int) :
    """
    VWN correlation functional
    See original paper eq 4.4 in https://cdnsciencepub.com/doi/abs/10.1139/p80-159
    See also text after eq 8.9.6.1 in https://www.theoretical-physics.com/dev/quantum/dft.html

    Parameters
    ----------
    rho : Float[Array, "grid spin"]
    clip_cte : float, optional
        small clip to prevent numerical instabilities.

    Returns
    -------
    Float[Array, "grid"]
    """
    A = 0.0621814
    b = 3.72744
    c = 12.9352
    x0 = -0.10498
   
    clip_cte = 1e-30
    den = jnp.where(den > clip_cte, den, 0.0)
    log_den = jnp.log2(den)
    log_den = jnp.log2(jnp.clip(den, a_min=clip_cte))
   
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_den / 3.0
    log_x = log_rs / 2
    rs = 2.**log_rs
    x = 2.**log_x

    X = 2. ** (2. * log_x) + 2. ** (log_x + jnp.log2(b)) + c
    X0 = x0**2 + b * x0 + c

    Q = jnp.sqrt(4 * c - b**2)

    # check eq with https://github.com/ElectronicStructureLibrary/libxc/blob/master/maple/vwn.mpl
    e_PF = (
        A
        / 2.
        * (
            2. * jnp.log(x)
            - jnp.log(X)
            + 2. * b / Q * jnp.arctan(Q / (2. * x + b))
            - b
            * x0
            / X0
            * (jnp.log((x - x0) ** 2. / X) + 2. * (2. * x0 + b) / Q * jnp.arctan(Q / (2. * x + b)))
        )
    )
    
    return e_PF * Ne

@jit 
def b88_x_e(den: Array, score: Array,Ne: int):
    r"""
    B88 exchange functional
    See eq 8 in https://journals.aps.org/pra/abstract/10.1103/PhysRevA.38.3098
    See also https://github.com/ElectronicStructureLibrary/libxc/blob/4bd0e1e36347c6d0a4e378a2c8d891ae43f8c951/maple/gga_exc/gga_x_b88.mpl#L22

    Parameters
    ----------
    rho : Float[Array, "grid spin"]
    grad_rho : Float[Array, "grid spin dimension"]
    clip_cte : float, optional
        small clip to prevent numerical instabilities.

    Returns
    -------
    Float[Array, "grid"]
    """
    clip_cte =  1e-30
    beta = 0.0042

    den = jnp.clip(den, a_min=clip_cte)

    log_den = jnp.log2(jnp.clip(den, a_min=clip_cte))

    score_sqr = jnp.einsum('ij,ij->i', score, score)
    den_sqr = den*den
    grad_den_norm_sq =lax.expand_dims(score_sqr, (1,)) * den_sqr

    log_grad_den_norm = jnp.log2(jnp.clip(grad_den_norm_sq, a_min=clip_cte)) / 2

    log_x_sigma = log_grad_den_norm - 4 / 3.0 * log_den

    x_sigma = 2**log_x_sigma

    # Eq 2.78 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    b88_e = -(
        beta
        * 2
        ** (
            4 * log_den / 3
            + 2 * log_x_sigma
            - jnp.log2(1 + 6 * beta * x_sigma * jnp.arcsinh(x_sigma))
        )
    )
    
    return b88_e*Ne**(2/3)

@jit
def Dirac_exchange(den: Array,score: Array, Ne: int) -> jax.Array:
    """
    ^{Dirac}E_{\text{x}}[\rho] = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}\int  \rho^{4/3} dr \\
    ^{Dirac}E_{\text{x}}[\rho] = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}\mathbb{E}_{\rho} \left[ \rho^{1/3} \right]

    Args:
        params (Any): _description_
        den (Array): _description_
        Ne (int): _description_

    Returns:
        jax.Array: _description_
    """
    l = -(3/4)*(Ne**(4/3))*(3/jnp.pi)**1/3

    return l*den**(1/3)


# ------------------------------------------------------------------------------------------------------------


def _hartree(name: str = 'HP'):
    if name.lower() == 'hartree_potential' or name.lower() == 'hartree':
        def wrapper(*args):
            return Hartree_potential(*args)
    elif name.lower() == 'soft_coulomb' or name.lower() == 'softc':
        def wrapper(*args): 
            return soft_coulomb(*args)
    elif name.lower() == 'hartree_MT' or name.lower() == 'hartree_mt':
        def wrapper(*args):
            return Hartree_potential_MT(*args)

    return wrapper


@jit
def soft_coulomb(x:Any,xp:Any,Ne: int):
    v_coul = 1/(jnp.sqrt( 1 + (x-xp)*(x-xp)))
    return v_coul*Ne**2

@jit
def Hartree_potential(x: Any, xp: Any, Ne: int, eps=1E-5):
    z = jnp.sum((x-xp)*(x-xp)+eps, axis=-1, keepdims=True)
    z = 1./(z**0.5)
    return 0.5*(Ne**2)*z


@jit
def Hartree_potential_MT(x: Any, xp: Any, Ne: int, alpha=0.5):
    # Martyna-Tuckerman J. Chem. Phys. 110, 2810–2821 (1999), Eq. B1, alpha_conv * L > 7
    # alpha_conv * L = 5, L = 10 A -> alpha_conv = 0.9448623 (Table 1 of J. Chem. Phys. 110, 2810–2821 (1999))

    r = jnp.sum((x-xp)*(x-xp), axis=-1, keepdims=True)
    r = jnp.sqrt(r)
    return 0.5*(Ne**2)*(lax.erf(alpha*r)/r + lax.erfc(alpha*r)/r)


# ------------------------------------------------------------------------------------------------------------
# POTENTIAL FUNCTIONALS
# ------------------------------------------------------------------------------------------------------------


def _nuclear(name: str = 'NP'):
    if name.lower() == 'nuclei_potential' or name.lower() == 'nuclei':
        def wrapper(*args):
            return Nuclei_potential(*args)
    elif name.lower() == 'madness' or name.lower() == 'mdns':
        def wrapper(*args):
            return Nuclei_potential_smooth(*args)
    elif name.lower() == 'harmonic' or name.lower() == 'ho':
        def wrapper(*args):
            return Nuclei_potential_smooth(*args)
    elif name.lower() == 'attraction' or name.lower() == 'attr':
        def wrapper(*args):
            return attraction(*args)
    elif name.lower() == 'hgh' or name.lower() == 'nuclei_potential_hgh':
        def wrapper(*args):
            return Nuclei_potential_HGH(*args)

    return wrapper

@jit
def attraction(x:Any, R:float, Z_alpha:int, Z_beta:int, Ne: int): 
    v_x = - Z_alpha/(jnp.sqrt(1 + (x + R/2)**2))  - Z_beta/(jnp.sqrt(1 + (x - R/2)**2))
    return v_x*Ne 

@jit
def harmonic_potential(params: Any, x: Any, Ne: int, k: Any = 1.) -> jax.Array:
    return 0.5*Ne*k*jnp.mean(x**2)


@partial(jax.jit,  static_argnums=(3,))
def Nuclei_potential(x: Any, Ne: int, mol_info: Any):
    eps = 1E-4  # 0.2162

    @jit
    def _potential(x: Any, molecule: Any):
        r = jnp.sqrt(
            jnp.sum((x-molecule['coords'])*(x-molecule['coords']), axis=1)) + eps
        z = molecule['z']
        return z/r

    r = vmap(_potential, in_axes=(None, 0), out_axes=-1)(x, mol_info)
    r = jnp.sum(r, axis=-1, keepdims=True)
    return -Ne*r  # lax.expand_dims(r, dimensions=(1,))


@jit
def Nuclei_potential_smooth(params: Any, x: Any,  Ne: int, mol_info: Any):
    # J. Chem. Phys. 121, 11587–11598 (2004)
    # Eq 25-27
    eps = 1E-2  # 0.2162
    c0 = 0.00435
    pi_sqrt = jnp.sqrt(jnp.pi)

    @jax.jit
    def _u(r: Any):
        r2 = r*r
        return lax.erf(r)/r + (1/(3*pi_sqrt))*(jnp.exp(-r2)+16*jnp.exp(-4*r2))

    @jax.jit
    def _potential(x: Any, molecule: Any):
        z = molecule['z']
        r = jnp.sqrt(
            jnp.sum((x-molecule['coords'])*(x-molecule['coords']), axis=1))
        c = (c0*eps/z**5)**1.3
        v = _u(r/c)/c
        return v

    r = vmap(_potential, in_axes=(None, 0), out_axes=-1)(x, mol_info)
    r = jnp.sum(r, axis=-1, keepdims=True)
    return -Ne*r  # lax.expand_dims(r, dimensions=(1,))


@jit
def Nuclei_potential_HGH(x: Any,  Ne: int, mol_info: Any):
    # INCORRECT only Hydrogen parameters
    #  Phys. Rev. B 58, 3641
    two_sqrt = jnp.sqrt(2)
    # H_pp_params = {'Zion': jnp.ones(1), 'rloc': 2*jnp.ones(1),
    #                'C1': -4.180237*jnp.ones(1), 'C2': 0.725075*jnp.ones(1), 'C3': jnp.zeros(1), 'C4': jnp.zeros(1), }
    H_pp_params = {'Zion': 1., 'rloc': 0.2,
                   'C1': -4.180237, 'C2': 0.725075, 'C3': 0., 'C4': 0., }

    @jax.jit
    def _u(r: Any, params_pp: Any):
        # eq 1
        zion = params_pp['Zion']
        rloc = params_pp['rloc']
        c1 = params_pp['C1']
        c2 = params_pp['C2']
        c3 = params_pp['C3']
        c4 = params_pp['C4']
        r_rloc = r/rloc
        r_rloc_2 = r_rloc*r_rloc
        v0 = (-zion/r)*lax.erf(r_rloc/two_sqrt)
        v1 = jnp.exp(-0.5*(r_rloc_2))
        v2 = c1 + c2*r_rloc_2 + c3*(r_rloc**4) + c4*(r_rloc**6)
        return v0 + v1*v2

    @jax.jit
    def _potential(x: Any, molecule: Any):
        z = molecule['z']
        # ai = molecule['atoms']
        r = jnp.sqrt(
            jnp.sum((x-molecule['coords'])*(x-molecule['coords']), axis=1))
        params_p_zi = H_pp_params
        v = _u(r, params_p_zi)
        return v

    r_all = vmap(_potential, in_axes=(None, 0), out_axes=-1)(x, mol_info)
    r = jnp.sum(r_all, axis=-1, keepdims=True)
    return Ne*r  # lax.expand_dims(r, dimensions=(1,))


@jit
def GaussianPotential1D(x: Any, Ne: int, params_pot: Any = None) -> jax.Array:
    if (params_pot is None):
        params_pot = {'alpha': jnp.array([[1.], [2.]]),  # Ha/electron
                      'beta': -1.*jnp.array([[-0.5], [1.]])}  # BHOR

    @jit
    def _f(x: Array, params_pot: Any):
        alpha, beta = params_pot['alpha'], params_pot['beta']
        return -alpha*jnp.exp(-(x-beta)*(x-beta))  # **2 OLD

    y = vmap(_f, in_axes=(None, 1))(x, params_pot)
    y = jnp.sum(y, axis=-1).transpose()
    return Ne*y


@partial(jax.jit,  static_argnums=(1,))
def cusp_condition(params: Any, fun: callable, mol_info: Any):

    @jax.jit
    def _cusp(molecule: Any):
        x = molecule['coords'][None]
        z = molecule['z']
        rho_val = fun(params, x)
        d_rho_val = score(params, x, fun)
        # check with the norm
        return jnp.sum(d_rho_val) - (-2*z*jnp.sum(rho_val))

    l = vmap(_cusp)(mol_info)
    return jnp.mean(jnp.abs(l))


if __name__ == '__main__':

    coords = jnp.array([[0., 0., -1.4008538753/2], [0., 0., 1.4008538753/2]])
    z = jnp.array([[1], [1]], dtype=int)
    # atoms = jnp.array(['H', 'H'], dtype=str)
    mol = {'coords': coords, 'z': z}  # 'atoms': atoms

    rng = jax.random.PRNGKey(0)
    _, key = jax.random.split(rng)
    x = jax.random.uniform(key, shape=(10, 3))
    _, key = jax.random.split(key)
    xp = jax.random.uniform(key, shape=(10, 3))
    def model_identity(params, x): return x
    y = Nuclei_potential(None, x, model_identity, mol)

    import matplotlib
    import matplotlib.pyplot as plt
    xt = jnp.linspace(-1.5, 1.5, 500)
    yz = jnp.zeros((xt.shape[0], 2))
    xyz = lax.concatenate((yz, xt[:, None]), 1)
    v_pot = y = Nuclei_potential(None, xyz, model_identity, mol)
    v_pot_s = Nuclei_potential_smooth(None, xyz, model_identity, mol)
    v_pot_hgh = Nuclei_potential_HGH(None, xyz, model_identity, mol)

    from dft_distrax import DFTDistribution
    atoms = ['H', 'H']
    geom = coords

    m = DFTDistribution(atoms, geom)

    x = jnp.ones((10, 3))
    # print(m.prob(m,geom))
    rho = m.prob(m, xyz)
    # def log_prob(value):
    #     return jnp.log(m.prob(m, value))

    # plt.plot(xt, rho, ls='--', color='k')
    plt.plot(xt, v_pot)
    plt.scatter(xt, v_pot_hgh, s=5)
    # plt.scatter(xt, v_pot_s, s=5)
    plt.ylim(bottom=-25., top=-1)
    plt.show()

    v_h = Hartree_potential_MT(None, x, xp, model_identity)
    print(v_h.shape)
