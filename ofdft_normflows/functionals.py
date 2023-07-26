from typing import Any, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, hessian, jacrev, lax

Array = jax.Array
BHOR = 1.  # 1.8897259886  # 1AA to BHOR


@partial(jit,  static_argnums=(2,))
def laplacian(params: Any, X: Array, fun: callable) -> jax.Array:
    """_summary_

    Args:
        X (Array): _description_
        params (Any): _description_
        fun (callable): _description_

    Returns:
        jax.Array: _description_
    """
    @partial(jit,  static_argnums=(2,))
    def _laplacian(params: Any, X: Array, fun: callable):
        hes_ = hessian(fun, argnums=1)(
            params, X[jnp.newaxis], )  # R[jnp.newaxis]
        hes_ = jnp.squeeze(hes_, axis=(0, 2, 4))
        hes_ = jnp.einsum('...ii', hes_)
        return hes_

    v_laplacian = vmap(_laplacian, in_axes=(None, 0,  None))
    return v_laplacian(params, X, fun)


@partial(jit,  static_argnums=(2,))
def score(params: Any, X: Array, fun: callable) -> jax.Array:

    @jit
    def _score(params: Any, xi: Array):
        score_ = jax.jacrev(fun, argnums=1)(params, xi[jnp.newaxis])
        return jnp.reshape(score_, xi.shape[0])

    v_score = vmap(_score, in_axes=(None, 0))
    return v_score(params, X)


# ------------------------------------------------------------------------------------------------------------
# POTENTIAL FUNCTIONALS
# ------------------------------------------------------------------------------------------------------------

@partial(jit,  static_argnums=(2,))
def harmonic_potential(params: Any, u: Any, T: Callable, k: Any = 1.) -> jax.Array:
    x, _ = T(params, u)
    return 0.5*k*jnp.mean(x**2)


@partial(jit,  static_argnums=(2,))
def dirac_exchange(params: Any, u: Any, rho: Callable) -> jax.Array:
    rho_val = rho(params, u)

    l = -(3/4)*(3/jnp.pi)**(1/3)
    return l*jnp.mean(rho_val**(1/3))

# ------------------------------------------------------------------------------------------------------------


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
    elif name.lower() == 'k' or name.lower() == 'kinetic':
        def wrapper(*args):
            return kinetic(*args)
    return wrapper


@ partial(jit,  static_argnums=(2,))
def kinetic(params: Any, u: Any, rho: Callable) -> jax.Array:
    def sqrt_rho(params, u): return (rho(params, u)+1E-4)**0.5  # flax format
    lap_val = laplacian(params, u, sqrt_rho)
    # return -0.5*jnp.mean(lap_val)
    rho_val = rho(params, u)
    rho_val = 1./(rho_val+1E-4)**0.5  # for numerical stability
    return -0.5*jnp.multiply(rho_val, lap_val)


@partial(jit,  static_argnums=(2,))
def weizsacker(params: Any, u: Array, fun: callable, l: Any = .2) -> jax.Array:
    """
    l = 0.2 (W Stich, EKU Gross., Physik A Atoms and Nuclei, 309(1):511, 1982.)
    T_{\text{Weizsacker}}[\rho] &=& \frac{\lambda}{8} \int \frac{(\nabla \rho)^2}{\rho} dr = 
                            &=&    \frac{\lambda}{8} \int  \rho \left(\frac{(\nabla \rho)}{\rho}\right)^2 dr\\
    T_{\text{Weizsacker}}[\rho] = \mathbb{E}_{\rho} \left[ \left(\frac{(\nabla \rho)}{\rho}\right)^2 \right]

    Args:
        params (Any): _description_
        u (Array): _description_
        fun (callable): _description_
        l (Any, optional): _description_. Defaults to 1..

    Returns:
        jax.Array: _description_
    """
    score_ = score(params, u, fun)
    rho_ = fun(params, u)
    val = (score_/rho_)**2
    return (l/8.)*val


@partial(jit,  static_argnums=(2,))
def thomas_fermi(params: Any, u: Array, fun: callable) -> jax.Array:
    """_summary_

    T_{\text{TF}}[\rho] &=& \frac{3}{10}(3\pi^2)^{2/3} \int ( \rho)^{5/3} dr \\
    T_{\text{TF}}[\rho] = \mathbb{E}_{\rho} \left[ ( \rho)^{2/3} \right]

    Args:
        params (Any): _description_
        u (Array): _description_
        fun (callable): _description_

    Returns:
        jax.Array: _description_
    """
    rho_ = fun(params, u)
    val = (rho_)**(2/3)
    l = (3./10.)*(3.*jnp.pi**2)**(2/3)
    return l*val


@partial(jit,  static_argnums=(2,))
def thomas_fermi_1D(params: Any, u: Array, fun: callable) -> jax.Array:
    """_summary_

    T_{\text{TF}}[\rho] &=& \frac{\pi^2}{12}\int ( \rho)^{} dr \\
    T_{\text{TF}}[\rho] = \frac{\pi^2}{12}\mathbb{E}_{\rho} \left[ ( \rho)^{2} \right]

    Args:
        params (Any): _description_
        u (Array): _description_
        fun (callable): _description_

    Returns:
        jax.Array: _description_
    """
    rho_ = fun(params, u)
    val = rho_*rho_
    l = (jnp.pi*jnp.pi)/12.
    return l*val
# ------------------------------------------------------------------------------------------------------------


@partial(jit,  static_argnums=(2,))
def Dirac_exchange(params: Any, u: Array, fun: callable) -> jax.Array:
    """_summary_

    ^{Dirac}E_{\text{x}}[\rho] = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}\int  \rho^{4/3} dr \\
    ^{Dirac}E_{\text{x}}[\rho] = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}\mathbb{E}_{\rho} \left[ \rho^{1/3} \right]

    Args:
        params (Any): _description_
        u (Array): _description_
        fun (callable): _description_

    Returns:
        jax.Array: _description_
    """
    rho_ = fun(params, u)
    l = -(3/4)*(3/jnp.pi)**1/3
    return l*rho_**(1/3)

# ------------------------------------------------------------------------------------------------------------


@partial(jit,  static_argnums=(2, 3))
def GaussianPotential1D(params: Any, u: Any, T: Callable,  params_pot: Any = None) -> jax.Array:
    if (params_pot is None):
        params_pot = {'alpha': jnp.array([[1.], [2.]]),  # Ha/electron
                      'beta': -1.*jnp.array([[-0.5], [1.]])}  # BHOR

    # x = T(u)
    x = T(params, u)

    @jit
    def _f(x: Array, params_pot: Any):
        alpha, beta = params_pot['alpha'], params_pot['beta']
        return -alpha*jnp.exp(-(x-beta)*(x-beta))  # **2 OLD

    y = vmap(_f, in_axes=(None, 1))(x, params_pot)
    y = jnp.sum(y, axis=-1).transpose()
    return y


@partial(jit,  static_argnums=(2, 3))
def GaussianPotential1D_pot(params: Any, u: Any, T: Callable,  params_pot: Any = None) -> jax.Array:
    if (params_pot is None):
        params_pot = {'alpha': jnp.array([[1.], [2.]]),  # Ha/electron
                      'beta': -1.*jnp.array([[-0.5], [1.]])}  # BHOR

    # x = T(u)
    x = T(params, u)

    @jit
    def _f(x: Array, params_pot: Any):
        alpha, beta = params_pot['alpha'], params_pot['beta']
        return -alpha*jnp.exp(-(x-beta)*(x-beta))  # **2 OLD

    y = vmap(_f, in_axes=(None, 1))(x, params_pot)
    y = jnp.sum(y, axis=-1).transpose()
    return y

# ------------------------------------------------------------------------------------------------------------


def _hartree(name: str = 'full'):
    if name.lower() == 'mt':
        def wrapper(*args):
            return Hartree_potential_MT(*args)
    else:  # full
        def wrapper(*args):
            return Hartree_potential(*args)

    return wrapper


@partial(jax.jit,  static_argnums=(3,))
def Coulomb_potential(params: Any, u: Any, up: Any, T: Callable, eps=1E-3):
    x = T(params, u)
    xp = T(params, up)
    z = 1./jnp.linalg.norm(x-xp, axis=1)
    return 0.5*z


@partial(jax.jit,  static_argnums=(3, 4,))
def Hartree_potential(params: Any, u: Any, up: Any, T: Callable, eps=1E-3):
    x = T(params, u)
    xp = T(params, up)
    z = jnp.sum((x-xp)*(x-xp), axis=-1, keepdims=True)
    z = 1./(z**0.5+eps)
    return 0.5*z


@partial(jax.jit,  static_argnums=(3, 4,))
def Hartree_potential_MT(params: Any, u: Any, up: Any, T: Callable, alpha=0.5):
    # Martyna-Tuckerman J. Chem. Phys. 110, 2810–2821 (1999)
    # alpha_conv * L = 5, L = 10 A -> alpha_conv = 0.9448623 (Table 1 of J. Chem. Phys. 110, 2810–2821 (1999))
    x = T(params, u)
    xp = T(params, up)
    r = jnp.sum((x-xp)*(x-xp), axis=-1, keepdims=True)
    r = jnp.sqrt(r)
    return 0.5*(lax.erf(alpha*r)/r + lax.erfc(alpha*r)/r)

# ------------------------------------------------------------------------------------------------------------


def _nuclear(name: str = 'HGH'):
    if name.lower() == 'hgh':
        def wrapper(*args):
            return Nuclei_potential_HGH(*args)
    elif name.lower() == 'madness':
        def wrapper(*args):
            return Nuclei_potential_smooth(*args)
    else:
        def wrapper(*args):
            return Nuclei_potential(*args)

    return wrapper


@partial(jax.jit,  static_argnums=(2,))
def Nuclei_potential(params: Any, u: Any, T: Callable, mol_info: Any):
    eps = 1E-4  # 0.2162

    @jit
    def _potential(x: Any, molecule: Any):
        r = jnp.sqrt(
            jnp.sum((x-molecule['coords'])*(x-molecule['coords']), axis=1)) + eps
        z = molecule['z']
        return z/r

    x = T(params, u)
    r = vmap(_potential, in_axes=(None, 0), out_axes=-1)(x, mol_info)
    r = jnp.sum(r, axis=-1, keepdims=True)
    return -r  # lax.expand_dims(r, dimensions=(1,))


@partial(jax.jit,  static_argnums=(2,))
def Nuclei_potential_smooth(params: Any, u: Any, T: Callable, mol_info: Any):
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

    x = T(params, u)
    r = vmap(_potential, in_axes=(None, 0), out_axes=-1)(x, mol_info)
    r = jnp.sum(r, axis=-1, keepdims=True)
    return -r  # lax.expand_dims(r, dimensions=(1,))


@partial(jax.jit,  static_argnums=(2,))
def Nuclei_potential_HGH(params: Any, u: Any, T: Callable, mol_info: Any):
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

    x = T(params, u)
    r_all = vmap(_potential, in_axes=(None, 0), out_axes=-1)(x, mol_info)
    r = jnp.sum(r_all, axis=-1, keepdims=True)
    return r  # lax.expand_dims(r, dimensions=(1,))


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
    # print(y.shape)
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
    # plt.plot(xt, v_pot)
    plt.scatter(xt, v_pot_hgh, s=5)
    # plt.scatter(xt, v_pot_s, s=5)
    # plt.ylim(bottom=-55., top=-1)
    plt.show()

    v_h = Hartree_potential_MT(None, x, xp, model_identity)
    print(v_h.shape)
