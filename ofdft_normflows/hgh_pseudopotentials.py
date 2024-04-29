import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import sph_harm, gamma

jax.config.update('jax_disable_jit', True)

H_pp_params = {'Zion': 1., 'rloc': 0.2,
               'C1': -4.180237, 'C2': 0.725075, 'C3': 0., 'C4': 0., }
O_pp_params = {'Zion': 6., 'rloc': 0.247621,
               'C1': -16.580318, 'C2': 2.395701, 'C3': 0., 'C4': 0., }

#
# UNDER CONSTRUCTION
#

@jax.jit
def cartesian_to_spherical(cartesian_coords):
    x, y, z = cartesian_coords[0], cartesian_coords[1], cartesian_coords[2]
    r = jnp.sqrt(x**2 + y**2 + z**2)
    # Clip the value to ensure it is within [-1, 1]
    phi = jnp.arccos(jnp.clip(z / r, -1.0, 1.0))
    theta = jnp.arctan2(y, x)

    # Convert theta to the range [0, 2*pi]
    theta = jnp.where(theta < 0, theta + 2*jnp.pi, theta)

    # Convert phi to the range [0, pi]
    phi = jnp.where(phi < 0, phi + jnp.pi, phi)

    return jnp.stack((r, phi, theta), axis=-1)


@jax.jit
def _sph_harm(r_theta_phi, l, m):
    _, theta, phi = r_theta_phi[0], r_theta_phi[2], r_theta_phi[1]
    print(l, m)
    return sph_harm(m, l, theta, phi)


sph_harm_vmap_m = vmap(_sph_harm, in_axes=(None, None, 0))


def p_l_i(r, rl, l, i):
    # Phys. Rev. B 58, 3641 1998
    # Eq. 3

    sqrt_2 = jnp.sqrt(2)
    c0 = l + (4*i - 1)/2
    c1 = l+2*i - 1
    val0 = jnp.exp(-(r*r)/(2*rl*rl))
    val0 = sqrt_2*(r*c1)*val0
    _sqrt_gamma = jnp.sqrt(gamma(c0))
    val1 = (rl**c0)*_sqrt_gamma
    return val0/val1


def v_l(r, rp, l_max, hl_params):

    r_sph = cartesian_to_spherical(r)
    rp_sph = cartesian_to_spherical(rp)

    m_ = jnp.arange(-l_max, l_max+1, dtype=int)

    return sph_harm(-2, jnp.array([2]), r_sph[2], r_sph[1])
    # y_lm_r = sph_harm_vmap_m(r_sph, l_max, m_)
    # y_lm_rp = sph_harm_vmap_m(rp_sph, l_max, m_)
    # return y_lm_r, y_lm_rp



if __name__ == '__main__':

    print(jax.__version__)
    import jax.random as jrnd
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)
    r_rp = jrnd.normal(key, (2, 3))
    r = r_rp[0]
    rp = r_rp[1]
    print(rp, cartesian_to_spherical(rp))
    l_max = 1
    hl_params = 0.

    # print(v_l(r, rp, l_max, hl_params))

    import matplotlib.pyplot as plt
    import jax.scipy.special as jp

    l = jnp.array([1])
    m = jnp.array([-1])
    print(jp.sph_harm(m, l, 0., jnp.pi/2))
    print(0.5*jnp.sqrt(3/(2*jnp.pi)))

    r_shp = cartesian_to_spherical(r)
    print(jp.sph_harm(m, l, r_shp[1][None], r_shp[2][None]))

    # notes theta and phi must be arrays. also m and l
