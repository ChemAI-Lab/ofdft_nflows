from typing import Any, Union, Callable
import jax
import jax.numpy as jnp
from jax import jit
import jax.random as jrnd

from ofdft_normflows.functionals import laplacian

Array = Any


@jit
def uniform_pdf(x: Any, a: Any = -10., b: Any = 10.0) -> jnp.DeviceArray:
    dx = 1./(b-a)
    return dx*jnp.ones_like(x)


@jit
def potential_energy(x: Any, k: Any = 10.) -> jnp.DeviceArray:
    return jnp.sum(k*x**2)


def kinetic_energy(x: Any, params: Any, rho: Callable) -> jnp.DeviceArray:
    def sqrt_rho(x, params): return jnp.sqrt(rho(params, x))  # flax format
    lap_val = laplacian(x, params, sqrt_rho)
    rho_val = rho(params, x)
    rho_val = 1./jnp.sqrt(rho_val)
    return jnp.multiply(rho_val, lap_val)


def main():
    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)
    u = jrnd.uniform(key, (10, 1), minval=-10., maxval=10.)
    print(u)
    print(uniform_pdf(u, -10., 10.))
    def rho(x, params): return jax.scipy.stats.norm.pdf(x)
    lap = laplacian(u, None, rho)
    print(lap)

    # x = jnp.zeros((10, 1))
    # print(uniform_pdf(x, -1., 1.))


if __name__ == "__main__":
    main()
