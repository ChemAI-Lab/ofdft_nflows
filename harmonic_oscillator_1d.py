from typing import Any, Callable, Sequence, Optional, Union
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, jit
import jax.random as jrnd
from jax._src import prng

import flax
from flax import linen as nn
import optax

from ofdft_normflows.functionals import laplacian
from ofdft_normflows.flows import NormFlow

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]


@jit
def uniform_pdf(x: Any, a: Any = -5., b: Any = 5.0) -> jnp.DeviceArray:
    dx = 1./(b-a)
    return dx*jnp.ones_like(x)


# @jit
def potential_energy(params: Any, u: Any, T: Callable, k: Any = 10.) -> jnp.DeviceArray:
    x, _ = T(params, u)
    return 0.5*k*jnp.mean(x**2)


def kinetic_energy(params: Any, u: Any, rho: Callable) -> jnp.DeviceArray:
    def sqrt_rho(u, params): return jnp.sqrt(rho(params, u))  # flax format
    lap_val = laplacian(u, params, sqrt_rho)
    rho_val = rho(params, u)
    rho_val = 1./jnp.sqrt(rho_val)
    return -0.5*jnp.mean(jnp.multiply(rho_val, lap_val))


def main():
    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)
    model = NormFlow(3, 1)
    params0 = model.init(key, jnp.zeros((1, 1)))
    z, ldj = model.apply(params0, jrnd.uniform(key, (2, 1)))

    def log_density_forward(params, u):
        x, log_det = model.apply(params, u)
        log_p_u = jnp.log(uniform_pdf(u))
        log_p_x = log_p_u - log_det
        return log_p_x, x

    def rho(params, u):
        log_p_x, _ = log_density_forward(params, u)
        return jnp.exp(log_p_x)

    def fit(params: optax.Params,
            optimizer: optax.GradientTransformation,
            key: KeyArray,
            epochs: int = 50000,
            batch_size: int = 512) -> optax.Params:

        opt_state = optimizer.init(params)

        def loss(params, u_samples):
            # log_p_x, x = log_density_forward(params, u_samples)
            # potential energy
            v = potential_energy(params, u_samples, model.apply)
            t = kinetic_energy(params, u_samples, rho)
            return t + v

        @jax.jit
        def step(params, opt_state, batch):
            loss_value, grads = jax.value_and_grad(loss)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        loss0 = jnp.inf
        for i in range(epochs):
            _, key = jrnd.split(key)

            u_samples = jrnd.uniform(key, shape=(
                batch_size, 1), minval=-5., maxval=5.)

            params, opt_state, loss_value = step(params, opt_state, u_samples)
            if loss_value < loss0 and loss_value > 0.:
                loss0 = loss_value
                params_opt = params
            if i % 250 == 0:
                print(f'step {i}, loss: {loss_value}')

        return params_opt

    optimizer = optax.adam(learning_rate=1e-3)
    _, key = jrnd.split(key)
    params = fit(params0, optimizer, key)

    import matplotlib
    import matplotlib.pyplot as plt

    u = jnp.linspace(-5., 5., 1000)
    u = jnp.expand_dims(u, axis=1)
    rho_x = rho(params, u)
    plt.plot(u, rho(params0, u), label='Rho init')
    plt.plot(u, rho_x, label='Rho opt')
    plt.legend()
    plt.savefig('Figures/HO1D.png')


if __name__ == "__main__":
    main()
