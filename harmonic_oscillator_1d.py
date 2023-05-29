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

from ofdft_normflows.functionals import _kinetic
from ofdft_normflows.functionals import harmonic_potential
from ofdft_normflows.flows import NormFlow

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]

jax.config.update("jax_enable_x64", True)


@jit
def uniform_pdf(x: Any, a: Any = -5., b: Any = 5.0) -> jnp.DeviceArray:
    dx = 1./(b-a)
    return dx*jnp.ones_like(x)


def main(kinetic_name: str = 'TF'):
    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)
    model = NormFlow(5, 1)
    params0 = model.init(key, jnp.zeros((1, 1)))
    z, ldj = model.apply(params0, jrnd
                         .uniform(key, (2, 1)))

    # select the kinetic functional
    t_functional = _kinetic(kinetic_name)

    # def log_p_z(u): return jnp.log(uniform_pdf(u))
    def log_p_u(u):  # beta distribution
        return jax.scipy.stats.norm.logpdf(u)

    @jit
    def log_density_forward(params, u):
        x, log_det = model.apply(params, u)
        # log_p_u = jnp.log(p_z_pdf(u))
        log_pu = log_p_u(u)
        log_p_x = log_pu - log_det
        return log_p_x, x

    @jit
    def rho(params, u):
        log_p_x, _ = log_density_forward(params, u)
        return jnp.exp(log_p_x)

    def fit(params: optax.Params,
            optimizer: optax.GradientTransformation,
            key: KeyArray,
            epochs: int = 10000,  # 10000
            batch_size: int = 512) -> optax.Params:

        opt_state = optimizer.init(params)

        def loss(params, u_samples):
            v = harmonic_potential(params, u_samples, model.apply)
            t = t_functional(params, u_samples, rho)
            return t + v, {"t": t, "v": v}

        @jax.jit
        def step(params, opt_state, batch):
            loss_value, grads = jax.value_and_grad(
                loss, has_aux=True)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        params_itr = {}
        loss0 = jnp.inf
        for i in range(1, epochs+1):
            _, key = jrnd.split(key)

            u_samples = jrnd.normal(key, shape=(batch_size, 1))

            params, opt_state, loss_value = step(params, opt_state, u_samples)
            loss_value, t_n_v = loss_value
            if jnp.isnan(loss_value):
                break
            if loss_value < loss0 and loss_value > 0.:
                loss0 = loss_value
                params_opt = params
            if i % 100 == 0 or i == 1 or i <= 10:
                params_itr.update(
                    {i: {'params': params_opt, 'T': t_n_v['t'], 'V': t_n_v['v'], 'energy': loss_value}})
            if i % 50 == 0 or i == 1 or i < 10:
                t = t_n_v['t']
                v = t_n_v['v']
                _u = jnp.linspace(-10., 10., 10000)
                _rho = rho(params, jnp.expand_dims(_u, axis=1))
                _x, _ = model.apply(params, jnp.expand_dims(_u, axis=1))
                # _rho = jnp.exp(log_p_z(jnp.expand_dims(_u, axis=1)))
                _int = jnp.trapz(_rho.ravel(), _x.ravel())
                print(
                    f'step {i}, loss: {loss_value}, T: {t}, V: {v}, int: {_int}')

        return params_opt, params_itr

    optimizer = optax.adam(learning_rate=1e-3)
    _, key = jrnd.split(key)
    params, params_itr = fit(params0, optimizer, key)
    # assert 0

    import matplotlib
    import matplotlib.pyplot as plt

    u = jnp.linspace(-5., 5., 1000)
    u = jnp.expand_dims(u, axis=1)
    plt.figure(figsize=(10, 10), dpi=160)
    for i, k in enumerate(params_itr):
        if k % 100 == 0 or k == 1 or k == 10:
            paramsi = params_itr[k]['params']
            x, _ = model.apply(paramsi, u)
            plt.plot(x, rho(paramsi, u), label=r'$\rho(r, N=%s)$' % (k))
    plt.ylabel(r'$\rho(r)$', fontsize=17)
    plt.xlabel(r'$r$', fontsize=17)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Figures/HO1D_{kinetic_name}.png')

    plt.figure(figsize=(10, 10), dpi=160)
    plt.clf()
    e_, t_, v_, epoch_ = [], [], [], []
    for i, k in enumerate(params_itr):
        e_.append(params_itr[k]['energy'])
        t_.append(params_itr[k]['T'])
        v_.append(params_itr[k]['V'])
        epoch_.append(k)

    plt.plot(epoch_, e_, ls='--', color='k', label=r'$E[\rho(x)]$')
    plt.plot(epoch_, t_, ls='solid', color='tab:blue',
             label=r'$T_{%s}[\rho(x)]$' % (kinetic_name))
    plt.plot(epoch_, v_, ls='-.', color='tab:orange',
             label=r'$V(x)=\frac{1}{2}kx^{2}$')
    plt.xlabel('epochs', fontsize=17)
    plt.ylabel(
        r'$\mathbb{E}_{\rho(N=512)} \left[ \hat{O} \right]$', fontsize=17)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Figures/HO1D_E_vs_epochs_{kinetic_name}.png')


if __name__ == "__main__":
    main('K')
