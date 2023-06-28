from functools import partial
from typing import Any, Callable, Sequence, Optional, NewType, Union

import jax
from jax import lax, random, vmap, scipy, numpy as jnp
from jax.experimental.ode import odeint
import jax.random as jrnd
from jax._src import prng

import flax
from flax import linen as nn
import optax

import numpy as np
import pyscf
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.dft import r_numint
from pyscf.data.nist import BOHR

from ofdft_normflows.functionals import _kinetic
from ofdft_normflows.functionals import harmonic_potential
from ofdft_normflows.cn_flows import neural_ode, Gen_CNFRicky
from ofdft_normflows.cn_flows import Gen_CNFSimpleMLP as CNF

import matplotlib.pyplot as plt

from sklearn.datasets import make_s_curve


Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_enable_x32", True)


# file from https://huggingface.co/flax-community/NeuralODE_SDE/raw/main/train_cnf.py

@jax.jit
def w_1(z):
    return jnp.sin((2 * jnp.pi * z[:, 0]) / 4)


@jax.jit
def w_1(z):
    return jnp.sin((2 * jnp.pi * z[:, 0]) / 4)


@jax.jit
def w_2(z):
    return 3 * jnp.exp(-.5 * ((z[:, 0] - 1) / .6) ** 2)


@jax.jit
def sigma(x):
    return 1 / (1 + jnp.exp(- x))


@jax.jit
def w_3(z):
    return 3 * sigma((z[:, 0] - 1) / .3)


@jax.jit
def pot_1(z):
    z_1, z_2 = z[:, 0], z[:, 1]
    norm = jnp.sqrt(z_1 ** 2 + z_2 ** 2)
    outer_term_1 = .5 * ((norm - 2) / .4) ** 2
    inner_term_1 = jnp.exp((-.5 * ((z_1 - 2) / .6) ** 2))
    inner_term_2 = jnp.exp((-.5 * ((z_1 + 2) / .6) ** 2))
    outer_term_2 = jnp.log(inner_term_1 + inner_term_2 + 1e-7)
    u = outer_term_1 - outer_term_2
    return - u


@jax.jit
def log_target_density(z): return pot_1(z)  # log_P
# ---------------


def batch_generator(batch_size: int = 512, d_dim: int = 2):
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    mean = jnp.zeros(d_dim)
    cov = 1.*jnp.eye(d_dim)
    while True:
        _, key = jrnd.split(key)
        u = jrnd.multivariate_normal(
            key, mean=mean, cov=cov, shape=(batch_size,))
        log_pdf = jax.scipy.stats.multivariate_normal.logpdf(
            u, mean=mean, cov=cov)
        log_pdf = jnp.zeros_like(log_pdf)  # don't know why :/

        u_and_log_pu = lax.concatenate((u, lax.expand_dims(log_pdf, (1,))), 1)
        yield u_and_log_pu
# ---------------


def main(batch_size, epochs):

    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model_rev = CNF(2, (96, 96, ), bool_neg=True)
    model_fwd = CNF(2, (96, 96, ), bool_neg=False)
    # model_rev = Gen_CNFRicky(2, 128, 254, bool_neg=True)
    # model_fwd = Gen_CNFRicky(2, 128, 254, bool_neg=False)
    test_inputs = lax.concatenate((jnp.ones((1, 2)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)

    @jax.jit
    def NODE_rev(params, batch): return neural_ode(
        params, batch, model_rev, -1., 0., 2)

    @jax.jit
    def NODE_fwd(params, batch): return neural_ode(
        params, batch, model_fwd, 0., 1., 2)

    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(params)

    @jax.jit
    def logp_z0(samples):
        d_dim = samples.shape[1]
        mean = jnp.zeros(d_dim)
        cov = 1.*jnp.eye(d_dim)
        logp = jax.scipy.stats.multivariate_normal.logpdf(
            samples, mean=mean, cov=cov)
        return logp.reshape(samples.shape[0], 1)

    @jax.jit
    def loss(params, samples):
        z0 = samples[:, :2]
        zt, logp_zt = NODE_fwd(params, samples)
        logp_x = logp_z0(z0) + logp_zt  # check the sign
        logp_true = log_target_density(zt)
        # return -1.*jnp.mean(logp_x)
        return jnp.mean(logp_x - logp_true)

    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    gen_batch = batch_generator(batch_size)
    loss0 = jnp.inf
    for i in range(epochs+1):

        batch = next(gen_batch)
        params, opt_state, loss_value = step(params, opt_state, batch)
        if i % 50 == 0:
            print(f'step {i}, loss: {loss_value}')
        if loss_value < loss0:
            print(f'step {i}, loss: {loss_value}')
            params_opt, loss0 = params, loss_value

    png = jrnd.PRNGKey(1)
    _, key = jrnd.split(png)

    z0 = jrnd.multivariate_normal(key, mean=jnp.zeros(
        (2,)), cov=1.*jnp.eye(2), shape=(2000,))
    log_pz0 = jax.scipy.stats.multivariate_normal.logpdf(
        z0, mean=jnp.zeros((2,)), cov=0.1*jnp.eye(2))
    u_and_log_pu = lax.concatenate((z0, log_pz0[:, None]), 1)

    zt_samples, logp_zt_samples = NODE_fwd(params_opt, u_and_log_pu)

    # zt_and_logp_zt = lax.concatenate((zt, logp_zt), 1)
    # z0_test, logp_z0_test = NODE_rev(params_opt, zt_and_logp_zt)

    # print(jnp.linalg.norm(z0 - z0_test))
    # print(log_pz0[:10], logp_z0_test[:10])

    # plt.title('Samples from the NormFlow')
    # plt.scatter(zt[:, 0], zt[:, 1])
    # plt.savefig('Figures/CNF_twomoons_KLrev.png')

    u0 = jnp.linspace(-4.25, 4.25, 100)
    u1 = jnp.linspace(-4.25, 4.25, 100)
    u0_, u1_ = jnp.meshgrid(u0, u1)
    zt = lax.concatenate(
        (jnp.expand_dims(u0_.ravel(), 1), jnp.expand_dims(u1_.ravel(), 1)), 1)
    zt_and_log_pzt = lax.concatenate((zt, jnp.zeros_like(zt[:, :1])), 1)

    z0, logp_diff_z0 = NODE_rev(params_opt, zt_and_log_pzt)
    logp_x = logp_z0(z0) - logp_diff_z0  # check the sign

    plt.figure(0)
    plt.clf()
    # plt.contour(u0_, u1_, logp_x.reshape(u0_.shape))
    plt.contourf(u0_, u1_, log_target_density(zt).reshape(u0_.shape))
    plt.contour(u0_, u1_, logp_x.reshape(u0_.shape), levels=25, cmap='binary',
                linestyles='dashed', linewidths=0.75)
    plt.scatter(zt_samples[:, 0], zt_samples[:, 1], s=10)
    plt.savefig('Figures/CNF_logP_twomoons_KLrev.png')


if __name__ == '__main__':
    batch_size = 254
    epochs = 1500

    main(batch_size, epochs)
