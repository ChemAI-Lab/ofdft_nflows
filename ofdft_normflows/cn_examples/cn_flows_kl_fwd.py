import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from functools import partial
import jax
from typing import Any, Callable, Sequence, Optional, NewType
from jax import lax, random, vmap, scipy, numpy as jnp
from jax.experimental.ode import odeint
import jax.random as jrnd

import flax
from flax import linen as nn
import optax

from cn_flows import neural_ode, Gen_CNF


def w_1(z):
    return jnp.sin((2 * jnp.pi * z[:, 0]) / 4)


def w_1(z):
    return jnp.sin((2 * jnp.pi * z[:, 0]) / 4)


def w_2(z):
    return 3 * jnp.exp(-.5 * ((z[:, 0] - 1) / .6) ** 2)


def sigma(x):
    return 1 / (1 + jnp.exp(- x))


def w_3(z):
    return 3 * sigma((z[:, 0] - 1) / .3)


def pot_1(z):
    z_1, z_2 = z[:, 0], z[:, 1]
    norm = jnp.sqrt(z_1 ** 2 + z_2 ** 2)
    outer_term_1 = .5 * ((norm - 2) / .4) ** 2
    inner_term_1 = jnp.exp((-.5 * ((z_1 - 2) / .6) ** 2))
    inner_term_2 = jnp.exp((-.5 * ((z_1 + 2) / .6) ** 2))
    outer_term_2 = jnp.log(inner_term_1 + inner_term_2 + 1e-7)
    u = outer_term_1 - outer_term_2
    return - u


def log_target_density(z): return pot_1(z)  # log_P


def batch_generator(batch_size: int = 512, d_dim: int = 2):
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    mean = jnp.zeros(d_dim)
    cov = jnp.eye(d_dim)
    while True:
        _, key = jrnd.split(key)
        u = jrnd.multivariate_normal(
            key, mean=mean, cov=cov, shape=(batch_size,))
        log_pdf = jax.scipy.stats.multivariate_normal.logpdf(
            u, mean=mean, cov=cov)
        # log_pdf = jnp.zeros_like(log_pdf)  # don't know why :/

        u_and_log_pu = lax.concatenate((u, lax.expand_dims(log_pdf, (1,))), 1)
        yield u_and_log_pu


def main(batch_size, epochs):

    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model = Gen_CNF()
    test_inputs = lax.concatenate((jnp.ones((1, 2)), jnp.ones((1, 1))), 1)
    params = model.init(key, jnp.array(0.), test_inputs)

    from jax.tree_util import tree_map
    # params = jax.tree_util.tree_map(lambda x: 0.1 * jnp.ones_like(x), params)
    # print(params)

    def NODE(params, batch): return neural_ode(
        params, batch, model, 0., 10., 2)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    def log_p_z0(samples):
        d_dim = samples.shape[1]
        mean = jnp.zeros(d_dim)
        cov = 0.1*jnp.eye(d_dim)
        logp = jax.scipy.stats.multivariate_normal.logpdf(
            samples, mean=mean, cov=cov)
        return logp.reshape(samples.shape[0], 1)

    def loss(params, samples):
        z0 = samples[:, :2]
        zt, logp_zt = NODE(params, samples)
        logp_x = log_p_z0(z0) + logp_zt
        logp_x_target = log_target_density(zt)

        return jnp.mean(logp_zt - logp_x_target)

    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    gen_batch = batch_generator(batch_size)
    for i in range(epochs+1):

        batch = next(gen_batch)
        params, opt_state, loss_value = step(params, opt_state, batch)
        if i % 10 == 0:
            print(f'step {i}, loss: {loss_value}')

    png = jrnd.PRNGKey(1)
    _, key = jrnd.split(png)
    z0 = jrnd.multivariate_normal(key, mean=jnp.zeros(
        (2,)), cov=jnp.eye(2), shape=(3000, ))
    logp_z0 = jax.scipy.stats.multivariate_normal.logpdf(
        z0, mean=jnp.zeros(2), cov=jnp.eye(2))
    z0_and_log_pz0 = lax.concatenate((z0, logp_z0[:, jnp.newaxis]), 1)
    print(z0_and_log_pz0.shape)
    x, logp_zt = NODE(params, z0_and_log_pz0)

    plt.title('Sampels from the NormFlow')
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.savefig('Figures/CNF_KLF_twomoons.png')


if __name__ == '__main__':
    batch_size = 512
    epochs = 150

    main(batch_size, epochs)
