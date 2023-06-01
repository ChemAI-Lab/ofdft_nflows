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
from flax.training import train_state
from flax import traverse_util
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import serialization
import optax
from tqdm import tqdm


# file from https://huggingface.co/flax-community/NeuralODE_SDE/raw/main/train_cnf.py
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
# ---------------


# @nn.jit
class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the Pytorch implementation at:
    https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py
    """
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64

    @nn.compact
    def __call__(self, t):
        # predict params
        blocksize = self.width * self.in_out_dim
        params = lax.expand_dims(t, (0, 1))
        params = nn.Dense(self.hidden_dim)(params)
        params = nn.tanh(params)
        params = nn.Dense(self.hidden_dim)(params)
        params = nn.tanh(params)
        params = nn.Dense(3 * blocksize + self.width)(params)

        # restructure
        params = lax.reshape(params, (3 * blocksize + self.width,))
        W = lax.reshape(params[:blocksize], (self.width, self.in_out_dim, 1))

        U = lax.reshape(params[blocksize:2 * blocksize],
                        (self.width, 1, self.in_out_dim))

        G = lax.reshape(params[2 * blocksize:3 * blocksize],
                        (self.width, 1, self.in_out_dim))
        U = U * nn.sigmoid(G)

        B = lax.expand_dims(params[3 * blocksize:], (1, 2))
        return W, B, U


class CNF(nn.Module):
    """Adapted from the Pytorch implementation at:
    https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py
    """
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64

    @nn.compact
    def __call__(self, t, states):
        z, logp_z = states[:, :self.in_out_dim], states[:, self.in_out_dim:]
        W, B, U = HyperNetwork(self.in_out_dim, self.hidden_dim, self.width)(t)

        def dzdt(z):
            h = nn.tanh(vmap(jnp.matmul, (None, 0))(z, W) + B)
            return jnp.matmul(h, U).mean(0)

        dz_dt = dzdt(z)
        def sum_dzdt(z): return dzdt(z).sum(0)
        df_dz = jax.jacrev(sum_dzdt)(z)
        dlogp_z_dt = -1.0 * jnp.trace(df_dz, 0, 0, 2)

        return lax.concatenate((dz_dt, lax.expand_dims(dlogp_z_dt, (1,))), 1)


class Gen_CNF(nn.Module):
    """Negative CNF for jax's odeint."""
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64
    bool_neg: bool = True

    def setup(self) -> None:
        self.cnf = CNF(self.in_out_dim, self.hidden_dim,
                       self.width)
        if self.bool_neg:
            self.y0 = -1.
        else:
            self.y0 = 1.

    @nn.compact
    def __call__(self, t, states):
        outputs = self.cnf(self.y0 * t, states)
        return self.y0 * outputs


@partial(jax.jit,  static_argnums=(2, 3, 4, 5,))
def neural_ode(params: Any, batch: Any, f: Callable, t0: float, t1: float, d_dim: int):
    start_and_end_time = -1.*jnp.array([t1, t0])  # jnp.array([t0, t1])

    def _evol_fun(states, t):
        return f.apply(params, t, states)

    outputs = odeint(
        _evol_fun,
        batch,
        start_and_end_time,
        atol=1e-5,
        rtol=1e-5
    )
    z_t, logp_diff_t = outputs[:, :,
                               :d_dim], outputs[:, :, d_dim:]
    # z_t0, logp_diff_t0 = z_t[0], logp_diff_t[0]
    z_t1, logp_diff_t1 = z_t[-1], logp_diff_t[-1]
    # return lax.concatenate((z_t0, z_t1), 2), lax.concatenate((lax.expand_dims(logp_diff_t0, (1,)), lax.expand_dims(logp_diff_t1, (1,))), 1)
    return z_t1, logp_diff_t1


def batch_generator(batch_size: int = 512, d_dim: int = 2):
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    mean = jnp.zeros(d_dim)
    cov = jnp.eye(d_dim)
    while True:
        _, key = jrnd.split(rng)
        u = jrnd.multivariate_normal(
            key, mean=mean, cov=cov, shape=(batch_size,))
        log_pdf = jax.scipy.stats.multivariate_normal.logpdf(
            u, mean=mean, cov=cov)
        log_pdf = jnp.zeros_like(log_pdf)  # don't know why :/

        u_and_log_pu = lax.concatenate((u, lax.expand_dims(log_pdf, (1,))), 1)
        yield u_and_log_pu


def main():

    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model = Gen_CNF()
    test_inputs = lax.concatenate((jnp.ones((1, 2)), jnp.ones((1, 1))), 1)
    params = model.init(key, jnp.array(0.), test_inputs)

    from jax.tree_util import tree_map
    params = jax.tree_util.tree_map(lambda x: 0.1 * jnp.ones_like(x), params)
    # print(params)

    def NODE(params, batch): return neural_ode(
        params, batch, model, 0., 1., 2)

    def log_target_density(z): return pot_1(z)  # log_P

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    def loss(params, samples):
        x, log_px = NODE(params, samples)
        log_p_x = log_target_density(x)
        # return -1.*jnp.mean(log_p_x)
        return jnp.linalg.norm(log_p_x - log_px)
        # return jnp.mean(- sum_log_det - (log_p_x))

    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    batch_size = 128
    gen_batch = batch_generator(batch_size, 2)
    for i in range(1000):

        batch = next(gen_batch)
        params, opt_state, loss_value = step(params, opt_state, batch)
        if i % 100 == 0:
            print(f'step {i}, loss: {loss_value}')

    png = jrnd.PRNGKey(1)
    _, key = jrnd.split(png)

    u = jrnd.multivariate_normal(key, mean=jnp.zeros(
        (2,)), cov=jnp.eye(2), shape=(2000,))
    log_pu = jax.scipy.stats.multivariate_normal.logpdf(u, mean=jnp.zeros(
        (2,)), cov=jnp.eye(2))
    u_and_log_pu = lax.concatenate((u, lax.expand_dims(log_pu, (1,))), 1)
    x, _ = NODE(params, u_and_log_pu)

    plt.title('Sampels from the NormFlow')
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.savefig('Figures/CNF_twomoons.png')


if __name__ == '__main__':
    main()
