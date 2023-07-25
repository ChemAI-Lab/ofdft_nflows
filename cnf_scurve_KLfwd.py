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

from ofdft_normflows.functionals import _kinetic
from ofdft_normflows.functionals import harmonic_potential
from ofdft_normflows.cn_flows import neural_ode, Gen_CNFRicky
from ofdft_normflows.cn_flows import Gen_CNFSimpleMLP as CNF

import matplotlib.pyplot as plt

from sklearn.datasets import make_s_curve


Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]

# jax.config.update("jax_enable_x32", True)


# file from https://huggingface.co/flax-community/NeuralODE_SDE/raw/main/train_cnf.py


# @nn.jit

# ---------------


def get_batch_scurve(num_samples):
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)
    while True:
        points, _ = make_s_curve(
            n_samples=num_samples, noise=0.05)
        x1 = jnp.array(points, dtype=jnp.float32)[:, :1]
        x2 = jnp.array(points, dtype=jnp.float32)[:, 2:]
        x = lax.concatenate((x1, x2), 1)
        logp_diff_t1 = jnp.zeros((num_samples, 1), dtype=jnp.float32)

        yield lax.concatenate((x, logp_diff_t1), 1)
# ---------------


def main(batch_size, epochs):

    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    # model_rev = CNF(2, (200, 200,), bool_neg=False)
    # model_fwd = CNF(2, (200, 200,), bool_neg=True)
    model_rev = Gen_CNFRicky(2, bool_neg=False)
    model_fwd = Gen_CNFRicky(2, bool_neg=True)
    test_inputs = lax.concatenate((jnp.ones((1, 2)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)

    @jax.jit
    def NODE_rev(params, batch): return neural_ode(
        params, batch, model_rev, -10., 0., 2)

    @jax.jit
    def NODE_fwd(params, batch): return neural_ode(
        params, batch, model_fwd, 0., 10., 2)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def log_p_z0(samples):
        d_dim = samples.shape[1]
        mean = jnp.zeros(d_dim)
        cov = 0.1*jnp.eye(d_dim)
        logp = jax.scipy.stats.multivariate_normal.logpdf(
            samples, mean=mean, cov=cov)
        return logp.reshape(samples.shape[0], 1)

    @jax.jit
    def loss(params, samples):
        zt0, logp_zt0 = NODE_rev(params, samples)
        # only works with - (when + it gives two points)
        logp_x = log_p_z0(zt0) - logp_zt0
        return -1.*jnp.mean(logp_x)

    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    gen_batch = get_batch_scurve(batch_size)
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
        (2,)), cov=0.1*jnp.eye(2), shape=(2000,))
    logp_z0 = jnp.zeros((2000, 1))
    u_and_log_pu = lax.concatenate((z0, logp_z0), 1)

    zt, logp_zt = NODE_fwd(params_opt, u_and_log_pu)

    plt.title('Samples from the NormFlow')
    plt.scatter(zt[:, 0], zt[:, 1])
    plt.savefig('Figures/CNF_twomoons_KLfwd.png')

    u0 = jnp.linspace(-1.5, 1.5, 100)
    u1 = jnp.linspace(-1.5, 1.5, 100)
    u0_, u1_ = jnp.meshgrid(u0, u1)
    zt = lax.concatenate(
        (jnp.expand_dims(u0_.ravel(), 1), jnp.expand_dims(u1_.ravel(), 1)), 1)
    logp_zt = jnp.zeros_like(zt[:, :1])
    zt_and_log_pzt = lax.concatenate((zt, logp_zt), 1)

    z0, logp_diff_z0 = NODE_rev(params_opt, zt_and_log_pzt)
    logp_x = log_p_z0(z0) - logp_diff_z0

    plt.figure(0)
    plt.clf()
    plt.contour(u0_, u1_, logp_x.reshape(u0_.shape))
    plt.savefig('Figures/CNF_logP_twomoons_KLfwd.png')


if __name__ == '__main__':
    batch_size = 254
    epochs = 2000

    main(batch_size, epochs)
