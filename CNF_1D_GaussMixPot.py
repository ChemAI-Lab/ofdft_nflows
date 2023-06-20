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
from distrax import Normal

from ofdft_normflows.functionals import _kinetic, GaussianPotential1D, Coulomb_potential
from ofdft_normflows.cn_flows import Gen_CNF, neural_ode, CNFMLP
from ofdft_normflows.flows import NormFlow

import matplotlib.pyplot as plt

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]

jax.config.update("jax_enable_x64", True)


def main(batch_size: int = 256, epochs: int = 100):
    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    # model = Gen_CNF(1)
    model = CNFMLP(1, (100,))
    test_inputs = lax.concatenate((jnp.ones((1, 1)), jnp.ones((1, 1))), 1)
    params = model.init(key, jnp.array(0.), test_inputs)

    def NODE(params, batch): return neural_ode(
        params, batch, model, 0., 10., 1)

    # model = NormFlow(10, 1)
    # params = model.init(key, jnp.zeros((1, 1)))
    # z, ldj = model.apply(params, jrnd
    #                      .uniform(key, (2, 1)))

    t_functional = _kinetic('TF1D')
    v_functional = lambda *args: GaussianPotential1D(*args)

    prior_dist = Normal(jnp.zeros(1), 0.1*jnp.ones(1))

    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(params)

    @jax.jit
    def rho(params, samples):
        zt0 = samples[:, :1]
        zt, logp_zt = NODE(params, samples)
        logp_x = prior_dist.log_prob(zt0) - logp_zt
        return jnp.exp(logp_x)

    @jax.jit
    def T(params, samples):
        zt, _ = NODE(params, samples)
        return zt

    @jax.jit
    def loss(params, u_samples):
        u_samples, up_samples = u_samples[:batch_size,
                                          :], u_samples[batch_size:, :]
        gauss_v = v_functional(params, u_samples, T)
        t = t_functional(params, u_samples, rho)
        c_v = Coulomb_potential(params, u_samples, up_samples, T)
        return t + gauss_v + c_v, {"t": t, "v": gauss_v + c_v}

    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(
            loss, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    def batches_generator(key: prng.PRNGKeyArray, batch_size: int):
        while True:
            _, key = jrnd.split(key)
            samples = prior_dist.sample(seed=key, sample_shape=batch_size)
            logp_samples = prior_dist.log_prob(samples)
            samples = lax.concatenate((samples, logp_samples), 1)
            yield samples

    _, key = jrnd.split(key)
    gen_batches = batches_generator(key, 2*batch_size)
    loss0 = jnp.inf
    for i in range(epochs+1):

        batch = next(gen_batches)
        params, opt_state, loss_value = step(params, opt_state, batch)
        loss_epoch, losses = loss_value
        if i % 10 == 0:
            print(f'step {i}, loss: {loss_epoch}')
        if loss_epoch < loss0:
            params_opt, loss0 = params, loss_epoch

    x = jnp.linspace(-4., 4., 1000)[:, jnp.newaxis]
    logp_samples = prior_dist.log_prob(x)
    print(x[:, jnp.newaxis, :].shape)
    plt.plot(x, rho(params_opt, x))
    plt.plot(x, prior_dist.prob(x))
    def T(params, x): return x

    def f_test(x):
        print(x.shape)
        return jnp.sum(1./x, axis=-1)

    f_v = vmap(GaussianPotential1D, in_axes=(None, 0, None))
    GP_pot = f_v(None, x[:, jnp.newaxis, :], T)
    plt.plot(x, GP_pot)
    plt.savefig('GaussPot.png')


if __name__ == "__main__":
    main(512, 10000)
    # x = jnp.linspace(-4., 4., 1000)[:, jnp.newaxis]
    # xv = x[:, jnp.newaxis, :]

    # def T(params, x): return x

    # def f_test(x):
    #     print(x.shape)
    #     return jnp.sum(1./x, axis=-1)

    # f_v = vmap(GaussianPotential1D, in_axes=(None, 0, None))
    # y = f_v(None, xv, T)
    # print(y.shape)
    # plt.plot(x, y)
    # plt.savefig('1d_pot.png')
