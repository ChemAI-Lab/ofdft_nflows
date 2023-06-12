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

from ofdft_normflows.functionals import _kinetic
from ofdft_normflows.functionals import harmonic_potential
from ofdft_normflows.cn_flows import Gen_CNF, neural_ode

import matplotlib.pyplot as plt

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]

jax.config.update("jax_enable_x64", True)


def batch_generator(batch_size: int = 512, d_dim: int = 3):
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    mean = 5*jnp.ones(d_dim)
    cov = 5.*jnp.eye(d_dim)
    while True:
        _, key = jrnd.split(key)
        u = jrnd.multivariate_normal(
            key, mean=mean, cov=cov, shape=(batch_size,))
        log_pdf = jax.scipy.stats.multivariate_normal.logpdf(
            u, mean=mean, cov=cov)
        # log_pdf = jnp.zeros_like(log_pdf)  # don't know why :/

        u_and_log_pu = lax.concatenate((u, lax.expand_dims(log_pdf, (1,))), 1)
        yield u_and_log_pu


def log_p_z0(samples):
    d_dim = samples.shape[1]
    mean = 5.*jnp.ones(d_dim)
    cov = 5.*jnp.eye(d_dim)
    logp = jax.scipy.stats.multivariate_normal.logpdf(
        samples, mean=mean, cov=cov)
    return logp.reshape(samples.shape[0], 1)


def batch_generator_rho(batch_size):

    mol = pyscf.M(atom='''
                H  0. 0. 0.
            ''', basis='sto-3g', spin=1)

    mf = dft.RKS(mol)
    mf.kernel()
    dm = mf.make_rdm1()
    coords = mf.grids.coords
    weights = mf.grids.weights

    def _rho_eval(coords: Any):
        coords = np.asanyarray(coords)
        ao_value = numint.eval_ao(mol, coords, deriv=1)
        rho_and_grho = numint.eval_rho(mol, ao_value, dm, xctype='GGA')
        return jnp.asarray(rho_and_grho[0])  # [:, jnp.newaxis]

    rho = _rho_eval(coords)
    X = jnp.asarray(coords, dtype=jnp.float32)
    i0 = jnp.arange(rho.shape[0])

    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    while True:
        _, key = jrnd.split(key)
        xi0 = jrnd.choice(key, i0, shape=(batch_size,), p=rho)
        x = X[xi0]
        logp_diff_t1 = jnp.log(rho[xi0])[:, jnp.newaxis]
        # logp_diff_t1 = jnp.zeros((batch_size, 1), dtype=jnp.float32)
        yield lax.concatenate((x, logp_diff_t1), 1)


def _rho_eval(coords: Any):

    mol = pyscf.M(atom='''
                H  0. 0. 0.
            ''', basis='sto-3g', spin=1)

    mf = dft.RKS(mol)
    mf.kernel()
    dm = mf.make_rdm1()
    coords = np.asanyarray(coords)
    ao_value = numint.eval_ao(mol, coords, deriv=1)
    rho_and_grho = numint.eval_rho(mol, ao_value, dm, xctype='GGA')
    return jnp.asarray(rho_and_grho[0])[:, jnp.newaxis]


def main(batch_size, epochs):

    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model = Gen_CNF(3)
    test_inputs = lax.concatenate((jnp.ones((1, 3)), jnp.ones((1, 1))), 1)
    params = model.init(key, jnp.array(0.), test_inputs)

    def NODE(params, batch): return neural_ode(
        params, batch, model, 0., 10., 3)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    def loss(params, samples):
        zt0, logp_zt0 = NODE(params, samples)
        logp_x = log_p_z0(zt0) - logp_zt0
        return -1.*jnp.mean(logp_x)

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

    # z0 = jrnd.multivariate_normal(key, mean=jnp.zeros(
    #     (3,)), cov=0.1*jnp.eye(3), shape=(2000,))
    # logp_z0 = jnp.zeros((2000, 1))
    # u_and_log_pu = lax.concatenate((z0, logp_z0), 1)

    # model1 = Gen_CNF(bool_neg=True)
    # zt, logp_zt = neural_ode(params, u_and_log_pu, model1, -10., 0., 3)

    # plt.title('Samples from the NormFlow')
    # plt.scatter(zt[:, 0], zt[:, 1])
    # plt.savefig('Figures/CNF_Hatom.png')

    u0 = jnp.linspace(-5.5, 5.5, 25)
    u1 = jnp.linspace(-5.5, 5.5, 25)
    u0_, u1_ = jnp.meshgrid(u0, u1)
    u01t = lax.concatenate(
        (jnp.expand_dims(u0_.ravel(), 1), jnp.expand_dims(u1_.ravel(), 1)), 1)

    u2 = jnp.linspace(-5.5, 5.5, 10)
    for i, u2i in enumerate(u2):
        zt = lax.concatenate(
            (u01t, jnp.ones((u01t.shape[0], 1))), 1)
        logp_zt = jnp.zeros_like(zt[:, :1])
        zt_and_log_pzt = lax.concatenate((zt, logp_zt), 1)

        z0, logp_diff_z0 = neural_ode(
            params, zt_and_log_pzt, model, 0., 10., 3)
        logp_x = log_p_z0(z0) - logp_diff_z0
        rho_pred = jnp.exp(logp_x)
        rho_true = _rho_eval(zt)

        plt.figure(i)
        plt.clf()
        plt.title(f'Z = {u2i:.3f}')
        plt.contour(u0_, u1_, rho_pred.reshape(u0_.shape), label='CNF')
        plt.contour(u0_, u1_, rho_true.reshape(u0_.shape),
                    linestyles='dashed', label='QChem')
        plt.legend()
        plt.tight_layout()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f'Figures/CNF_rho_Hatom_{i}.png')


if __name__ == '__main__':

    batch_size = 512
    epochs = 100

    main(batch_size, epochs)
