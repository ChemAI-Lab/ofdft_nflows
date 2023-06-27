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

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]

# jax.config.update("jax_enable_x64", True)


def log_p_z0(samples):
    d_dim = samples.shape[1]
    # mean = jnp.zeros(d_dim)  # 5.*jnp.ones(d_dim)
    mean = jnp.array([[0., 0., 0.], [0.76, 0., 0.]])
    cov = 0.01*jnp.eye(d_dim)
    logp = jax.scipy.stats.multivariate_normal.logpdf(
        samples, mean=mean, cov=cov)
    return logp.reshape(samples.shape[0], 1)


def batch_generator_rho(batch_size: int, _level: int = 5):

    mol = pyscf.M(atom='''
                H  0. 0. 0.
                H  0.76 0. 0.
            ''', basis='sto-3g')  # , spin=1
    mf = dft.RKS(mol)
    mf.kernel()
    mf.grids.level = _level
    mf.grids.build(with_non0tab=True)
    dm = mf.make_rdm1()

    coords = mf.grids.coords
    weights = mf.grids.weights

    def _rho_eval(coords: Any):  # int rho(r) / N dr = 1;
        coords = np.asanyarray(coords)
        ao_value = numint.eval_ao(mol, coords, deriv=1)
        rho_and_grho = numint.eval_rho(mol, ao_value, dm, xctype='GGA')
        return jnp.asarray(rho_and_grho[0], dtype=jnp.float32)/mol.tot_electrons()

    rho = _rho_eval(coords)
    X = jnp.asarray(coords, dtype=jnp.float32)
    i0 = jnp.arange(rho.shape[0])

    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    while True:
        _, key = jrnd.split(key)
        xi0 = jrnd.choice(key, i0, shape=(batch_size,), p=rho, replace=True)

        # xi0 = jrnd.permutation(key, i0)
        # xi0 = xi0[:batch_size]

        x = X[xi0]
        logp_diff_t1 = jnp.log(rho[xi0])[:, jnp.newaxis]
        # logp_diff_t1 = jnp.array(logp_diff_t1, dtype=jnp.float64)
        logp_diff_t1 = jnp.zeros((batch_size, 1), dtype=jnp.float32)
        yield lax.concatenate((x, logp_diff_t1), 1), jnp.log(rho[xi0])[:, jnp.newaxis]


def batch_generator(batch_size: int, d_dim: int = 2):
    # mean = jnp.zeros(d_dim)  # 5.*jnp.ones(d_dim)
    mean = jnp.array([[0., 0., 0.], [0.76, 0., 0.]])
    cov = 0.01*jnp.eye(d_dim)
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)
    logp_z0 = jnp.zeros((batch_size, 1))
    while True:
        _, key = jrnd.split(key)
        samples = jrnd.multivariate_normal(
            key, mean=mean, cov=cov, shape=(batch_size,))
        # logp_z0 = jnp.zeros((batch_size, 1))
        logp_z0 = jax.scipy.stats.multivariate_normal.logpdf(
            samples, mean=mean, cov=cov)
        yield lax.concatenate((samples, logp_z0[:, None]), 1)


def _rho_eval(coords: Any):

    mol = pyscf.M(atom='''
                H  0. 0. 0.
                H  0.76 0. 0.
            ''', basis='sto-3g')  # , spin=1

    mf = dft.RKS(mol)
    mf.kernel()
    dm = mf.make_rdm1()
    coords = np.asanyarray(coords)
    ao_value = numint.eval_ao(mol, coords, deriv=1)
    rho_and_grho = numint.eval_rho(mol, ao_value, dm, xctype='GGA')
    return jnp.asarray(rho_and_grho[0])[:, jnp.newaxis]/mol.tot_electrons()


def _plotting(rho_pred, rho_true, XY, _label: Any):
    x, y = XY
    i, u2i = _label

    fig, ax = plt.subplots()
    # plt.clf()
    ax.set_title(f'Z = {u2i:.3f}')
    contour1 = ax.contour(x, y, rho_pred.reshape(x.shape), levels=25)
    # cbar1 = fig.colorbar(contour1, ax=ax)

    contour2 = ax.contour(x, y, rho_true.reshape(x.shape), cmap='plasma',
                          linestyles='dashed', levels=25)
    # cbar2 = fig.colorbar(contour2, ax=ax)
    ax.scatter(jnp.array([0.76, 0.])/BOHR, jnp.zeros(
        2),  marker='o', color='k', s=35)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    plt.savefig(f'Figures/CNF_rho_H2_{i}_H2Pz.png')


def main(batch_size, epochs):

    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model_rev = CNF(3, (200, 200,), bool_neg=False)
    model_fwd = CNF(3, (200, 200,), bool_neg=True)
    # model_rev = Gen_CNFRicky(3, bool_neg=False)
    # model_fwd = Gen_CNFRicky(3, bool_neg=True)
    test_inputs = lax.concatenate((jnp.ones((1, 3)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)

    @jax.jit
    def NODE_rev(params, batch): return neural_ode(
        params, batch, model_rev, -10., 0., 3)

    @jax.jit
    def NODE_fwd(params, batch): return neural_ode(
        params, batch, model_fwd, 0., 10., 3)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss(params, samples):
        samples, log_true_rho = samples
        zt0, logp_zt0 = NODE_rev(params, samples)
        logp_x = log_p_z0(zt0) - logp_zt0
        return - 1.*jnp.mean(logp_x)
        # return jnp.mean(logp_x - log_true_rho)

    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    gen_batch = batch_generator_rho(batch_size)
    loss0 = jnp.inf
    for i in range(epochs+1):

        batch = next(gen_batch)
        params, opt_state, loss_value = step(params, opt_state, batch)
        if i % 10 == 0:
            print(f'step {i}, loss: {loss_value}')
        if loss_value < loss0:
            params_opt, loss0 = params, loss_value

    png = jrnd.PRNGKey(1)
    _, key = jrnd.split(png)

    u0 = jnp.linspace(-10., 10.5, 10)
    u1 = jnp.linspace(-10.5, 10.5, 10)
    u0_, u1_ = jnp.meshgrid(u0, u1)
    u01t = lax.concatenate(
        (jnp.expand_dims(u0_.ravel(), 1), jnp.expand_dims(u1_.ravel(), 1)), 1)

    u2 = jnp.linspace(-2.5, 2.5, 15)
    u2 = jnp.append(u2, jnp.zeros(1))
    for i, u2i in enumerate(u2):
        zt = lax.concatenate(
            (u01t, u2i*jnp.ones((u01t.shape[0], 1))), 1)
        logp_zt = jnp.zeros_like(zt[:, :1])
        zt_and_log_pzt = lax.concatenate((zt, logp_zt), 1)

        z0, logp_diff_z0 = NODE_rev(params_opt, zt_and_log_pzt)
        logp_x = log_p_z0(z0) - logp_diff_z0
        rho_pred = logp_x  # jnp.exp(logp_x)
        rho_true = jnp.log(_rho_eval(zt))

        _plotting(rho_pred[:, None], rho_true, (u0_, u1_), (i, u2i))


if __name__ == '__main__':

    batch_size = int(512/2)
    epochs = 500

    main(batch_size, epochs)


# def batch_generator(batch_size: int = 512, d_dim: int = 3):
#     rng = jrnd.PRNGKey(0)
#     _, key = jrnd.split(rng)
#     mean = 5*jnp.ones(d_dim)
#     cov = .1*jnp.eye(d_dim)
#     while True:
#         _, key = jrnd.split(key)
#         u = jrnd.multivariate_normal(
#             key, mean=mean, cov=cov, shape=(batch_size,))
#         # log_pdf = jax.scipy.stats.multivariate_normal.logpdf(
#         # u, mean=mean, cov=cov)
#         log_pdf = jnp.zeros_like(log_pdf)  # don't know why :/
#         u_and_log_pu = lax.concatenate((u, lax.expand_dims(log_pdf, (1,))), 1)
#         yield u_and_log_pu


# z0 = jrnd.multivariate_normal(key, mean=jnp.zeros(
#     (3,)), cov=0.1*jnp.eye(3), shape=(2000,))
# logp_z0 = jnp.zeros((2000, 1))
# u_and_log_pu = lax.concatenate((z0, logp_z0), 1)

# model1 = Gen_CNF(bool_neg=True)
# zt, logp_zt = neural_ode(params, u_and_log_pu, model1, -10., 0., 3)

# plt.title('Samples from the NormFlow')
# plt.scatter(zt[:, 0], zt[:, 1])
# plt.savefig('Figures/CNF_Hatom.png')
