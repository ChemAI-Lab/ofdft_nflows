import argparse
from typing import Any, Union

import jax
from jax import lax,  numpy as jnp
import jax.random as jrnd
from jax._src import prng

import optax
from flax.training import checkpoints
from distrax import MultivariateNormalDiag
from distrax._src.distributions import distribution

from ofdft_normflows.cn_flows import neural_ode
from ofdft_normflows.cn_flows import Gen_CNFSimpleMLP as CNF
from ofdft_normflows.dft_distrax import DFTDistribution

import matplotlib.pyplot as plt

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]

# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

CKPT_DIR = "Results/acrolein_MLL"
FIG_DIR = "Figures/acrolein_MLL"


def batch_generator(prior_dist: distribution, batch_size: int = 512, l: Any = 0, bool_logp_z0_zeros: bool = True):
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    while True:
        z0_and_logp_z0 = prior_dist.sample_and_log_prob(
            seed=key, sample_shape=batch_size)
        z0 = z0_and_logp_z0[0]
        logp_z0 = z0_and_logp_z0[1][:, None]
        if bool_logp_z0_zeros:
            logp_z0 = jnp.zeros_like(logp_z0)

        yield lax.concatenate((z0, logp_z0), 1)


def _plotting(rho_pred, rho_true, XY, _label: Any):
    x, y = XY
    i, u2j = _label

    fig, ax = plt.subplots()
    # plt.clf()
    ax.set_title(f'Z = {u2j:.3f}')
    contour1 = ax.contour(x, y, rho_pred.reshape(x.shape), levels=25)
    # cbar1 = fig.colorbar(contour1, ax=ax)

    contour2 = ax.contour(x, y, rho_true.reshape(x.shape), cmap='plasma',
                          linestyles='dashed', levels=25)
    # cbar2 = fig.colorbar(contour2, ax=ax)
    geom = jnp.array([[-1.808864,  -0.137998,  0.000000],
                      [1.769114,  0.136549,  0.000000],
                      [0.588145,  -0.434423,  0.000000],
                      [-0.695203,  0.361447,  0.000000],
                      [-0.548852,  1.455362,  0.000000],
                      [0.477859,  -1.512556,  0.000000],
                      [2.688665,  -0.434186,  0.000000],
                      [1.880903,  1.213924,  0.000000]])
    ax.scatter(geom[:, 0], geom[:, 1],  marker='o', color='k', s=35)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/CNFrho_acrolein_{i}_KLrev.png')


def training(batch_size, epochs):
    mean = jnp.zeros((3,))
    cov = 0.1 * jnp.ones((3,))
    prior_dist = MultivariateNormalDiag(mean, cov)
    gen_batch = batch_generator(prior_dist, batch_size)

    atoms = ['O', 'C', 'C', 'C', 'H', 'H', 'H', 'H']
    geom = jnp.array([[-1.808864,  -0.137998,  0.000000],
                      [1.769114,  0.136549,  0.000000],
                      [0.588145,  -0.434423,  0.000000],
                      [-0.695203,  0.361447,  0.000000],
                      [-0.548852,  1.455362,  0.000000],
                      [0.477859,  -1.512556,  0.000000],
                      [2.688665,  -0.434186,  0.000000],
                      [1.880903,  1.213924,  0.000000]])
    dft_dist = DFTDistribution(atoms, geom)

    def log_target_density(x):
        return jnp.log(dft_dist.prob(dft_dist, x))  # +1E-7

    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model_rev = CNF(3, (96, 96,), bool_neg=False)
    model_fwd = CNF(3, (96, 96,), bool_neg=True)
    test_inputs = lax.concatenate((jnp.ones((1, 3)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)

    @jax.jit
    def NODE_rev(params, batch): return neural_ode(
        params, batch, model_rev, -1., 0., 3)

    @jax.jit
    def NODE_fwd(params, batch): return neural_ode(
        params, batch, model_fwd, 0., 1., 3)

    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(params)

    # load prev parameters
    # restored_state = checkpoints.restore_checkpoint(
    # ckpt_dir=CKPT_DIR, target=params, step=0)
    # params = restored_state

    def loss(params, samples):
        z0 = samples[:, :-1]
        zt, logp_zt = NODE_fwd(params, samples)
        logp_x = prior_dist.log_prob(
            z0)[:, jnp.newaxis] + logp_zt  # check the sign
        logp_true = log_target_density(zt)
        return jnp.linalg.norm(logp_x - logp_true)

    # @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(
            loss, has_aux=False)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    loss0 = jnp.inf
    for i in range(epochs+1):

        batch = next(gen_batch)
        params, opt_state, loss_value = step(params, opt_state, batch)

        if loss_value < loss0:
            params_opt, loss0 = params, loss_value
            print(f'step {i}, loss: {loss_value}')

        if i % 100 == 0:
            print(f'step {i}, loss: {loss_value}')

        if i % 100 == 0:

            # checkpointing model model
            checkpoints.save_checkpoint(
                ckpt_dir=CKPT_DIR, target=params, step=0, overwrite=True)

            # plotting results
            f_ = f"{CKPT_DIR}/acrolein_{i}.npy"
            u0 = jnp.linspace(-3., 3., 25)
            u1 = jnp.linspace(-3., 3., 25)
            u0_, u1_ = jnp.meshgrid(u0, u1)
            u01t = lax.concatenate(
                (jnp.expand_dims(u0_.ravel(), 1), jnp.expand_dims(u1_.ravel(), 1)), 1)

            u2 = jnp.linspace(-1.5, 1.5, 10)
            u2 = jnp.sort(jnp.append(u2, jnp.zeros(1)))

            results = {'grid': {'x': u0_, 'y': u1_, 'z': u2}}
            for j, u2j in enumerate(u2):
                zt = lax.concatenate(
                    (u01t, u2j*jnp.ones((u01t.shape[0], 1))), 1)
                logp_zt = jnp.zeros_like(zt[:, :1])
                zt_and_log_pzt = lax.concatenate((zt, logp_zt), 1)

                z0, logp_diff_z0 = NODE_rev(params_opt, zt_and_log_pzt)
                logp_x = prior_dist.log_prob(z0)[:, jnp.newaxis] - logp_diff_z0
                rho_pred = logp_x  # jnp.exp(logp_x)
                rho_true = log_target_density(zt)

                results.update({'j': {'cnf': rho_pred, 'dft': rho_true}})
                jnp.save(f_, results, allow_pickle=True)
                _plotting(rho_pred, rho_true, (u0_, u1_), (f"{j}-{i}", u2j))


def main():
    parser = argparse.ArgumentParser(description="Density fitting training")
    parser.add_argument("--epochs", type=int,
                        default=10000, help="training epochs")
    parser.add_argument("--bs", type=int, default=512, help="batch size")
    args = parser.parse_args()

    batch_size = args.bs
    epochs = args.epochs

    training(batch_size, epochs)


if __name__ == '__main__':
    main()

    # batch_size = 512
    # epochs = 10000

    # main(batch_size, epochs)
# https://www.molcas.org/documentation/manual/node28.html

#   O  -1.808864  -0.137998  0.000000
#   C  1.769114  0.136549  0.000000
#   C  0.588145  -0.434423  0.000000
#   C  -0.695203  0.361447  0.000000
#   H  -0.548852  1.455362  0.000000
#   H  0.477859  -1.512556  0.000000
#   H  2.688665  -0.434186  0.000000
#   H  1.880903  1.213924  0.000000
