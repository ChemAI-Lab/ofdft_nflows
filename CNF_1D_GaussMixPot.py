import os
import argparse
from typing import Any, Union

import jax
from jax import lax, numpy as jnp
import jax.random as jrnd
from jax._src import prng

from flax.training import checkpoints
import optax
from distrax import Normal

from ofdft_normflows.functionals import _kinetic, GaussianPotential1D,  GaussianPotential1D_pot, Coulomb_potential
from ofdft_normflows.cn_flows import neural_ode
from ofdft_normflows.cn_flows import Gen_CNFSimpleMLP as CNF

import matplotlib.pyplot as plt

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]

jax.config.update("jax_enable_x64", True)

BHOR = 1.8897259886  # 1AA to BHOR
CKPT_DIR = "Results/GP_pot"
FIG_DIR = "Figures/GP_pot"


def training(batch_size: int = 256, epochs: int = 100):
    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model_rev = CNF(1, (96, 96,), bool_neg=False)
    model_fwd = CNF(1, (96, 96,), bool_neg=True)
    test_inputs = lax.concatenate((jnp.ones((1, 1)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)

    @jax.jit
    def NODE_rev(params, batch): return neural_ode(
        params, batch, model_rev, -1., 0., 1)

    @jax.jit
    def NODE_fwd(params, batch): return neural_ode(
        params, batch, model_fwd, 0., 1., 1)

    t_functional = _kinetic('TF1D')
    v_functional = lambda *args: GaussianPotential1D(*args)

    prior_dist = Normal(jnp.zeros(1), 0.1*jnp.ones(1))

    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(params)

    # load prev parameters
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=CKPT_DIR, target=params, step=0)
    params = restored_state

    @jax.jit
    def rho(params, samples):
        zt0 = samples[:, :1]
        zt, logp_zt = NODE_fwd(params, samples)
        logp_x = prior_dist.log_prob(zt0) + logp_zt
        return jnp.exp(logp_x)

    @jax.jit
    def T(params, samples):
        zt, _ = NODE_fwd(params, samples)
        return zt

    @jax.jit
    def loss(params, u_samples):
        u_samples, up_samples = u_samples[:batch_size,
                                          :], u_samples[batch_size:, :]
        gauss_v = v_functional(params, u_samples, T)
        t = t_functional(params, u_samples, rho)
        c_v = Coulomb_potential(params, u_samples, up_samples, T)
        return t + gauss_v + c_v, {"t": t, "v": gauss_v, "c": c_v}

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

        if i % 5 == 0:
            _s = f"step {i}, E: {loss_epoch:.5f}, T: {losses['t']:.5f}, V: {losses['v']:.5f}, C: {losses['c']:.5f}"
            print(_s,
                  file=open('loss_epochs_GPpot.txt', 'a'))

        if loss_epoch < loss0:
            params_opt, loss0 = params, loss_epoch
            # checkpointing model model
            checkpoints.save_checkpoint(
                ckpt_dir=CKPT_DIR, target=params, step=0, overwrite=True)

        if i % 10 == 0:
            zt = jnp.linspace(-5., 5., 1000)[:, jnp.newaxis]
            zt_and_logp_zt = lax.concatenate((zt, jnp.zeros_like(zt)), 1)

            z0, logp_diff_z0 = NODE_rev(params_opt, zt_and_logp_zt)
            logp_x = prior_dist.log_prob(z0) - logp_diff_z0
            rho_pred = logp_x  # jnp.exp(logp_x)
            def model_identity(params, x): return x
            def f_v(x): return GaussianPotential1D_pot(None, x, model_identity)
            y_GP_pot = f_v(zt)

            plt.figure(0)
            plt.clf()
            plt.title(f'epoch {i}')
            plt.plot(zt/BHOR, jnp.exp(rho_pred)*BHOR,
                     color='tab:blue', label=r'$\rho(x)$')
            plt.plot(zt/BHOR, y_GP_pot,
                     ls='--', color='k', label=r'$V_{GP}(x)$')
            plt.xlabel('x')
            plt.ylabel('Energy units')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{FIG_DIR}/rho_and_GP_pot_{i}.png')


def main():
    parser = argparse.ArgumentParser(description="Density fitting training")
    parser.add_argument("--epochs", type=int,
                        default=10, help="training epochs")
    parser.add_argument("--bs", type=int, default=512, help="batch size")
    args = parser.parse_args()

    batch_size = args.bs
    epochs = args.epochs

    cwd = os.getcwd()
    rwd = os.path.join(cwd, CKPT_DIR)
    if not os.path.exists(rwd):
        os.makedirs(rwd)
    fwd = os.path.join(cwd, FIG_DIR)
    if not os.path.exists(fwd):
        os.makedirs(fwd)

    training(batch_size, epochs)


if __name__ == "__main__":
    main()
