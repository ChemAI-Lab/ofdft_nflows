import os
import argparse
from typing import Any, Union
import pandas as pd

import jax
from jax import lax, vmap, numpy as jnp
import jax.random as jrnd
from jax._src import prng

from flax.training import checkpoints
import optax
from distrax import Normal

from ofdft_normflows.functionals import _kinetic, GaussianPotential1D,  GaussianPotential1D_pot, Coulomb_potential, Hartree_potential
from ofdft_normflows.cn_flows import neural_ode
from ofdft_normflows.cn_flows import Gen_CNFSimpleMLP as CNF
from ofdft_normflows.cn_flows import Gen_CNFRicky as CNFRicky

import matplotlib.pyplot as plt

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]

jax.config.update("jax_enable_x64", True)

BHOR = 1.8897259886  # 1AA to BHOR
# CKPT_DIR = "Results/GP_pot"
# FIG_DIR = "Figures/GP_pot"


def get_scheduler(epochs: int, type: str = 'zero'):
    if type == 'zero':
        return optax.constant_schedule(0.0)
    elif type == 'one':
        return optax.constant_schedule(1.)
    elif type == 'cos_deacay':
        return optax.warmup_cosine_decay_schedule(
            init_value=.0,
            peak_value=1.0,
            warmup_steps=1,
            decay_steps=epochs,
            end_value=1.0,
        )
    elif type == 'mix':
        constant_scheduler_min = optax.constant_schedule(0.0)
        cosine_decay_scheduler = optax.cosine_onecycle_schedule(transition_steps=epochs, peak_value=1.,
                                                                div_factor=50., final_div_factor=1.)
        constant_scheduler_max = optax.constant_schedule(1.0)
        return optax.join_schedules([constant_scheduler_min, cosine_decay_scheduler,
                                     constant_scheduler_max], boundaries=[epochs/4, 2*epochs/4])


def load_true_results(n_particles: int):
    import numpy as onp
    d_ = f'Data_1D_GaussMixPot/true_rho_grid_Ne_{n_particles}.txt'
    data = jnp.array(onp.loadtxt(d_))
    return data


def training(n_particles: int = 2, batch_size: int = 256, epochs: int = 100, bool_load_params: bool = False, scheduler_type: str = 'zero'):
    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model_rev = CNF(1, (264, 264,), bool_neg=False)
    model_fwd = CNF(1, (264, 264,), bool_neg=True)
    # model_rev = CNFRicky(1, 512, 512, bool_neg=False)
    # model_fwd = CNFRicky(1, 512, 512, bool_neg=True)
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

    prior_dist = Normal(jnp.zeros(1), 1.*jnp.ones(1))

    # optimizer = optax.adam(learning_rate=1E-3)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=1E-3),
    )
    opt_state = optimizer.init(params)

    # load prev parameters
    if bool_load_params:
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=CKPT_DIR, target=params, step=0)
        params = restored_state

    @jax.jit
    def rho(params, samples):
        zt0 = samples[:, :1]
        zt, logp_zt = NODE_fwd(params, samples)
        # logp_x = prior_dist.log_prob(zt0) + logp_zt
        return jnp.exp(logp_zt)  # logp_x

    @jax.jit
    def T(params, samples):
        zt, _ = NODE_fwd(params, samples)
        return zt

    @jax.jit
    def loss(params, u_samples, ci):
        u_samples, up_samples = u_samples[:batch_size,
                                          :], u_samples[batch_size:, :]
        gauss_v = v_functional(params, u_samples, T)
        t = t_functional(params, u_samples, rho)
        c_v = Hartree_potential(params, u_samples, up_samples, T)
        e = (n_particles**3)*t + n_particles*gauss_v + ci*(n_particles**2)*c_v
        return jnp.mean(e), {"t": (n_particles**3)*jnp.mean(t),
                             "v": n_particles*jnp.mean(gauss_v),
                             "c": (n_particles**2)*jnp.mean(c_v)}

    @jax.jit
    def step(params, opt_state, batch, ci):
        loss_value, grads = jax.value_and_grad(
            loss, has_aux=True)(params, batch, ci)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    def batches_generator(key: prng.PRNGKeyArray, batch_size: int):
        while True:
            _, key = jrnd.split(key)
            samples = prior_dist.sample(seed=key, sample_shape=batch_size)
            logp_samples = prior_dist.log_prob(samples)
            samples0 = lax.concatenate((samples, logp_samples), 1)

            _, key = jrnd.split(key)
            samples = prior_dist.sample(seed=key, sample_shape=batch_size)
            logp_samples = prior_dist.log_prob(samples)
            samples1 = lax.concatenate((samples, logp_samples), 1)

            yield lax.concatenate((samples0, samples1), 0)

    @jax.jit
    def _integral(params):
        zt = jnp.linspace(-5., 5., num=1024)[:, jnp.newaxis]
        zt_and_logp_zt = lax.concatenate((zt, jnp.zeros_like(zt)), 1)
        z0, logp_diff_z0 = NODE_rev(params, zt_and_logp_zt)
        logp_x = prior_dist.log_prob(z0) - logp_diff_z0
        p_x = jnp.exp(logp_x)
        return jnp.trapz(p_x.ravel(), zt.ravel(), jnp.abs(zt[1, 0]-zt[0, 0])), jnp.mean(logp_diff_z0)

    loss0 = jnp.inf
    df = pd.DataFrame()
    _, key = jrnd.split(key)
    gen_batches = batches_generator(key, batch_size)
    scheduler = get_scheduler(epochs, scheduler_type)
    x_and_rho_true = load_true_results(
        n_particles)  # read results from JCTC paper

    for i in range(epochs+1):
        ci = scheduler(i)
        batch = next(gen_batches)
        params, opt_state, loss_value = step(params, opt_state, batch, ci)
        loss_epoch, losses = loss_value

        norm_integral, log_det_Jac = _integral(params)
        mean_energy = losses['t']+losses['v']+losses['c']
        r_ = {'epoch': i,
              'L': loss_epoch, 'E': mean_energy,
              'T': losses['t'], 'V': losses['v'], 'C': losses['c'],
              'I': norm_integral, 'ci': ci, 'logDetJac': log_det_Jac
              }
        df = pd.concat([df, pd.DataFrame(r_, index=[0])], ignore_index=True)
        df.to_csv(
            f"{CKPT_DIR}/training_trajectory_Ne_{n_particles}.csv", index=False)

        if i % 5 == 0:
            _s = f"step {i}, L: {loss_epoch:.3f}, E:{mean_energy:.3f}\
            T: {losses['t']:.5f}, V: {losses['v']:.5f}, C: {losses['c']:.5f}, \
            I: {norm_integral:.4f}, ci: {ci:.5f}, logDetJac: {log_det_Jac:.4}"
            print(_s,
                  file=open(f"{CKPT_DIR}/loss_epochs_GPpot_Ne_{n_particles}.txt", 'a'))

        if loss_epoch < loss0:
            params_opt, loss0 = params, loss_epoch
            # checkpointing model model
            checkpoints.save_checkpoint(
                ckpt_dir=CKPT_DIR, target=params, step=0, overwrite=True)

        if i % 20 == 0 or i < 25:
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
            plt.title(
                f'epoch {i}, L = {loss_epoch:.3f}, E = {mean_energy:.3f}, ci = {ci:.3f}')
            plt.plot(x_and_rho_true[:, 0], x_and_rho_true[:, 1],
                     color='k', ls=":", label=r"$\hat{\rho}(x)$")
            plt.plot(zt, n_particles*jnp.exp(rho_pred),
                     color='tab:blue', label=r'$N_{e}\;\rho_{NF}(x)$')
            plt.plot(zt, y_GP_pot,
                     ls='--', color='k', label=r'$V_{GP}(x)$')
            plt.xlabel('x [Bhor]')
            # plt.ylabel('Energy units')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{FIG_DIR}/rho_and_GP_pot_{i}.png')


def main():
    parser = argparse.ArgumentParser(description="Density fitting training")
    parser.add_argument("--epochs", type=int,
                        default=1500, help="training epochs")
    parser.add_argument("--bs", type=int, default=512, help="batch size")
    parser.add_argument("--params", type=bool, default=False,
                        help="load pre-trained model")
    parser.add_argument("--N", type=int, default=1, help="number of particles")
    parser.add_argument("--sched", type=str, default='zero',
                        help="Hartree integral scheduler")
    args = parser.parse_args()

    batch_size = args.bs
    epochs = args.epochs
    bool_params = args.params
    n_particles = args.N
    scheduler_type = args.sched

    global CKPT_DIR
    global FIG_DIR
    CKPT_DIR = f"Results/GP_pot_Ne_{n_particles}_Hsched_{scheduler_type}"
    FIG_DIR = f"{CKPT_DIR}/Figures"

    cwd = os.getcwd()
    rwd = os.path.join(cwd, CKPT_DIR)
    if not os.path.exists(rwd):
        os.makedirs(rwd)
    fwd = os.path.join(cwd, FIG_DIR)
    if not os.path.exists(fwd):
        os.makedirs(fwd)

    training(n_particles, batch_size, epochs, bool_params, scheduler_type)


if __name__ == "__main__":
    # main()
    epochs = 500
    constant_scheduler_min = optax.constant_schedule(0.0)
    cosine_decay_scheduler = optax.cosine_onecycle_schedule(transition_steps=epochs, peak_value=1.,
                                                            div_factor=50., final_div_factor=1.)
    constant_scheduler_max = optax.constant_schedule(1.0)
    scheduler = optax.join_schedules([constant_scheduler_min, cosine_decay_scheduler,
                                      constant_scheduler_max], boundaries=[epochs/3, 2*epochs/3])

    ci = [scheduler(i) for i in range(epochs)]

    plt.plot(ci)
    plt.show()
