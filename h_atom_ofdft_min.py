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
from distrax import MultivariateNormalDiag, Laplace

from ofdft_normflows.functionals import _kinetic, Dirac_exchange, cusp_condition
from ofdft_normflows.functionals import Hartree_potential, Nuclei_potential
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


def get_scheduler(epochs: int, sched_type: str = 'zero'):
    try:
        float(sched_type)
        v = float(sched_type)
        return optax.constant_schedule(v)
    except ValueError:
        if sched_type == 'zero':
            return optax.constant_schedule(0.0)
        elif sched_type == 'one':
            return optax.constant_schedule(1.)
        elif sched_type == 'cos_deacay':
            return optax.warmup_cosine_decay_schedule(
                init_value=.0,
                peak_value=1.0,
                warmup_steps=1,
                decay_steps=epochs,
                end_value=1.0,
            )
        elif sched_type == 'mix':
            constant_scheduler_min = optax.constant_schedule(0.0)
            cosine_decay_scheduler = optax.cosine_onecycle_schedule(transition_steps=epochs, peak_value=1.,
                                                                    div_factor=50., final_div_factor=1.)
            constant_scheduler_max = optax.constant_schedule(1.0)
            return optax.join_schedules([constant_scheduler_min, cosine_decay_scheduler,
                                        constant_scheduler_max], boundaries=[epochs/4, 2*epochs/4])


# def load_true_results(n_particles: int):
#     import numpy as onp
#     d_ = f'Data_1D_GaussMixPot/true_rho_grid_Ne_{n_particles}.txt'
#     data = jnp.array(onp.loadtxt(d_))
#     return data


def training(batch_size: int = 256, epochs: int = 100, bool_load_params: bool = False, scheduler_type: str = 'ones'):

    mol_name = 'H'
    n_particles = 1
    coords = jnp.array([[0., 0., 0.]])
    z = jnp.array([[1.]])
    # atoms = ['H']
    mol = {'coords': coords, 'z': z}

    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model_rev = CNF(3, (264, 264,), bool_neg=False)
    model_fwd = CNF(3, (264, 264,), bool_neg=True)
    # model_rev = CNFRicky(1, 512, 512, bool_neg=False)
    # model_fwd = CNFRicky(1, 512, 512, bool_neg=True)
    test_inputs = lax.concatenate((jnp.ones((1, 3)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)

    @jax.jit
    def NODE_rev(params, batch): return neural_ode(
        params, batch, model_rev, -1., 0., 3)

    @jax.jit
    def NODE_fwd(params, batch): return neural_ode(
        params, batch, model_fwd, 0., 1., 3)

    t_functional = _kinetic('TF')
    v_functional = lambda *args: Nuclei_potential(*args)

    mean = jnp.zeros((3,))
    cov = 0.1 * jnp.ones((3,))
    prior_dist = MultivariateNormalDiag(mean, cov)
    # prior_dist = Laplace()

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
        zt0 = samples[:, :-1]
        zt, logp_zt = NODE_fwd(params, samples)
        # logp_x = prior_dist.log_prob(zt0) + logp_zt
        return jnp.exp(logp_zt)  # logp_x

    @jax.jit
    def rho_rev(params, x):
        zt = lax.concatenate((x, jnp.zeros((x.shape[0], 1))), 1)
        z0, logp_z0 = NODE_rev(params, zt)
        logp_x = prior_dist.log_prob(z0)[:, None] + logp_z0
        return jnp.exp(logp_x)  # logp_x

    @jax.jit
    def T(params, samples):
        zt, _ = NODE_fwd(params, samples)
        return zt

    @jax.jit
    def loss(params, u_samples, ci):
        u_samples, up_samples = u_samples[:batch_size,
                                          :], u_samples[batch_size:, :]
        t = (n_particles**(5/3))*t_functional(params, u_samples, rho)
        c_v = (n_particles**2) * \
            Hartree_potential(params, u_samples, up_samples, T)
        nuc_v = (n_particles)*v_functional(params, u_samples, T, mol)
        e_xc = (n_particles**(4/3))*Dirac_exchange(params, u_samples, rho)

        cusp = cusp_condition(params, rho_rev, mol)

        e = t + nuc_v + ci*c_v + e_xc + cusp
        return jnp.mean(e), {"t": jnp.mean(t),
                             "v": jnp.mean(nuc_v),
                             "h": jnp.mean(c_v),
                             "x": jnp.mean(e_xc),
                             "e": jnp.mean(t + nuc_v + c_v + e_xc),
                             "cusp": cusp}

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
            z0_and_logp_z0 = prior_dist.sample_and_log_prob(
                seed=key, sample_shape=batch_size)
            z0 = z0_and_logp_z0[0]
            logp_z0 = z0_and_logp_z0[1][:, None]
            samples0 = lax.concatenate((z0, logp_z0), 1)

            _, key = jrnd.split(key)
            z0_and_logp_z0 = prior_dist.sample_and_log_prob(
                seed=key, sample_shape=batch_size)
            z0 = z0_and_logp_z0[0]
            logp_z0 = z0_and_logp_z0[1][:, None]
            samples1 = lax.concatenate((z0, logp_z0), 1)

            yield lax.concatenate((samples0, samples1), 0)

    # @jax.jit
    # def _integral(params):
    #     return jnp.trapz(p_x.ravel(), zt.ravel(), jnp.abs(zt[1, 0]-zt[0, 0])), jnp.mean(logp_diff_z0)

    loss0 = jnp.inf
    df = pd.DataFrame()
    _, key = jrnd.split(key)
    gen_batches = batches_generator(key, batch_size)
    scheduler = get_scheduler(epochs, scheduler_type)
    # x_and_rho_true = load_true_results(
    # n_particles)  # read results from JCTC paper

    for i in range(epochs+1):
        ci = scheduler(i)
        batch = next(gen_batches)
        params, opt_state, loss_value = step(params, opt_state, batch, ci)
        loss_epoch, losses = loss_value

        # norm_integral, log_det_Jac = _integral(params)
        mean_energy = losses['e']
        r_ = {'epoch': i,
              'L': loss_epoch, 'E': mean_energy,
              'T': losses['t'], 'V': losses['v'], 'H': losses['h'], 'X': losses['x'], 'cusp': losses['cusp'],
              #   'I': norm_integral, 'ci': ci
              }
        df = pd.concat([df, pd.DataFrame(r_, index=[0])], ignore_index=True)
        df.to_csv(
            f"{CKPT_DIR}/training_trajectory_{mol_name}_{n_particles}.csv", index=False)

        if i % 5 == 0:
            _s = f"step {i}, L: {loss_epoch:.3f}, E:{mean_energy:.3f}\
            T: {losses['t']:.5f}, V: {losses['v']:.5f}, H: {losses['h']:.5f}, X: {losses['x']:.5f}, cusp: {losses['cusp']:.5f}"
            print(_s,
                  file=open(f"{CKPT_DIR}/loss_epochs_{mol_name}.txt", 'a'))

        if loss_epoch < loss0:
            params_opt, loss0 = params, loss_epoch
            # checkpointing model model
            # checkpoints.save_checkpoint(
            #     ckpt_dir=CKPT_DIR, target=params, step=0, overwrite=True)

        if i % 20 == 0 or i < 25:
            xt = jnp.linspace(-4.5, 4.5, 1000)
            yz = jnp.zeros((xt.shape[0], 2))
            zt = lax.concatenate((xt[:, None], yz), 1)
            zt_and_logp_zt = lax.concatenate(
                (zt, jnp.zeros_like(xt)[:, None]), 1)
            z0, logp_diff_z0 = NODE_rev(params_opt, zt_and_logp_zt)
            logp_x = prior_dist.log_prob(z0)[:, None] - logp_diff_z0
            rho_pred = logp_x  # jnp.exp(logp_x)
            def model_identity(params, x): return x
            def f_v(x): return v_functional(None, zt, model_identity, mol)
            v_pot = f_v(zt)

            # exact density n(r) = e−2r /π

            @jax.jit
            def exact_rho(x): return jnp.exp(-2 *
                                             jnp.linalg.norm(x, axis=1))/jnp.pi

            rho_exact = exact_rho(zt)

            plt.figure(0)
            plt.clf()
            plt.title(
                f'epoch {i}, L = {loss_epoch:.3f}, E = {mean_energy:.3f}, ci = {ci:.3f}')
            plt.plot(xt, rho_exact,
                     color='k', ls=":", label=r"$\hat{\rho}(x) = e^{-2r}/\pi$")
            plt.plot(xt, n_particles*jnp.exp(rho_pred),
                     color='tab:blue', label=r'$N_{e}\;\rho_{NF}(x)$')
            # plt.plot(xt, v_pot,
            #  ls='--', color='k', label=r'$V(x)$')
            plt.xlabel('x [Bhor]')
            plt.ylim(-10, 2.1)
            # plt.ylabel('Energy units')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{FIG_DIR}/rho_and_V_pot_{i}.png')

            # assert 0


def main():
    parser = argparse.ArgumentParser(description="Density fitting training")
    parser.add_argument("--epochs", type=int,
                        default=50, help="training epochs")
    parser.add_argument("--bs", type=int, default=12, help="batch size")
    parser.add_argument("--params", type=bool, default=False,
                        help="load pre-trained model")
    # parser.add_argument("--N", type=int, default=1, help="number of particles")
    # parser.add_argument("--sched", type=str, default='one',
    #                     help="Hartree integral scheduler")
    parser.add_argument("--sched", type=str, default='one',
                        help="Hartree integral scheduler")
    args = parser.parse_args()

    batch_size = args.bs
    epochs = args.epochs
    bool_params = args.params
    # n_particles = args.N
    scheduler_type = args.sched

    global CKPT_DIR
    global FIG_DIR
    CKPT_DIR = f"Results/H_OFDFT"
    FIG_DIR = f"{CKPT_DIR}/Figures"

    cwd = os.getcwd()
    rwd = os.path.join(cwd, CKPT_DIR)
    if not os.path.exists(rwd):
        os.makedirs(rwd)
    fwd = os.path.join(cwd, FIG_DIR)
    if not os.path.exists(fwd):
        os.makedirs(fwd)

    training(batch_size, epochs, bool_params, scheduler_type)


if __name__ == "__main__":
    main()
