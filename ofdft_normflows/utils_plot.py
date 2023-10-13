import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import lax
import jax.random as jrnd
from jax._src import prng

from flax.training import checkpoints
from distrax import MultivariateNormalDiag

from ofdft_normflows.utils_cubegen import cube_generator
from ofdft_normflows.cn_flows import neural_ode
from ofdft_normflows.cn_flows import Gen_CNFSimpleMLP as CNF
from ofdft_normflows.dft_distrax import DFTDistribution

BHOR = 1.8897259886


def latex_format(input_string):
    return_string = input_string
    if input_string.isnumeric():
        return_string = f"{return_string}_{{{input_string}}}"
    elif input_string[-1].isdigit():
        # Find the last digit in the input string
        last_digit_index = len(input_string) - 1
        while last_digit_index >= 0 and input_string[last_digit_index].isdigit():
            last_digit_index -= 1

        # Insert "_{" and "}" after the last digit
        return_string = (
            input_string[:last_digit_index + 1] +
            f"_{{{input_string[last_digit_index + 1:]}}}"
        )

    return "$" + return_string + "$"


def plot_training_trajectory(rwd: str, mol_name: str, bool_save: bool = False):

    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS)

    mol_name_latex = latex_format(mol_name)

    n_init = 10  # remove some of the noisy initial points

    file = rwd + f'training_trajectory_{mol_name}_ema.csv'
    results_ema = pd.read_csv(file)

    file = rwd + f'training_trajectory_{mol_name}.csv'
    results = pd.read_csv(file)

    keys = results.columns
    epochs = results['epoch'].to_numpy()

    labels_keys = {'E': r"$E[\rho(x)]$", 'T': r"$T_{TFDW}[\rho(x)]$", 'V': r"$V_{e-N}[\rho(x)]$",
                   'H': r"$V_{H}[\rho(x)]$", 'X': r"$V_{X}[\rho(x)]$"}

    fig, ax = plt.subplots()
    for i, k in enumerate(keys[1:-1]):
        r_ = results[k].to_numpy()
        r_ema = results_ema[k].to_numpy()

        ax.plot(epochs[n_init:], r_[n_init:], ls='--', c=colors[i], lw=.2)
        ax.plot(epochs[n_init:], r_ema[n_init:], c=colors[i],
                zorder=1.5, lw=2., label=labels_keys[k])
        # ax.text(0.75*epochs[-1],r_ema[-1]*(1.1),labels_keys[k],)

    ax.text(0.075, 0.95,
            mol_name_latex, transform=ax.transAxes, va='top', fontsize=18)
    plt.ylabel('Funcitonal value [a.u.]', fontsize=15)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylim(top=1.3*np.max(results_ema['T'].to_numpy()[n_init:]))
    plt.legend(loc=1)
    plt.tight_layout()

    if bool_save:
        plt.savefig(f'fig_epochs_vs_values_{mol_name}.svg', transparent=True)
        plt.savefig(f'fig_epochs_vs_values_{mol_name}.png')
    else:
        plt.show()


def plot_rho_h2(epochs: list, CKPT_DIR: str, nn_arch: tuple = (512, 512, 512,)):
    mol_name = 'H2'
    Ne = 2
    coords = jnp.array([[0., 0., -1.4008538753/2], [0., 0., 1.4008538753/2]])
    z = jnp.array([[1.], [1.]])
    atoms = ['H', 'H']
    mol = {'coords': coords, 'z': z}

    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model_rev = CNF(3, nn_arch, bool_neg=False)
    model_fwd = CNF(3, nn_arch, bool_neg=True)
    test_inputs = lax.concatenate((jnp.ones((1, 3)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)

    mean = jnp.zeros((3,))
    cov = jnp.ones((3,))
    prior_dist = MultivariateNormalDiag(mean, cov)

    # CNF functions
    @jax.jit
    def NODE_rev(params, batch): return neural_ode(
        params, batch, model_rev, -1., 0., 3)

    @jax.jit
    def _rho_rev(params, x):
        zt = lax.concatenate((x, jnp.zeros((x.shape[0], 1))), 1)
        z0, logp_z0 = NODE_rev(params, zt)
        logp_x = prior_dist.log_prob(z0)[:, None] - logp_z0
        return Ne*jnp.exp(logp_x)  # logp_x

    @jax.jit
    def rho_rev(x): return _rho_rev(params, x)
    # load pretrained model

    def get_params(ei: int):
        if ei > 0:
            restored_state = checkpoints.restore_checkpoint(
                ckpt_dir=f"{CKPT_DIR}/checkpoints_all/", target=params, step=ei)
            return restored_state
        else:
            return params

    # 1D Figure
    # 1D Figure
    xt = jnp.linspace(-4.5, 4.5, 1000)
    yz = jnp.zeros((xt.shape[0], 2))
    zt = lax.concatenate((yz, xt[:, None]), 1)

    m = DFTDistribution(atoms, coords)
    rho_exact = m.prob(m, zt)

    rho_pred_epochs = {}
    for ei in epochs:
        params = get_params(ei)
        rho_pred_ei = rho_rev(params, zt)
        rho_pred_epochs[ei] = rho_pred_ei

    FIG_DIR = f"{CKPT_DIR}/Figures"
    fig, ax = plt.subplots()
    plt.plot(xt, rho_exact,
             color='k', ls=":", label=r"$\hat{\rho}_{{\cal M}}$")
    plt.plot(xt, rho_pred_epochs[0],
             color='k', ls="--", label=r"$N_{e}\rho_{0}$")
    for i, k in enumerate(rho_pred_epochs):
        if k != 0:
            rho_ei = rho_pred_epochs[k]
            plt.plot(xt, rho_pred_epochs[ei],
                     color='tab:blue', ls="--", label=r"$\rho_{{\cal M}}(%s)$" % ei)
    plt.xlabel('X [Bhor]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/epoch_rho_z_epochs.svg', transparent=True)
    plt.savefig(f'{FIG_DIR}/epoch_rho_z_epochs.png')


if __name__ == "__main__":
    rwd = '/Users/ravh011/Documents/GitHub/ofdft_normflows/Results_SCRATCH/H2_TF-W_V_H_X_lr_3.0e-04_sched_MIX/'

    mol_name = 'H2'
    plot_training_trajectory(rwd, mol_name, True)

    rwd = '/Users/ravh011/Documents/GitHub/ofdft_normflows/Results_SCRATCH/H_TF-W_V_H_X_lr_3.0e-04/'

    mol_name = 'H'
    plot_training_trajectory(rwd, mol_name, True)