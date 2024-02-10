import os
import json
import argparse
from functools import partial
from typing import Any, Union
import pandas as pd

import jax
from jax import lax, vmap, numpy as jnp
import jax.random as jrnd
from jax._src import prng

import chex
from flax.training import checkpoints
import optax
from optax import ema
from distrax import MultivariateNormalDiag, Laplace

from ofdft_normflows.functionals import _kinetic, _nuclear, _hartree,_exchange_correlation
from ofdft_normflows.dft_distrax import DFTDistribution,MixGaussian
from ofdft_normflows.jax_ode import neural_ode, neural_ode_score
from ofdft_normflows.cn_flows import Gen_CNFSimpleMLP as CNF
from ofdft_normflows.utils import get_scheduler,batches_generator_w_score_mix_gaussian

import matplotlib.pyplot as plt

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]

jax.config.update("jax_enable_x64", True)

BHOR = 1.8897259886  # 1AA to BHOR


@ partial(jax.jit,  static_argnums=(2,))
def compute_integral(params: Any, grid_array: Any, rho: Any, Ne: int, bs: int):
    grid_coords, grid_weights = grid_array
    rho_val = Ne*rho(params, grid_coords)
    return jnp.vdot(grid_weights, rho_val)


@chex.dataclass
class F_values:
    energy: chex.ArrayDevice
    kin: chex.ArrayDevice
    vnuc: chex.ArrayDevice
    hart: chex.ArrayDevice
    xc: chex.ArrayDevice
    

def training(tw_kin: str = 'TF',
            v_pot: str = 'HGH',
            h_pot: str = 'MT',
            x_pot: str = 'dirac',
            c_pot: str = 'vwn',
            Ne: int = 10,
            batch_size: int = 256,
            epochs: int = 100,
            lr: float = 1E-5,
            nn_arch: tuple = (512, 512,),
            bool_load_params: bool = False,
            scheduler_type: str = 'ones'):

    CKPT_DIR_ALL = f"{CKPT_DIR}/checkpoints_all/"
    
    # O	0.0000000	0.0000000	0.1189120
    # H	0.0000000	0.7612710	-0.4756480
    # H	0.0000000	-0.7612710	-0.4756480
    coords = jnp.array([[0.0,	0.0,	0.1189120],
                        [0.0,	0.7612710,	-0.4756480],
                        [0.0,	-0.7612710,	-0.4756480]])*BHOR
    z = jnp.array([[8.], [1.], [1.]])
    atoms = ['O', 'H', 'H']
    mol = {'coords': coords, 'z': z}

    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model_rev = CNF(3, nn_arch, bool_neg=False)
    model_fwd = CNF(3, nn_arch, bool_neg=True)
    test_inputs = lax.concatenate((jnp.ones((1, 3)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)

    @jax.jit
    def NODE_rev(params, batch): return neural_ode(
        params, batch, model_rev, -1., 0., 3)

    @jax.jit
    def NODE_fwd(params, batch): return neural_ode(
        params, batch, model_fwd, 0., 1., 3)

    @jax.jit
    def NODE_fwd_score(params, batch): return neural_ode_score(
        params, batch, model_fwd, 0., 1., 3)

    mean = jnp.array([[[-2.,0.,0.]],[[2.,0.,0.]],[[2.,0.,0.]]])
    c = jnp.array([[[1.,1.,1.]],[[1.,1.,1.]],[[1.,1.,1.]]])

    mean = mean[:,:,:]
    c = c[:,:,:]

    probs = jnp.array([.8,.1,.1])
    mu = coords[:,None]
    cov = c
    
    prior_dist = MixGaussian(mu,cov,probs)

    m = DFTDistribution(atoms=atoms, geometry=coords)
    normalization_array = (m.coords, m.weights)

    # optimizer = optax.adam(learning_rate=1E-3)
    lr_sched = get_scheduler(epochs, scheduler_type, lr)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.rmsprop(learning_rate=lr_sched)
        # optax.adam(learning_rate=lr_sched),
    )
    opt_state = optimizer.init(params)
    energies_ema = ema(decay=0.99)
    energies_state = energies_ema.init(
       F_values(energy=jnp.array(0.), kin=jnp.array(0.), vnuc=jnp.array(0.), hart=jnp.array(0.), xc=jnp.array(0.)))

    # load prev parameters
    if bool_load_params:
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=CKPT_DIR, target=params, step=0)
        params = restored_state

    @jax.jit
    def rho_x(params, samples):
        zt, logp_zt = NODE_fwd(params, samples)
        return jnp.exp(logp_zt), zt, None

    @jax.jit
    def rho_x_score(params, samples):
        zt, logp_zt, score_zt = NODE_fwd_score(params, samples)
        return jnp.exp(logp_zt), zt, score_zt

    @jax.jit
    def rho_rev(params, x):
        zt = lax.concatenate((x, jnp.zeros((x.shape[0], 1))), 1)
        z0, logp_z0 = NODE_rev(params, zt)
        logp_x = prior_dist.log_prob(z0) - logp_z0
        return jnp.exp(logp_x)  # logp_x

    @jax.jit
    def T(params, samples):
        zt, _ = NODE_fwd(params, samples)
        return zt

    t_functional = _kinetic(tw_kin)
    v_functional = _nuclear(v_pot)
    vh_functional = _hartree(h_pot)
    x_functional = _exchange_correlation(x_pot)
    c_functional = _exchange_correlation(c_pot)

    @jax.jit
    def loss(params, u_samples):
        # den_all, x_all = rho_and_x(params, u_samples)
        den_all, x_all, score_all = rho_x_score(params, u_samples)

        den, denp = den_all[:batch_size], den_all[batch_size:]
        x, xp = x_all[:batch_size], x_all[batch_size:]
        score, scorep = score_all[:batch_size], score_all[batch_size:]

        e_t = t_functional(den, score, Ne)
        e_h = vh_functional(x, xp, Ne)
        e_nuc_v = v_functional(x, Ne, mol)
        e_x = x_functional(den,score,Ne)
        e_c = c_functional(den,Ne)
       

        e = e_t + e_nuc_v + e_h + e_x + e_c 
        energy = jnp.mean(e)
        f_values = F_values(energy=energy,
                            kin=jnp.mean(e_t),
                            vnuc=jnp.mean(e_nuc_v),
                            hart=jnp.mean(e_h),
                            xc=jnp.mean(e_x + e_c))
        
        return energy, f_values

    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(
            loss, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    _, key = jrnd.split(key)
    gen_batches = batches_generator_w_score_mix_gaussian(key, batch_size, prior_dist)

    df = pd.DataFrame()
    df_ema = pd.DataFrame()
    for i in range(epochs+1):
        batch = next(gen_batches)
        params, opt_state, loss_value = step(params, opt_state, batch)  # , ci
        loss_epoch, losses = loss_value

        # functionals values ema
        energies_i_ema, energies_state = energies_ema.update(
            losses, energies_state)
        ei_ema = energies_i_ema.energy
        norm_val = compute_integral(
            params, normalization_array, rho_rev, Ne, 0)

        r_ = {'epoch': i,
              'E': loss_epoch,
              'T': losses.kin, 'V': losses.vnuc, 'H': losses.hart, 'XC': losses.xc,
              'I': norm_val,
              }

        df = pd.concat([df, pd.DataFrame(r_, index=[0])], ignore_index=True)
        df.to_csv(
            f"{CKPT_DIR}/training_trajectory_{mol_name}.csv", index=False)

        r_ema = {'epoch': i,
                 'E': energies_i_ema.energy,
                 'T': energies_i_ema.kin, 'V': energies_i_ema.vnuc, 'H': energies_i_ema.hart, 'XC': energies_i_ema.xc,
                 'I': norm_val,
                 }
        df_ema = pd.concat(
            [df_ema, pd.DataFrame(r_ema, index=[0])], ignore_index=True)
        df_ema.to_csv(
            f"{CKPT_DIR}/training_trajectory_{mol_name}_ema.csv", index=False)

        # save models
        checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR_ALL, target=params, step=i, keep_every_n_steps=10)


def main():
    parser = argparse.ArgumentParser(description="Density fitting training")
    parser.add_argument("--epochs", type=int,default=1000, 
                        help="training epochs")
    parser.add_argument("--bs", type=int, default=512,
                         help="batch size")
    parser.add_argument("--params", type=bool, default=False,
                        help="load pre-trained model")
    parser.add_argument("--lr", type=float, default=3E-4,
                        help="learning rate")
    parser.add_argument("--kin", type=str, default='tf-w',
                        help="Kinetic energy funcitonal")
    parser.add_argument("--nuc", type=str, default='nuclei_potential',
                        help="Nuclear Potential energy funcitonal")
    parser.add_argument("--hart", type=str, default='hartree',
                        help="Hartree energy funcitonal")
    parser.add_argument("--x", type=str, default='dirac_b88_x_e',
                        help="Exchange energy funcitonal")
    parser.add_argument("--c", type=str, default='vwn_c_e',
                        help="Correlation energy funcitonal")
    parser.add_argument("--N", type=int, default=10, 
                        help="number of particles")
    parser.add_argument("--sched", type=str, default='mix',
                        help="Hartree integral scheduler")
    args = parser.parse_args()

    Ne = args.N
    batch_size = args.bs
    epochs = args.epochs
    bool_params = args.params
    lr = args.lr
    sched_type = args.sched

    kin = args.kin
    v_pot = args.nuc
    h_pot = args.hart
    x_pot = args.x
    c_pot = args.c
    nn = (512, 512, 512,)
  

    global CKPT_DIR
    global FIG_DIR
    global mol_name
    mol_name = 'H2O'
    CKPT_DIR = f"Results/{mol_name}_{kin.upper()}_{v_pot.upper()}_{h_pot.upper()}_{x_pot.upper()}_{c_pot.upper()}_lr_{lr:.1e}"
    if sched_type.lower() != 'c' or sched_type.lower() != 'const':
        CKPT_DIR = CKPT_DIR + f"_sched_{sched_type.upper()}"
    FIG_DIR = f"{CKPT_DIR}/Figures"

    cwd = os.getcwd()
    rwd = os.path.join(cwd, CKPT_DIR)
    if not os.path.exists(rwd):
        os.makedirs(rwd)
    fwd = os.path.join(cwd, FIG_DIR)
    if not os.path.exists(fwd):
        os.makedirs(fwd)

    job_params ={'Ne':Ne,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'kin': kin,
                'v_nuc': v_pot,
                'h_pot': h_pot,
                'x_pot': x_pot,
                'c_pot': c_pot,
                'nn': tuple(nn),
                'sched': sched_type,
                  }
    with open(f"{CKPT_DIR}/job_params.json", "w") as outfile:
        json.dump(job_params, outfile, indent=4)


    training(kin, v_pot, h_pot, x_pot,c_pot,Ne, batch_size,
             epochs, lr, nn, bool_params, sched_type)


if __name__ == "__main__":
    main()
