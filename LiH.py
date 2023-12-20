import os
import argparse
from typing import Any, Union
import pandas as pd

import jax
from jax import lax, numpy as jnp
import jax.random as jrnd
from jax._src import prng

import chex
from flax.training import checkpoints
import optax
from optax import ema
from distrax import MultivariateNormalDiag


from ofdft_normflows.functionals import _kinetic,_exchange, _hartree, _nuclear
from ofdft_normflows.jax_ode import neural_ode, neural_ode_score
from ofdft_normflows.cn_flows import Gen_CNFSimpleMLP as CNF
from ofdft_normflows.utils import get_scheduler, batches_generator_w_score
from ofdft_normflows.figures import f

import matplotlib.pyplot as plt

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]

jax.config.update("jax_enable_x64", True)


@chex.dataclass
class F_values:
    energy: chex.ArrayDevice
    kin: chex.ArrayDevice
    vnuc: chex.ArrayDevice
    hart: chex.ArrayDevice
    x: chex.ArrayDevice
  
def training(Ne: int = 4, 
             batch_size: int = 256, 
             epochs: int = 2000, 
             bool_load_params: bool = False, 
             scheduler_type: str = 'mix', 
             R:float = 10., 
             Z_alpha:int = 3, 
             Z_beta:int = 1):
    
    CKPT_DIR_ALL = f"{CKPT_DIR}/checkpoints_all/"

    mol_name = 'LiH'
  
    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model_rev = CNF(1, (512, 512, 512, ), bool_neg=False)
    model_fwd = CNF(1, (512, 512, 512, ), bool_neg=True)
    test_inputs = lax.concatenate((jnp.ones((1, 1)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)
    params_init = params

    @jax.jit
    def NODE_rev(params, batch): return neural_ode(
        params, batch, model_rev, -1., 0., 1)


    @jax.jit
    def NODE_fwd_score(params, batch): return neural_ode_score(
        params, batch, model_fwd, 0., 1., 1)
    
    prior_dist = MultivariateNormalDiag(jnp.zeros(1), 1.*jnp.ones(1))
    
    lr_sched = get_scheduler(epochs,scheduler_type,3E-4)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.rmsprop(learning_rate=lr_sched)
            )
    opt_state = optimizer.init(params)
    energies_ema = ema(decay=0.99)
    energies_state = energies_ema.init(
        F_values(energy=jnp.array(0.), kin=jnp.array(0.), vnuc=jnp.array(0.), hart=jnp.array(0.), x=jnp.array(0.)))

    @jax.jit
    def rho_x_score(params, samples):
        zt, logp_zt, score_zt = NODE_fwd_score(params, samples)
        return jnp.exp(logp_zt), zt, score_zt

    @jax.jit
    def rho_rev(params, x):
        zt = lax.concatenate((x, jnp.zeros((x.shape[0], 1))), 1)
        z0, logp_z0 = NODE_rev(params, zt)
        logp_x = prior_dist.log_prob(z0)[:, None] - logp_z0
        return jnp.exp(logp_x)  
    
    @jax.jit
    def _integral(params,x):
        p_x = rho_rev(params,x)
        return jnp.trapz(p_x.ravel(), x.ravel(), jnp.abs(zt[1, 0]-zt[0, 0])), p_x
    

    thomas_fermi_1d = _kinetic('tf1d')
    weizsacker_1d = _kinetic('w1d')
    #x_functional = _exchange('dirac')
    c_functional = _exchange('c')
    x_functional = _exchange('x')
    vnuc = _nuclear('attraction')
    hart = _hartree('soft_coulomb')

    @jax.jit
    def loss(params, u_samples):
        den_all, x_all, score_all = rho_x_score(params, u_samples)

        den, denp = den_all[:batch_size], den_all[batch_size:]
        x, xp = x_all[:batch_size], x_all[batch_size:]
        score, scorep = score_all[:batch_size], score_all[batch_size:]
        tf = thomas_fermi_1d(den,score,Ne)
        tw = weizsacker_1d(den,score,Ne)
        c_v = hart(x,xp,Ne)
        c_a = vnuc(x, R, Z_alpha, Z_beta,Ne)
        # x = x_functional(den,Ne)
        # c = c_functional(den,Ne)
        LDA = x_functional(den,Ne) +  c_functional(den,Ne)
        e = tf + tw + c_a + c_v + LDA
        energy = jnp.mean(e)
        f_values = F_values(energy=energy,
                            kin=jnp.mean(tf+tw),
                            vnuc=jnp.mean(c_a),
                            hart=jnp.mean(c_v),
                            x= jnp.mean(LDA),
                            )
        return energy, f_values

    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(
            loss, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    df = pd.DataFrame()
    df_ema = pd.DataFrame()
    _, key = jrnd.split(key)
    gen_batches = batches_generator_w_score(key, batch_size, prior_dist)

    for i in range(epochs+1):
        batch = next(gen_batches)
        params, opt_state, loss_value = step(params, opt_state, batch)  # , ci
        loss_epoch, losses = loss_value

        energies_i_ema, energies_state = energies_ema.update(
            losses, energies_state)
        ei_ema = energies_i_ema.energy
        zt = jnp.linspace(-20., 20., num=4096)[:, jnp.newaxis]
        norm_val, rho_pred = _integral(params,zt)
        
        r_ = {'epoch': i,
              'E': loss_epoch,
              'T': losses.kin, 'V': losses.vnuc, 'H': losses.hart, 'X':losses.x,
              'I': norm_val,
              }

        df = pd.concat([df, pd.DataFrame(r_, index=[0])], ignore_index=True)
        df.to_csv(
            f"{CKPT_DIR}/training_trajectory_{mol_name}.csv", index=False)

        r_ema = {'epoch': i,
                 'E': energies_i_ema.energy,
                 'T': energies_i_ema.kin, 'V': energies_i_ema.vnuc, 'H': energies_i_ema.hart, 'X': energies_i_ema.x,
                 'I': norm_val,
                 }
        df_ema = pd.concat(
            [df_ema, pd.DataFrame(r_ema, index=[0])], ignore_index=True)
        df_ema.to_csv(
            f"{CKPT_DIR}/training_trajectory_{mol_name}_ema.csv", index=False)

        checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR_ALL, target=params, step=i, keep_every_n_steps=10, overwrite=True)

        #Saving plottings 
        if i % 10 == 0:
            plt.clf()
            fig, ax = plt.subplots()
            ax.text(0.075, 0.92,
                    f'({i}):  E = {ei_ema:.3f}', transform=ax.transAxes, va='top', fontsize=10)
            ax.plot(zt, Ne*rho_pred,
                    color='tab:blue', label=r'$N_{e}\;\rho_{NF}(x)$'f',R={R}')

            plt.xlabel('X [Bhor]')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{FIG_DIR}/epoch_rho_z_{i}.svg', transparent=True)
            plt.savefig(f'{FIG_DIR}/epoch_rho_z_{i}.png')

def main():
    parser = argparse.ArgumentParser(description="Density fitting training")
    parser.add_argument("--epochs", type=int,
                        default=2000, help="training epochs")
    parser.add_argument("--bs", type=int, default=512, help="batch size")
    parser.add_argument("--params", type=bool, default=False,
                        help="load pre-trained model")
    parser.add_argument("--N", type=int, default=2, help="number of particles")
    parser.add_argument("--sched", type=str, default='mix',
                        help="Hartree integral scheduler")
    parser.add_argument("--R", type=float, default=10., help="R parameter")
    parser.add_argument("--Z_alpha", type=int, default=3,help="Nuclei of charges")
    parser.add_argument("--Z_beta", type=int, default=1,help="Nucleis of charges")
    args = parser.parse_args()

    batch_size = args.bs
    epochs = args.epochs
    bool_params = args.params
    Ne = args.N
    R = args.R
    scheduler_type = args.sched
    Z_alpha = args.Z_alpha 
    Z_beta = args.Z_beta

    global CKPT_DIR
    global FIG_DIR
    CKPT_DIR = f"LiH/GP_pot_Ne_{Ne}_Hsched_{scheduler_type}/R{R}"
    FIG_DIR = f"{CKPT_DIR}/Figures"

    cwd = os.getcwd()
    rwd = os.path.join(cwd, CKPT_DIR)
    if not os.path.exists(rwd):
        os.makedirs(rwd)
    fwd = os.path.join(cwd, FIG_DIR)
    if not os.path.exists(fwd):
        os.makedirs(fwd)

    training(Ne, batch_size, epochs, bool_params, scheduler_type, R, Z_alpha, Z_beta)

if __name__ == "__main__":
    main()
    
