import jax.numpy as jnp 
import pandas as pd
import matplotlib.pyplot as plt

from flax.training import checkpoints
from jax import jit

@jit
def f(params,epochs,loss_epoch,losses,norm_val,energies_i_ema,ei_ema,zt,rho_pred):
    Ne = 4
    R = 10.
    mol_name = 'LiH'
    CKPT_DIR = f"LiH/GP_pot_Ne_{Ne}/R{R}"
    FIG_DIR = f"{CKPT_DIR}/Figures"
    CKPT_DIR_ALL = f"{CKPT_DIR}/checkpoints_all/"
    for i in range(epochs+1):
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