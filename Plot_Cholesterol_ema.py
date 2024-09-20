import os
import json
import argparse
from functools import partial
from typing import Any, Union
# import pandas as pd0

import numpy as np
import jax.numpy as jnp 
from grid import GaussLegendre, BeckeRTransform, AtomGrid
from grid import UniformInteger, LinearInfiniteRTransform
from grid import UniformGrid
from pyscf import gto, dft
from matplotlib.ticker import ScalarFormatter

from grid.becke import BeckeWeights
from grid.molgrid import MolGrid
from scipy.integrate import nquad

import jax
from jax import lax, vmap, numpy as jnp
import jax.random as jrnd
from jax._src import prng

import chex
from flax.training import checkpoints
import optax
from optax import ema


import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
import os
colors = list(mcolors.TABLEAU_COLORS)

from ofdft_normflows.jax_ode import neural_ode, neural_ode_score
#from ofdft_normflows.flax_ode import neural_ode, neural_ode_score
from ofdft_normflows.equiv_flows import Gen_EqvFlow as GCNF
from ofdft_normflows.promolecular_distrax import ProMolecularDensity

import matplotlib.pyplot as plt

from ofdft_normflows.utils import one_hot_encode, coordinates
  
   
Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)



BOHR = 1.8897259886  # 1AA to BHOR

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
    

def training(mol_name: str,
            tw_kin: str = 'TF',
            v_pot: str = 'HGH',
            h_pot: str = 'MT',
            x_pot: str = 'dirac',
            c_pot: str = 'PW92_c_e', 
            batch_size: int = 256,
            epochs: int = 100,
            lr: float = 1E-5,
            nn: int = 2,
            bool_load_params: bool = False,
            scheduler_type: str = 'ones',
            optimizer: str = 'adam',
            prior_distribution: str = 'pro_mol'):
    
    CKPT_DIR_ALL = f"{CKPT_DIR}/checkpoints_all/"
   

    fig, ax = plt.subplots()
    i = 0
        
   
 
    columns = ["epoch", "E"]
    
    df_4layer = pd.read_csv(
         f"/Users/alexandre/Projects/accumulate_gradient_ofdft/ofdft_nflows/training_trajectory_C27H46O_pw92_c_e_ema.csv", usecols=columns)
    results_4layer = pd.read_csv(
         f"/Users/alexandre/Projects/accumulate_gradient_ofdft/ofdft_nflows/training_trajectory_C27H46O.csv", usecols=columns)
    
    
    
    #n_init = 0
    # n_init = 10000
    keys = results_4layer.columns
    epochs = results_4layer['epoch'].to_numpy()
   
    
    
    for i,k in enumerate(keys[1:]):
          
            r1_ema = df_4layer[k].to_numpy()
            ax.plot(epochs[:],r1_ema[:], c=colors[i],zorder = 0,alpha=0.5, lw=2.)
    
           
    
    

    FIG_DIR = f"Plots/Ema_{mol_name}{c_pot}/"    
    
    cwd = os.getcwd()
    fwd = os.path.join(cwd, FIG_DIR)
    if not os.path.exists(fwd):
            os.makedirs(fwd)
    # plt.ylim(-85,-60)
   

    # Plot the arrow
    # ax.arrow(start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1],
    #         head_width=1, head_length=1, fc='k', ec='k')
    # plt.annotate("", xy=end_point, xytext=start_point, 
    #              arrowprops=dict(arrowstyle="->", lw=2, color='k'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    # ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False, style="sci"))
    # ax.set_xticks(np.arange(10000, 20001, 10**3))
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    # ax.arrow(1000, -72, 5000, 0, head_width=10, head_length=0.1, fc='k', ec='k')
    # ax.set_xlim(9200,20300)
    # ax.set_ylim(-418,-370)
    plt.ylabel(r"$E[\rho_{{\cal M}}]$ [a.u.]", fontsize=20)
    plt.xlabel('Iterations', fontsize=20)
    # ax.text(0.01,.94,'a)',transform=ax.transAxes,fontsize=20)
    # ax.text(.8,.94,'C$_{6}$H$_{6}$',transform=ax.transAxes,fontsize=20)
    
    font = font_manager.FontProperties(size=12)
    
    plt.legend(prop=font,loc='lower left')
    plt.tight_layout()
   
    # plt.savefig(f'{FIG_DIR}/Energy_comparisson{mol_name}_{c_pot}',dpi=1200)
    plt.show()
            
    # assert 0 

    
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
    parser.add_argument("--c", type=str, default='PW92_C_E',
                        help="Correlation energy funcitonal")
    parser.add_argument("--N", type=int, default=2, 
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
    nn = (64, 64,)
  

    global CKPT_DIR
    global FIG_DIR
    global mol_name
    mol_name = 'H2'
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
