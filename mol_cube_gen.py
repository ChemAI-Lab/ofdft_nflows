import argparse

import jax
import jax.numpy as jnp
from jax import lax
import jax.random as jrnd

from flax.training import checkpoints
from distrax import MultivariateNormalDiag

from ofdft_normflows.utils_cubegen import cube_generator
from ofdft_normflows.cn_flows import neural_ode
from ofdft_normflows.cn_flows import Gen_CNFSimpleMLP as CNF

BHOR = 1.8897259886  # 1AA to BHOR


def get_mol_info(mol_name: str):
    if mol_name.lower() == 'h2':
        mol_name = 'H2'
        Ne = 2
        coords = jnp.array(
            [[0., 0., -1.4008538753/2], [0., 0., 1.4008538753/2]])
        z = jnp.array([[1.], [1.]])
        atoms = ['H', 'H']
        mol = {'mol_name': mol_name, 'coords': coords,
               'z': z, 'atoms': atoms, 'Ne': Ne}

    elif mol_name.lower() == 'h2o':
        mol_name = 'H2O'
        Ne = 10
        # O	0.0000000	0.0000000	0.1189120
        # H	0.0000000	0.7612710	-0.4756480
        # H	0.0000000	-0.7612710	-0.4756480
        coords = jnp.array([[0.0,	0.0,	0.1189120],
                            [0.0,	0.7612710,	-0.4756480],
                            [0.0,	-0.7612710,	-0.4756480]])*BHOR
        z = jnp.array([[8.], [1.], [1.]])
        atoms = ['O', 'H', 'H']
        mol = {'mol_name': mol_name, 'coords': coords,
               'z': z, 'atoms': atoms, 'Ne': Ne}
    elif mol_name.lower() == 'h':
        mol_name = 'H'
        Ne = 1
        coords = jnp.array([[0., 0., 0.]])
        z = jnp.array([[1.]])
        atoms = ['H']
        mol = {'mol_name': mol_name, 'coords': coords,
               'z': z, 'atoms': atoms, 'Ne': Ne}

    return mol


def _plot(mol_name: str, rwd: str, nn_arch: any, nn_id: int):

    CKPT_DIR = rwd
    mol_info = get_mol_info(mol_name)
    Ne = mol_info['Ne']

    # init CNF model
    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model_rev = CNF(3, nn_arch, bool_neg=False)
    model_fwd = CNF(3, nn_arch, bool_neg=True)
    test_inputs = lax.concatenate((jnp.ones((1, 3)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)
    # load pretrained model
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=f"{CKPT_DIR}/checkpoints_all/", target=params, step=nn_id)
    params = restored_state

    # init prior distribution
    mean = jnp.zeros((3,))
    cov = jnp.ones((3,))
    prior_dist = MultivariateNormalDiag(mean, cov,)

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

    # generate cube files
    cube_array = cube_generator(
        rho_rev, mol_info, f'{mol_name}_CNF_{nn_id}', CKPT_DIR,
        nx=80, ny=80, nz=80)
    cube_array_file = f"{mol_name}_CNF_{nn_id}_data.npy"
    jnp.save(cube_array_file, cube_array, allow_pickle=True)

    # print(cube_array['mep'].shape, cube_array['rho'].shape)


'''
def main_h2():

    mol_name = 'H2'
    Ne = 2
    coords = jnp.array([[0., 0., -1.4008538753/2], [0., 0., 1.4008538753/2]])
    z = jnp.array([[1.], [1.]])
    atoms = ['H', 'H']
    mol = {'coords': coords, 'z': z}

    mol_inf = {"mol_name": mol_name, "Ne": Ne,
               "coords": coords, "atoms": atoms, "z": z
               }

    # load pre-trained model
    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model_rev = CNF(3, (512, 512,), bool_neg=False)
    test_inputs = lax.concatenate((jnp.ones((1, 3)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)

    CKPT_DIR = '/Users/ravh011/Documents/GitHub/ofdft_normflows/Results_SCRATCH/H2_TF-W_V_H_X_lr_3.0e-04_sched_MIX/'
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=f"{CKPT_DIR}/checkpoints_all/", target=params, step=2000)
    params = restored_state
    print(params)

    # prior-distribution
    mean = jnp.zeros((3,))
    cov = jnp.ones((3,))
    prior_dist = MultivariateNormalDiag(mean, cov)

    @jax.jit
    def NODE_rev(params, batch): return neural_ode(
        params, batch, model_rev, -1., 0., 3)

    @jax.jit
    def _rho_rev(params, x):
        zt = lax.concatenate((x, jnp.zeros((x.shape[0], 1))), 1)
        z0, logp_z0 = NODE_rev(params, zt)
        logp_x = prior_dist.log_prob(z0)[:, None] - logp_z0
        return jnp.exp(logp_x)  # logp_x

    @jax.jit
    def rho_rev(x): return _rho_rev(params, x)

    cube_array = cube_generator(
        rho_rev, mol_inf, 'H2_NF', CKPT_DIR,
        nx=20, ny=20, nz=10)
    print(cube_array['mep'].shape, cube_array['rho'].shape)


def main_h2o():
    import jax
    from jax import lax
    import jax.random as jrnd

    from cn_flows import neural_ode
    from cn_flows import Gen_CNFSimpleMLP as CNF

    mol_name = 'H2O'
    Ne = 10
    # O	0.0000000	0.0000000	0.1189120
    # H	0.0000000	0.7612710	-0.4756480
    # H	0.0000000	-0.7612710	-0.4756480
    coords = jnp.array([[0.0,	0.0,	0.1189120],
                        [0.0,	0.7612710,	-0.4756480],
                        [0.0,	-0.7612710,	-0.4756480]])*BHOR
    z = jnp.array([[8.], [1.], [1.]])
    atoms = ['O', 'H', 'H']
    mol = {'coords': coords, 'z': z}

    mol_inf = {"mol_name": mol_name, "Ne": Ne,
               "coords": coords, "atoms": atoms, "z": z
               }

    # load pre-trained model
    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model_rev = CNF(3, (512, 512,), bool_neg=False)
    test_inputs = lax.concatenate((jnp.ones((1, 3)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)

    # CS-MARIANA SERVER change home directory later
    CKPT_DIR = '/u/rvargas/ofdft_normflows/Results/H2O_TF-W_V_H_X_lr_3.0e-04_sched_MIX_rndW0/checkpoints_all/'
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=CKPT_DIR, target=params, step=2000)
    params = restored_state

    # prior-distribution
    mean = jnp.zeros((3,))
    cov = jnp.ones((3,))
    prior_dist = MultivariateNormalDiag(mean, cov)

    @jax.jit
    def NODE_rev(params, batch): return neural_ode(
        params, batch, model_rev, -1., 0., 3)

    @jax.jit
    def _rho_rev(params, x):
        zt = lax.concatenate((x, jnp.zeros((x.shape[0], 1))), 1)
        z0, logp_z0 = NODE_rev(params, zt)
        logp_x = prior_dist.log_prob(z0)[:, None] - logp_z0
        return jnp.exp(logp_x)  # logp_x

    @jax.jit
    def rho_rev(x): return _rho_rev(params, x)

    cube_array = cube_generator(
        rho_rev, mol_inf, f'{mol_name}_CNF_{nn_id}', CKPT_DIR,
        nx=20, ny=20, nz=10)
    print(cube_array['mep'].shape, cube_array['rho'].shape)

'''


def main():
    import os
    parser = argparse.ArgumentParser(description="CUBE GEN CALCULATIONS")
    parser.add_argument("--mol", type=str, default='H',
                        help="molecule name")
    parser.add_argument("--i", type=int, default=1, help="epoch number ")

    args = parser.parse_args()
    mol_name = args.mol
    nnid = args.i
    nn = (512, 512, 512,)

    cwd = '/u/rvargas/ofdft_normflows/Results/'
    mols_ = ['H', 'H2', 'H2O']
    rwd_ = ['H_TF-W_V_H_X_lr_3.0e-04', 'H2_TF-W_V_H_X_lr_3.0e-04_sched_MIX',
            'H2O_TF-W_V_H_X_lr_3.0e-04_sched_MIX_rndW0']
    nnid_ = [0, 1, 11, 101, 201, 1001, 2000]

    for mi in mols_[::-2]:
        for rwdi in rwd_[::-2]:
            for nnid in nnid_[::-1]:
                rwdi = os.path.join(cwd, rwdi)
                _plot(mol_name=mol_name, rwd=rwdi, nn_arch=nn, nn_id=nnid)
                assert 0


if __name__ == "__main__":
    main()
