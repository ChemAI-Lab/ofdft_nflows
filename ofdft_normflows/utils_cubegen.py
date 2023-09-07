import numpy as onp
import jax
import jax.numpy as jnp
from jax import lax
from flax.training import checkpoints

import pyscf
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.dft import r_numint
from pyscf.data.nist import BOHR
from pyscf.tools.cubegen import Cube
from pyscf import scf
from pyscf.tools import cubegen

import distrax
from distrax import MultivariateNormalDiag

BHOR = 1.8897259886  # 1AA to BHOR


def get_molecule(atoms, geometry):
    m_ = ""
    for a, xi in zip(atoms, geometry):
        # print(a, xi)
        mi_ = f'{a} '
        mxi_ = ""
        for xii in xi:
            mxi_ += str(xii) + " "
        mi_ += mxi_ + '\n'
        m_ += mi_
    return m_


def density(mol, outfile='caca.cube', nx=80, ny=80, nz=80, resolution=None,
            margin=5.):

    from pyscf.pbc.gto import Cell
    cc = Cube(mol, nx, ny, nz, resolution, margin)
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    print(coords.shape, ngrids)

    prior_dist = MultivariateNormalDiag(jnp.zeros(3), jnp.ones(3))
    rho = jnp.exp(prior_dist.log_prob(jnp.array(coords)))
    print(rho.shape)
    rho = rho.reshape(cc.nx, cc.ny, cc.nz)

    # Write out density to the .cube file
    cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')
    return rho


def _density(f_density: callable, mol_pyscf: any, outfile: str = 'caca.cube', save_cube_file: bool = True,
             nx: int = 80, ny: int = 80, nz: int = 80, resolution: any = None, margin: float = 5.):

    from pyscf.pbc.gto import Cell
    cc = Cube(mol_pyscf, nx, ny, nz, resolution, margin)
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    grid = jnp.array(coords)

    if grid.shape[1] != 3:
        assert 0

    # rho = f_density(grid)
    total_data_size = grid.shape[0]
    batch_size = 1024

    rho_ = jnp.zeros((1, 1))
    for i in range(0, total_data_size, batch_size):
        start_idx = i
        end_idx = jax.lax.min(i + batch_size, total_data_size)

        # Get the current batch of input data
        input_data = jnp.arange(start_idx, end_idx, dtype=jnp.int32)
        grid_batch = grid[input_data]
        rho_batch = f_density(jax.device_put(grid_batch))
        rho_ = jax.lax.concatenate((rho_, jax.device_get(rho_batch)), 0)

    rho = rho_[1:]
    print(grid.shape, rho.shape)
    rho = rho.reshape(cc.nx, cc.ny, cc.nz)

    if save_cube_file:
        # Write out density to the .cube file
        cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')
    return rho, grid


def _density_and_mep(f_density: callable, mol_inf: any, mol_pyscf: any, outfile_head: str = 'caca', save_cube_file: bool = True,
                     nx: int = 80, ny: int = 80, nz: int = 80, resolution: any = None, margin: float = 5.):

    # becke grid
    mf_hf = dft.RKS(mol_pyscf)
    mf_hf.kernel()
    # mf_hf.grids.level = 3
    int_coords = jax.device_put(jnp.array(mf_hf.grids.coords))
    int_weights = jax.device_put(jnp.array(mf_hf.grids.weights))

    from functionals import Nuclei_potential

    @jax.jit
    def f_vnuc(x):
        return -1.*Nuclei_potential(x=x, Ne=1., mol_info=mol_inf)  # positive

    @jax.jit
    def integral(x, grid, w):
        rho_val = f_density(grid)
        z = lax.expand_dims(1./jnp.linalg.norm(x-grid, axis=1), (1,))
        integrand = rho_val * z
        return jnp.vdot(w, integrand)
    v_integral = jax.vmap(integral, in_axes=(0, None, None))

    from pyscf.pbc.gto import Cell
    cc = Cube(mol_pyscf, nx, ny, nz, resolution, margin)
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    grid = jnp.array(coords)

    if grid.shape[1] != 3:
        assert 0

    # rho = f_density(grid)
    total_data_size = grid.shape[0]
    batch_size = 1024

    rho_ = jnp.zeros((1, 1))
    mep_ = jnp.zeros((1, 1))
    vnuc_ = jnp.zeros((1, 1))
    for i in range(0, total_data_size, batch_size):
        start_idx = i
        end_idx = jax.lax.min(i + batch_size, total_data_size)

        # Get the current batch of input data
        input_data = jnp.arange(start_idx, end_idx, dtype=jnp.int32)
        grid_batch = grid[input_data]
        grid_batch = jax.device_put(grid_batch)

        # density
        rho_batch = f_density(grid_batch)
        rho_ = jax.lax.concatenate((rho_, jax.device_get(rho_batch)), 0)

        # nuclei potential sum_A Z_a / |r - Ra|
        vnuc_batch = f_vnuc(grid_batch)
        vnuc_ = jax.lax.concatenate((vnuc_, jax.device_get(vnuc_batch)), 0)

        # add integration
        integ_batch = v_integral(grid_batch, int_coords, int_weights)
        mep_batch = vnuc_batch - integ_batch[:, None]
        mep_ = jax.lax.concatenate((mep_, jax.device_get(mep_batch)), 0)

    rho, vnuc, mep = rho_[1:], vnuc_[1:], mep_[1:]
    rho = rho.reshape(cc.nx, cc.ny, cc.nz)
    vnuc = vnuc.reshape(cc.nx, cc.ny, cc.nz)
    mep = mep.reshape(cc.nx, cc.ny, cc.nz)
    if save_cube_file:
        # Write out density to the .cube file
        outfile = f"{outfile_head}_rho.cube"
        cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')
        outfile = f"{outfile_head}_vnuc.cube"
        cc.write(vnuc, outfile, comment='Nuclei potential in real space (e/Bohr)')
        outfile = f"{outfile_head}_mep.cube"
        cc.write(vnuc, outfile,
                 comment='Molecular electrostatic potential in real space')
    return {'rho': rho,
            'grid': grid,
            'Vnuc': vnuc,
            'mep': mep,
            }


def cube_generator(rho_rev: callable, mol_info: any,
                   outfile: str = 'molecule', save_cube_file: bool = True,
                   nx: int = 80, ny: int = 80, nz: int = 80, resolution: any = None, margin: float = 5.):

    mol_name = mol_info['mol_name']  # 'H2'
    Ne = mol_info['Ne']  # 2
    # jnp.array([[0., 0., -1.4008538753/2], [0., 0., 1.4008538753/2]])
    coords = mol_info['coords']
    z = mol_info['z']  # jnp.array([[1.], [1.]])
    atoms = mol_info['atoms']  # ['H', 'H']
    mol_ = {'coords': coords, 'z': z}

    # pyscf
    mol = gto.M(atom=get_molecule(atoms, coords), basis='sto-3g',
                unit='B')  # , symmetry = True)

    cube_arrays = _density_and_mep(f_density=rho_rev, mol_inf=mol_, mol_pyscf=mol,
                                   outfile_head=outfile,
                                   save_cube_file=save_cube_file,
                                   nx=nx, ny=ny, nz=nz,
                                   resolution=resolution, margin=margin)
    return cube_arrays
    # cube_density, cube_grid = _density(f_density=rho_rev, mol_pyscf=mol, outfile=outfile,
    #                                    save_cube_file=save_cube_file,
    #                                    nx=nx, ny=ny, nz=nz,
    #                                    resolution=resolution, margin=margin)
    # return cube_density, cube_grid


def main_h2():
    import jax
    from jax import lax
    import jax.random as jrnd

    from cn_flows import neural_ode
    from cn_flows import Gen_CNFSimpleMLP as CNF

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

    CKPT_DIR = '/Users/ravh011/Documents/GitHub/ofdft_normflows/Results_SCRATCH/H2_TF-W_V_H_X_lr_3.0e-04_sched_MIX/checkpoints_all/'
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=CKPT_DIR, target=params, step=2000)
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
        rho_rev, mol_inf, 'H2_NF', nx=20, ny=20, nz=10)
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

    cube_rho, cube_grid = cube_generator(
        rho_rev, mol_inf, f'{mol_name}_NF.cube', nx=10, ny=10, nz=10)


if __name__ == "__main__":
    main_h2()
