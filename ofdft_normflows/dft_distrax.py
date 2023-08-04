
from functools import partial
from typing import Tuple, Optional, Union, Any
import numpy as onp

import jax
import jax.numpy as jnp

import pyscf
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.dft import r_numint
from pyscf.data.nist import BOHR

import distrax
from distrax._src.distributions import distribution
from distrax._src.distributions.distribution import Array

Any = Any
Array = jax.Array
PRNGKey = jax.random.PRNGKey
EventT = distribution.EventT
Dtype = Any

# jax.config.update('jax_disable_jit', True)


class DFTDistribution(distrax.Distribution):

    def __init__(self, atoms: Any, geometry: Any, basis_set: str = 'sto-3g', exc: str = 'b3lyp', dtype_: Dtype = jnp.float32):

        self.atoms = atoms
        self.geometry = geometry
        self.basis_set = basis_set
        self.exc = exc
        self.dtype_ = dtype_

        self._grid_level = 2  # change this for larger molecules
        self.mol = self._mol()
        self.Ne = self.mol.tot_electrons()
        self.dft, self.rdm1 = self._dft()

        self.coords = jnp.array(self.dft.grids.coords)
        self.weights = jnp.array(self.dft.grids.weights)

    def get_molecule(self):
        m_ = ""
        for a, xi in zip(self.atoms, self.geometry):
            print(a, xi)
            mi_ = f'{a} '
            mxi_ = ""
            for xii in xi:
                mxi_ += str(xii) + " "
            mi_ += mxi_ + '\n'
            m_ += mi_
        return m_

    def _mol(self):
        atoms = self.get_molecule()
        mol = gto.M(atom=atoms, basis=self.basis_set,
                    unit='B')  # , symmetry = True)
        return mol

    def _dft(self):
        mf_hf = dft.RKS(self.mol)
        mf_hf.xc = self.exc  # default
        mf_hf = mf_hf.newton()
        mf_hf.kernel()
        mf_hf.grids.level = self._grid_level
        mf_hf.grids.build(with_non0tab=True)
        dm = mf_hf.make_rdm1()
        return mf_hf, dm

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def prob(self, value):
        coords = onp.array(value)
        ao_value = numint.eval_ao(self.mol, coords, deriv=1)
        rho_and_grho = numint.eval_rho(
            self.mol, ao_value, self.rdm1, xctype='GGA')
        rho = jnp.asarray(rho_and_grho[0], dtype=self.dtype_)  # /self.Ne
        return rho[:, None]  # includes batch dimension

    def prob_fwd(self, value):
        coords = value
        ao_value = numint.eval_ao(self.mol, coords, deriv=1)
        rho_and_grho = numint.eval_rho(
            self.mol, ao_value, self.rdm1, xctype='GGA')
        rho = jnp.array(rho_and_grho[0], dtype=self.dtype_)/self.Ne
        drho_dx = jnp.array(
            rho_and_grho[1:, :].T, dtype=self.dtype_)/self.Ne
        return rho[:, None], (drho_dx)

    def prob_bwd(self, res, g):
        drho_dx = res
        return (drho_dx*g,)
        # return (jnp.matmul(g.T, drho_dx),)

    prob.defvjp(prob_fwd, prob_bwd)

    def log_prob(self, value: Any) -> Array:
        pass

    def _sample_n(self, key, n):
        pass

    def event_shape(self):
        pass

    def _sample_n_and_log_prob(self, key, n):
        pass


if __name__ == '__main__':
    atoms = ['H', 'H']
    geom = jnp.array([[0., 0., 0.], [0.76, 0., 0.]])

    m = DFTDistribution(atoms, geom)

    x = jnp.ones((10, 3))
    # print(m.prob(m,geom))
    rho = m.prob(m, x)
    print(rho)

    xx = jax.jacrev(m.prob, argnums=(1,))(m, x)

    print(xx[0].shape)
    print(xx[0])

    def log_prob(value):
        return jnp.log(m.prob(m, value))
    print(jax.jacrev(log_prob)(x))

    print('caca')
    print(m.coords.shape)
    print(m.weights.shape)
