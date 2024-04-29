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
from distrax import MixtureSameFamily, MultivariateNormalDiag, Categorical
from distrax._src.distributions import distribution
from distrax._src.distributions.distribution import Array

Any = Any
Array = jax.Array
PRNGKey = jax.random.PRNGKey
EventT = distribution.EventT
Dtype = Any

class MixGaussian(distrax.Distribution):
  def __init__(self, loc: Array , scale_diag: Array , probs: Array):
    r"""
    Creates a distribution with a mixture of Gaussian components.

    Parameters
    ----------
    loc : Array
        Molecular coordinates.
    scale_diag : Array
        Sigma matrix.
    probs : Array
        Mixture probabilities.
    """    
   
    self.loc = loc
    self.scale_diag = scale_diag
    self.probs = probs
    self.mixture_dist = Categorical(probs=probs)
    self.components_dist = MultivariateNormalDiag(loc=self.loc,scale_diag=self.scale_diag)

   
  @jax.jit
  def prob(self, value: Array) -> jax.Array:
    """
    Calculates the probability of an event.

    Parameters
    ----------
    value : Array
        An event. 

    Returns
    -------
    jax.Array
        The probability of the event.
    """    
    log_px_components_dist = self.components_dist.log_prob(value).T
    px_components_dist = jnp.exp(log_px_components_dist)
    px = px_components_dist@self.probs[:,None]
    return px

  @jax.jit
  def log_prob(self, value: Array) -> jax.Array:
    """
    Calculates the log probability of an event.

    Parameters
    ----------
    value : Array
        An event.

    Returns
    -------
    jax.Array
        The log probability of the event.
    """    
    return jnp.log(self.prob(value))


  def _sample_n(self, key: PRNGKey, n: int) -> jax.Array:
    """
    Returns 'n' samples. 

    Parameters
    ----------
    key : PRNGKey
        Random key.
    n : int
        Number of samples to generate. 

    Returns
    -------
    jax.Array
        An array of 'n' samples. 
    """    
    _, key_mixt, key_comp = jax.random.split(key,3)
    samples_mixt = self.mixture_dist._sample_n(key_mixt,n)
    samples_mixt_one_hot = jax.nn.one_hot(samples_mixt,self.probs.shape[-1])
    samples_comp = self.components_dist.sample(seed=key_comp, sample_shape=n)
    samples_comp = jnp.squeeze(samples_comp,axis=-2)

    samples = jnp.einsum('ijl,ij->il',samples_comp,samples_mixt_one_hot)
    return samples

  def event_shape(self):
      pass
      #6-31G(d,p)
  @jax.jit
  def score(self,values):
    return jax.vmap(jax.grad(lambda x:
                              self.log_prob(x).sum()))(values)
  
  
class DFTDistribution(distrax.Distribution):

    def __init__(self, atoms: Any, geometry: Any, basis_set: str = '6-31G(d,p)', exc: str = 'b3lyp', dtype_: Dtype = jnp.float32):

        self.atoms = atoms
        self.geometry = geometry
        self.basis_set = basis_set
        self.exc = exc
        self.dtype_ = dtype_

        self._grid_level = 5 # change this for larger molecules
        self.mol = self._mol()
        self.grids = dft.gen_grid.Grids(self.mol)
        self.grids.level = self._grid_level
        self.grids.build()
        self.Ne = self.mol.tot_electrons()
        self.dft, self.rdm1 = self._dft()

        self.coords = jnp.array(self.grids.coords)
        self.weights = jnp.array(self.grids.weights)
    
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
        LDA_X = 1.
        B88_X = 1.
        VWN_C = 1.

        mf_hf.xc = f'{LDA_X:} * LDA + {B88_X:} * B88, {VWN_C:} * VWN'

        mf_hf = mf_hf.newton() # second-order algortihm
        mf_hf.kernel()
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


