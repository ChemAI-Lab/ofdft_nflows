from typing import Optional

import jax
from jax import lax
import jax.numpy as jnp
import chex
import distrax
from distrax import MultivariateNormalDiag, Categorical

Array = chex.Array
PRNGKey = chex.PRNGKey


AAtoBhor = 1.8897259886

class ProMolecularDensity(distrax.Distribution):
  def __init__(self,z: Optional[Array], 
               loc: Optional[Array],
               scale_diag: Optional[Array] = None,
               units:str='Bhor',
               ):
    """_summary_

      Args:
          z (Optional[Array]): _description_
          loc (Optional[Array]): _description_
          scale_diag (Optional[Array], optional): _description_. Defaults to None.
    """
    
    self.loc = lax.expand_dims(loc,dimensions=(1,))
    self.units = units
    
    if scale_diag is None:
        self.scale_diag = jnp.ones_like(self.loc)
    else:
        self.scale_diag = lax.expand_dims(scale_diag,dimensions=(1))
    
    if self.units.lower() != 'aa' or  self.units.lower() != 'angstrom':
        self.loc = self.loc*AAtoBhor
        self.scale_diag = self.scale_diag*AAtoBhor
        
    self.logits = z
    self.probs = z/jnp.linalg.norm(z)
    self.mixture_dist = Categorical(probs=self.probs)
    self.mixture_probs = self.mixture_dist.probs
    self.components_dist = MultivariateNormalDiag(loc=self.loc,scale_diag=self.scale_diag)

    # assert self.loc.shape[1] == self.scale_diag.shape[1] == self.probs.shape[-1]
  @jax.jit
  def prob(self, value):
    log_px_components_dist = self.components_dist.log_prob(value).T
    px_components_dist = jnp.exp(log_px_components_dist)
    px = px_components_dist@self.mixture_probs[:,None]
    return px

  @jax.jit
  def log_prob(self, value):
    return jnp.log(self.prob(value))


  def _sample_n(self, key, n):
    _, key_mixt, key_comp = jax.random.split(key,3)
    samples_mixt = self.mixture_dist._sample_n(key_mixt,n)
    samples_mixt_one_hot = jax.nn.one_hot(samples_mixt,self.mixture_probs.shape[-1])

    samples_comp = self.components_dist.sample(seed=key_comp, sample_shape=n)
    samples_comp = jnp.squeeze(samples_comp,axis=-2)
    
    samples = jnp.einsum('ijl,ij->il',samples_comp,samples_mixt_one_hot)
    return samples

  def event_shape(self):
      pass

  @jax.jit
  def score(self,values):
    return jax.vmap(jax.grad(lambda x:
                              self.log_prob(x).sum()))(values)

