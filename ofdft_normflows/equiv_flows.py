from typing import Any, Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap, jacrev, random
from jax.experimental.ode import odeint


import flax
from flax import linen as nn


@jax.custom_jvp
def safe_sqrt(x):
    return jnp.sqrt(x)


@safe_sqrt.defjvp
def safe_sqrt_jvp(primals, tangents):
    x = primals[0]
    x_dot = tangents[0]
    primal_out = safe_sqrt(x)
    tangent_out = 0.5 * x_dot / jnp.where(x > 0, primal_out, jnp.inf)
    return primal_out, tangent_out
    # https://github.com/cagrikymk/JAX-ReaxFF/blob/master/jaxreaxff/forcefield.py


'''
Adaptation of 
Equivariant Flows: sampling configurations for
multi-body systems with symmetric energies
https://ml4physicalsciences.github.io/2019/files/NeurIPS_ML4PS_2019_79.pdf

'''
class NN(nn.Module):
    in_out_dims: Any
    features: Tuple[int]

    def setup(self):
        self.layers = [nn.Dense(feat)
                       for feat in self.features]  # [1:]
        self.last_layer = nn.Dense(1,
        kernel_init=jax.nn.initializers.zeros,
        bias_init=jax.nn.initializers.zeros)
        # self.nuclei = self.xyz_nuclei[:,None]

    @nn.compact
    def __call__(self, t, xi_norm, zi_one_hot):

        z = jnp.hstack((t, xi_norm, zi_one_hot))
        # z = xi_norm
        for i, lyr in enumerate(self.layers):
            z = lyr(z)
            z = nn.tanh(z)
        z = self.last_layer(z)
        return z

class RadialMLP(nn.Module):
    in_out_dims: Any
    features: Tuple[int]
    xyz_nuclei: Any
    z_one_hot: Any

    def setup(self):
        self.nuclei = self.xyz_nuclei[:, None]
        self.z_ = self.z_one_hot

    @nn.compact
    def __call__(self, t, samples):
        vmap_radialblock = nn.vmap(NN,
                                   variable_axes={'params': None, },
                                   split_rngs={'params': False, },
                                   in_axes=(None, 0, 0))(self.in_out_dims, self.features)

        z = lax.expand_dims(samples, dimensions=(0,)) - self.nuclei
        z_norm = jnp.linalg.norm(z, axis=-1)
        x = vmap_radialblock(t, z_norm, self.z_)
        x = jnp.einsum('ijk,ij->k', z, x)
        return x


class EqvFlow(nn.Module):
    """Equivariant Flows: sampling configurations for 
        multi-body systems with symmetric energies
    """
    in_out_dim: Any
    features: Tuple[int]
    xyz_nuclei: Any
    z_one_hot: Any

    def setup(self):
        self.net = RadialMLP(self.in_out_dim, self.features,
                             self.xyz_nuclei, self.z_one_hot)

    @nn.compact
    def __call__(self, t, states):
        z, logp_z = states[:, :self.in_out_dim], states[:, self.in_out_dim:]

        def f(z): return self.net(t, z)
        df_dz = vmap(jax.jacrev(f))(z)
        dlogp_z_dt = -1.0 * jnp.trace(df_dz, 0, 1, 2)

        dz = vmap(f)(z)
        return lax.concatenate((dz, dlogp_z_dt[:, None]), 1)


class Gen_EqvFlow(nn.Module):
    """Negative CNF for jax's odeint."""
    in_out_dim: Any
    features: Tuple[int]
    xyz_nuclei: Any
    z_one_hot: Any
    bool_neg: bool = False

    def setup(self) -> None:
        self.cnf = EqvFlow(self.in_out_dim, self.features,
                           self.xyz_nuclei, self.z_one_hot)
        if self.bool_neg:
            self.y0 = -1.
        else:
            self.y0 = 1.

    @nn.compact
    def __call__(self, t, states):
        outputs = self.cnf(self.y0 * t, states)
        return self.y0 * outputs




