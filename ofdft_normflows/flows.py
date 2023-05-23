from typing import Any, Callable, Sequence, Optional, Union
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap
import jax.random as jrnd
from jax._src import prng

import flax
from flax import linen as nn
from flax.linen import jit as jit_flax
from jaxopt import ScipyRootFinding
import optax

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]


# @jit_flax
class PlanarFlow(nn.Module):
    dims: int
    inv_num_iterations: int = 1000

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.linear0 = nn.Dense(1)
        self.linear1 = nn.Dense(self.dims, use_bias=False)

    def _f(self, z):
        y = self.linear0(z)
        y = nn.tanh(y)
        return z + self.linear1(y)

    def _log_det(self, z):
        df_dz = jax.jacrev(self._f)(z)
        df_dz = df_dz.reshape(z.shape[-1], z.shape[-1])
        v_det = jnp.abs(jax.numpy.linalg.det(df_dz))
        return jnp.log(v_det)

    # def _f_inverse(self, f_z):
    #     z = f_z
    #     for _ in range(self.inv_num_iterations):
    #         wtz_b = self.linear0(z)
    #         z = f_z - self.linear1(nn.tanh(wtz_b))
    #     return z

    def _f_inverse(self, f_z):
        def _f_root(z, f_z):
            return self._f(z) - f_z
        roots = ScipyRootFinding(method='hybr',
                                 optimality_fun=_f_root)

        return roots.run(init_params=jnp.zeros_like(f_z), f_z=f_z).params

    def __call__(self, z, forward=True):
        if forward:
            x = self._f(z)
            log_det = vmap(self._log_det)(jnp.expand_dims(z, axis=1))
            return x, log_det
        else:
            y = self._f_inverse(z)
            log_det = vmap(self._log_det)(jnp.expand_dims(z, axis=1))
            return y, log_det


# @jit_flax
class NormFlow(nn.Module):
    n_flows: int
    dims: int
    # N: int

    def setup(self):
        self.flows = [PlanarFlow(self.dims)
                      for i in range(self.n_flows)]

    def __call__(self, z, forward=True):
        sum_log_det = 0
        x = z  # transformed_sample
        for flow in self.flows:
            z, ldj = flow(x, forward)
            x = z
            sum_log_det += ldj
        return x, sum_log_det


def main_single_flow():
    import jax.random as jrnd
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)
    model = PlanarFlow(1)
    params = model.init(key, jnp.ones((3, 1)))
    print(params)
    x, ldj = model.apply(params, jnp.ones((3, 1)))
    w = model.apply(params, x, False)
    print(x)
    print(w)


def main_flows():
    import jax.random as jrnd
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)
    model = NormFlow(1, 1)
    params = model.init(key, jnp.ones((3, 1)))
    print(params)
    x, ldj = model.apply(params, jnp.ones((3, 1)))
    w = model.apply(params, x, False)
    print(x)
    print(w)


if __name__ == "__main__":
    # main_single_flow()
    main_flows()
