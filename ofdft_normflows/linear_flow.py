from typing import Any, Callable, Sequence, Optional, Union, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, lax
import jax.random as jrnd
from jax._src import prng

import flax
from flax import linen as nn
from flax.linen import jit as jit_flax
from flax.linen import initializers
from flax.linen.module import compact, Module

from jaxopt import ScipyRootFinding
import optax

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]

default_kernel_init = initializers.lecun_normal()
normal_init = initializers.normal(stddev=1.0)
ones_init = initializers.ones


class LinearFlow(nn.Module):
    features: int

    def setup(self) -> None:
        self.lin_layer = nn.Dense(self.features)

    def _forward(self, z):
        return self.lin_layer(z)

    def _det_forward(self, z: Any):
        df_dz = jax.jacrev(self._forward)(z)
        df_dz = df_dz.reshape(z.shape[-1], z.shape[-1])
        v_det = jnp.abs(jax.numpy.linalg.det(df_dz))
        return v_det

    @compact
    def __call__(self, z) -> Any:

        x = self._forward(z)
        _det = vmap(self._det_forward)(jnp.expand_dims(z, axis=1))
        return x, jnp.log(_det).reshape(z.shape[0], 1)


def main_test():

    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng, 2)

    u = jrnd.uniform(key, (5, 1), minval=-10., maxval=10.)
    log_pu = jax.scipy.stats.norm.logpdf(u)

    print(u.shape, log_pu.shape)
    # assert 0

    flow = LinearFlow(1)
    params = flow.init(key, jnp.zeros((1, 1)))
    print(params)
    x, ldj = flow.apply(params, u)

    u = jnp.linspace(-50., 50., 100000)
    u = jnp.expand_dims(u, axis=1)
    log_pu = jax.scipy.stats.norm.logpdf(u)
    pu = jnp.exp(log_pu)

    int_pu = jnp.trapz(pu.ravel(), u.ravel())
    print(f'int_pu = {int_pu}')

    x, ldj = flow.apply(params, u)
    log_px = log_pu - ldj
    px = jnp.exp(log_px)
    int_px = jnp.trapz(px.ravel(), x.ravel())
    print(f'int_px = {int_px}')
    print(ldj)


if __name__ == "__main__":
    main_test()
