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

# @jit_flax


class PlanarFlow(nn.Module):
    # https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows/planar.py
    features: int
    x_features: int
    kernel_init: Callable[[PRNGKey, Shape, Dtype],
                          Array] = initializers.ones_init()   # default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = initializers.zeros_init()
    param_dtype: Dtype = jnp.float64
    precision: PrecisionLike = None

    def setup(self) -> None:
        stdv = 1./jnp.sqrt(self.features)
        self.w = self.param('w', lambda rng, shape, dtype:
                            jrnd.uniform(rng, shape, minval=-stdv,
                                         maxval=stdv, dtype=dtype),
                            (self.x_features, 1), self.param_dtype)
        self.b = self.param('b', self.bias_init, (1,),
                            self.param_dtype)
        self.u = self.param('u', lambda rng, shape, dtype:
                            jrnd.uniform(rng, shape, minval=-stdv,
                                         maxval=stdv, dtype=dtype),
                            (1, self.features), self.param_dtype)
        self.act = nn.tanh

    def _forward0(self, z: Any):
        lin = lax.dot_general(z, self.w, (((z.ndim - 1,), (0,)), ((), ())),
                              precision=self.precision,) + self.b
        inner = jnp.sum(self.w*self.u)
        u = self.u + (jnp.log(1 + jnp.exp(inner)) - 1 - inner) \
            * self.w / jnp.sum(self.w ** 2)  # constraint w.T * u > -1

        z_ = z + lax.dot_general(self.act(lin), self.u, (((lin.ndim - 1,), (0,)), ((), ())),
                                 precision=self.precision,)
        return z_

    @property
    def _u_hat(self):
        dot = lax.dot_general(self.u, self.w, (((self.u.ndim - 1,), (0,)), ((), ())),
                              precision=self.precision,)
        m = -1 + jnp.log(1+jnp.exp(dot))
        du = (m-dot)/jnp.linalg.norm(self.w)*self.w
        u_hat = self.u + du
        return u_hat

    def _forward(self, z):
        lin = lax.dot_general(z, self.w, (((z.ndim - 1,), (0,)), ((), ())),
                              precision=self.precision,) + self.b
        return z + lax.dot_general(self.act(lin), self._u_hat, (((lin.ndim - 1,), (0,)), ((), ())),
                                   precision=self.precision,)

    def _log_det_forward_AD(self, z: Any):
        df_dz = jax.jacrev(self._forward)(z)
        df_dz = df_dz.reshape(z.shape[-1], z.shape[-1])
        v_det = jnp.abs(jax.numpy.linalg.det(df_dz))
        return jnp.log(v_det)

    def _det_forward(self, z: Any):

        # note: has to return scalar (does not include the batch dimension)
        def _h(z): return jnp.sum(self.act(lax.dot_general(z, self.w, (((z.ndim - 1,), (0,)), ((), ())),
                                                           precision=self.precision,) + self.b))
        _g_h = jax.grad(_h)
        psi = _g_h(z)
        u_dot_psi = jnp.dot(psi, self._u_hat.ravel())
        return 1. + u_dot_psi

    def _inverse(self, x):
        z = x
        lin = lax.dot_general(z, self.w, (((z.ndim - 1,), (0,)), ((), ())),
                              precision=self.precision,) + self.b

    @compact
    def __call__(self, z) -> Any:
        self._u_hat
        x = self._forward(z)
        _det = vmap(self._det_forward)(z)
        log_det = jnp.log(_det + 1E-8)
        return x, jnp.expand_dims(log_det, axis=1)


# @jit_flax
class NormFlow(nn.Module):
    n_flows: int
    dims: int
    x_features: int = 1
    inv_num_iterations: int = 100
    # N: int

    def setup(self):
        self.flows = [PlanarFlow(self.dims, self.x_features)
                      for i in range(self.n_flows)]

    def _forward(self, z):
        sum_log_det = 0
        x = z  # transformed_sample
        for flow in self.flows:
            z, ldj = flow(x)
            x = z
            sum_log_det += ldj
        return x, sum_log_det

    def _inverse(self, f_z):
        def _f_root(z, f_z):
            z, _ = self._forward(z)
            return z - f_z
        roots = ScipyRootFinding(method='hybr',
                                 optimality_fun=_f_root)

        return roots.run(init_params=jnp.zeros_like(f_z), f_z=f_z).params

    @compact
    def __call__(self, z, forward=True):
        if forward:
            return self._forward(z)
        else:
            return self._inverse(z)
