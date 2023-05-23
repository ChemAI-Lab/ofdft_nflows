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


class PlanarFlow_OLD(nn.Module):
    dims: int
    inv_num_iterations: int = 1000

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.linear0 = nn.Dense(1, kernel_init=jax.ones_init)
        self.linear1 = nn.Dense(self.dims, use_bias=False,
                                kernel_init=jax.ones_init)

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


class PlanarFlow(nn.Module):
    # https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows/planar.py
    features: int
    x_features: int
    kernel_init: Callable[[PRNGKey, Shape, Dtype],
                          Array] = initializers.ones_init()   # default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = initializers.zeros_init()
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None

    def setup(self) -> None:
        self.w = self.param('w',
                            self.kernel_init,
                            (self.x_features, 1),
                            self.param_dtype)
        self.b = self.param('b', self.bias_init, (1,),
                            self.param_dtype)
        self.u = self.param('u',
                            self.kernel_init,
                            (1, self.features),
                            self.param_dtype)
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

    def _forward(self, z):
        lin = lax.dot_general(z, self.w, (((z.ndim - 1,), (0,)), ((), ())),
                              precision=self.precision,) + self.b
        return z + lax.dot_general(self.act(lin), self.u, (((lin.ndim - 1,), (0,)), ((), ())),
                                   precision=self.precision,)

    def _log_det_forward(self, z: Any):
        df_dz = jax.jacrev(self._forward)(z)
        df_dz = df_dz.reshape(z.shape[-1], z.shape[-1])
        v_det = jnp.abs(jax.numpy.linalg.det(df_dz))
        return jnp.log(v_det)

    def _inverse(self, x):
        z = x
        lin = lax.dot_general(z, self.w, (((z.ndim - 1,), (0,)), ((), ())),
                              precision=self.precision,) + self.b

    @compact
    def __call__(self, z) -> Any:
        x = self._forward(z)
        log_det = vmap(self._log_det_forward)(jnp.expand_dims(z, axis=1))
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


def main_single_flow_ravh():
    import jax.random as jrnd
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)
    model = PlanarFlow_OLD(1)
    params = model.init(key, jnp.ones((1, 1)))
    print(params)
    x, ldj = model.apply(params, jnp.ones((3, 1)))
    print(x)

    print('ravh')
    model = PlanarFlow(1, 1)
    params1 = model.init(key, jnp.ones((1, 1)))
    print(params1)
    x, ldj = model.apply(params1, jnp.ones((3, 1)))
    print(x)
    print(ldj)

    zz1, _ = jax.tree_util.tree_flatten(params1)
    for zi in enumerate(zz):
        print(zi)
    zz, _ = jax.tree_util.tree_flatten(params)
    for zi in enumerate(zip(zz, zz1)):
        print(zi)


def main_flows():
    import jax.random as jrnd
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)
    model = NormFlow(2, 1)
    params = model.init(key, jnp.ones((1, 1)))
    z = jrnd.normal(key, (10, 1))
    x, ldj = model.apply(params, z)
    z = model.apply(params, x, False)

    print('NormFlow')
    # print(params)
    print('z', z)
    print('forward', x)
    print('reverse', z)


if __name__ == "__main__":
    # main_single_flow()
    # main_single_flow_ravh()
    main_flows()
