from typing import Any, Callable, ClassVar
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, hessian, jacrev

Array = jnp.ndarray


@partial(jit,  static_argnums=(2,))
def laplacian(X: Array, params: Any, fun: callable) -> jnp.DeviceArray:
    """_summary_

    Args:
        X (Array): _description_
        params (Any): _description_
        fun (callable): _description_

    Returns:
        jnp.DeviceArray: _description_
    """
    @partial(jit,  static_argnums=(2,))
    def _laplacian(X: Array, params: Any, fun: callable):
        hes_ = hessian(fun)(
            X[jnp.newaxis], params)  # R[jnp.newaxis]
        hes_ = jnp.squeeze(hes_, axis=(0, 2, 4))
        hes_ = jnp.einsum('...ii', hes_)
        return hes_

    v_laplacian = vmap(_laplacian, in_axes=(0, None, None))
    return v_laplacian(X, params, fun)
