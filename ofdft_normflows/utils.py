from typing import Any, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, hessian, jacrev, lax

Array = jax.Array
BHOR = 1.  # 1.8897259886  # 1AA to BHOR


@partial(jit,  static_argnums=(2,))
def laplacian(params: Any, X: Array, fun: callable) -> jax.Array:
    """_summary_

    Args:
        X (Array): _description_
        params (Any): _description_
        fun (callable): _description_

    Returns:
        jax.Array: _description_
    """
    @partial(jit,  static_argnums=(2,))
    def _laplacian(params: Any, X: Array, fun: callable):
        hes_ = hessian(fun, argnums=1)(
            params, X[jnp.newaxis], )  # R[jnp.newaxis]
        hes_ = jnp.squeeze(hes_, axis=(0, 2, 4))
        hes_ = jnp.einsum('...ii', hes_)
        return hes_

    v_laplacian = vmap(_laplacian, in_axes=(None, 0,  None))
    return v_laplacian(params, X, fun)


@partial(jit,  static_argnums=(2,))
def score(params: Any, X: Array, fun: callable) -> jax.Array:

    @jit
    def _score(params: Any, xi: Array):
        score_ = jax.jacrev(fun, argnums=1)(params, xi[jnp.newaxis])
        return jnp.reshape(score_, xi.shape[0])

    v_score = vmap(_score, in_axes=(None, 0))
    return v_score(params, X)
