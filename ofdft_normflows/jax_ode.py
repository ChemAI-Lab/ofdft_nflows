import jax

from jax import lax,vmap,vjp
from jax import numpy as jnp 
from jax.experimental.ode import odeint
from typing import Any, Callable


def neural_ode(params: Any, batch: Any, f: Callable, t0: float, t1: float, d_dim: int) -> Any:
    """
    A function that computes the neural ODE for a given batch of data. Defines the initial and final time as 
    an array and then computes the output of the neural ODE using the odeint function from jax.experimental.ode.

    Parameters
    ----------
    params : Any
       Flow parameters.
    batch : Any
        Sampled batch of data. 
    f : Callable
        Neural ODE function.
    t0 : float
        Initial time.
    t1 : float
        Final time.
    d_dim : int
        Dimension of the system. 

    Returns
    -------
    Any
        Returns 'z' and log-likelihood of the function 'f' at the final time 't1' with the same shape as the input batch.
    """     
    start_and_end_time = jnp.array([t0, t1])

    def _evol_fun(states, t):
        return f.apply(params, t, states)

    outputs = odeint(
        _evol_fun,
        batch,
        start_and_end_time,
        atol=1e-7,
        rtol=1e-7
    )
    z_t, logp_diff_t = outputs[:, :,
                               :d_dim], outputs[:, :, d_dim:]
    z_t1, logp_diff_t1 = z_t[-1], logp_diff_t[-1]
    return z_t1, logp_diff_t1

def neural_ode_score(params: Any, batch: Any, f: Callable, t0: float, t1: float, d_dim: int) -> Any:
    """
    A function that computes the neural ODE for a given batch of data. Defines the initial and final time as 
    an array and then computes the output of the neural ODE using the odeint function from jax.experimental.ode.


    Parameters
    ----------
    params : Any
       Flow parameters.
    batch : Any
        Sampled batch of data.
    f : Callable
        Neural ODE function.
    t0 : float
        Initial time.
    t1 : float
        Final time.
    d_dim : int
        Dimension of the system.

    Returns
    -------
    Any
        Returns 'z', log-likelihood 'log_p" and the score 'score' of the function 'f' at the final time 't1' 
        with the same shape as the input batch.
    """    
    start_and_end_time = jnp.array([t0, t1])

    def _evol_fn_i(params, t, state):
        state = lax.expand_dims(state, dimensions=(0,))
        state, score = state[:, :-d_dim], state[:, -d_dim:]
        def _f_div(state): return jnp.sum(
            f.apply(params, t, state)[:, -1:])

        def _f_dx(state): return f.apply(params, t, state)[:, :-1]

        div, grad_div = jax.value_and_grad(_f_div)(state[:, :-1])
        dx, _f_vjp = vjp(_f_dx, state[:, :-1])
        score_vjp = _f_vjp(score)[0]
        dscore = -score_vjp+grad_div 
        state = lax.concatenate(
            (dx, lax.expand_dims(div, dimensions=(0, 1)), dscore), 1)
        return state.ravel()
    v_evol_fn_i = vmap(_evol_fn_i, in_axes=(None, None, 0), out_axes=(0))

    def _evol_fun(states, t):
        return v_evol_fn_i(params, t, states)

    outputs = odeint(
        _evol_fun,
        batch,
        start_and_end_time,
        atol=1e-7,
        rtol=1e-7
    )
    z_t, logp_diff_t, score_t = outputs[:, :,
                                        :d_dim], outputs[:, :, d_dim:d_dim+1], outputs[:, :, d_dim+1:]
    z_t1, logp_diff_t1, score_t1 = z_t[-1], logp_diff_t[-1], score_t[-1]
    return z_t1, logp_diff_t1, score_t1
    
def neural_ode_plotting(params: Any, batch: Any, f: Callable, t0: float, t1: float, d_dim: int, grid_t:int=10):    
    t_grid = jnp.linspace(t0,t1,grid_t)

    def _evol_fun(states, t):
        return f.apply(params, t, states)

    outputs = odeint(
        _evol_fun,
        batch,
        t_grid,
        atol=1e-5,
        rtol=1e-5
    )
    z_t, logp_diff_t = outputs[:, :,
                               :d_dim], outputs[:, :, d_dim:]
    return z_t, logp_diff_t


