from typing import Any, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, hessian, jacrev, lax
import optax

import jax.random as jrnd
from jax._src import prng

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


def batches_generator_w_score(key: prng.PRNGKeyArray, batch_size: int, prior_dist: Callable):
    v_score = vmap(jax.jacrev(lambda x:
                              prior_dist.log_prob(x)))
    while True:
        _, key = jrnd.split(key)
        samples = prior_dist.sample(seed=key, sample_shape=batch_size)
        logp_samples = prior_dist.log_prob(samples)
        score = v_score(samples)
        samples0 = lax.concatenate(
            (samples, logp_samples[:, None], score), 1)

        _, key = jrnd.split(key)
        samples = prior_dist.sample(seed=key, sample_shape=batch_size)
        logp_samples = prior_dist.log_prob(samples)
        score = v_score(samples)
        samples1 = lax.concatenate(
            (samples, logp_samples[:, None], score), 1)

        yield lax.concatenate((samples0, samples1), 0)


def get_scheduler(epochs: int, sched_type: str = 'zero', lr: float = 3E-4):
    try:
        float(sched_type)
        v = float(sched_type)
        return optax.constant_schedule(v)
    except ValueError:
        if sched_type == 'zero':
            return optax.constant_schedule(0.0)
        elif sched_type == 'one':
            return optax.constant_schedule(1.)
        elif sched_type == 'const' or sched_type == 'c':
            return optax.constant_schedule(lr)
        elif sched_type == 'cos_decay':
            return optax.warmup_cosine_decay_schedule(
                init_value=lr,
                peak_value=lr,
                warmup_steps=150,
                decay_steps=epochs,
                end_value=1E-5,
            )
        elif sched_type == 'mix':
            init_scheduler_min = optax.warmup_cosine_decay_schedule(
                init_value=lr,
                peak_value=lr,
                warmup_steps=150,
                decay_steps=int(2*epochs/3),
                end_value=1E-6,
            )
            constant_scheduler_max = optax.constant_schedule(1E-6)
            return optax.join_schedules([init_scheduler_min,
                                        constant_scheduler_max], boundaries=[2*epochs/3, 3*epochs/3])
        elif sched_type == 'mix_old':
            constant_scheduler_min = optax.constant_schedule(lr)
            cosine_decay_scheduler = optax.cosine_onecycle_schedule(transition_steps=epochs, peak_value=lr,
                                                                    div_factor=50., final_div_factor=1.)
            constant_scheduler_max = optax.constant_schedule(1E-5)
            return optax.join_schedules([constant_scheduler_min, cosine_decay_scheduler,
                                        constant_scheduler_max], boundaries=[epochs/4, 2*epochs/4])


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    epochs = 1000
    total_steps = epochs
    cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1.,
        peak_value=1.0,
        warmup_steps=100,
        decay_steps=epochs,
        end_value=1E-6,
    )

    lrs = [cosine_decay_scheduler(i) for i in range(total_steps)]

    plt.scatter(range(total_steps), lrs)
    plt.title("Cosine Decay Scheduler")
    plt.ylabel("Learning Rate")
    plt.xlabel("Epochs/Steps")
    plt.show()
