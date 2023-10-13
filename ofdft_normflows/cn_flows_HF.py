import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from functools import partial
import jax
from typing import Any, Callable, Sequence, Optional, NewType
from jax import lax, random, vmap, scipy, numpy as jnp
from jax.experimental.ode import odeint

import flax
from flax.training import train_state
from flax import traverse_util
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import serialization
import optax
from sklearn.datasets import make_circles, make_moons, make_s_curve
from tqdm import tqdm


# os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# file from https://huggingface.co/flax-community/NeuralODE_SDE/raw/main/train_cnf.py

# @nn.jit
class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the Pytorch implementation at:
    https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py
    """
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64

    @nn.compact
    def __call__(self, t):
        # predict params
        blocksize = self.width * self.in_out_dim
        params = lax.expand_dims(t, (0, 1))
        params = nn.Dense(self.hidden_dim)(params)
        params = nn.tanh(params)
        params = nn.Dense(self.hidden_dim)(params)
        params = nn.tanh(params)
        params = nn.Dense(3 * blocksize + self.width)(params)

        # restructure
        params = lax.reshape(params, (3 * blocksize + self.width,))
        W = lax.reshape(params[:blocksize], (self.width, self.in_out_dim, 1))

        U = lax.reshape(params[blocksize:2 * blocksize],
                        (self.width, 1, self.in_out_dim))

        G = lax.reshape(params[2 * blocksize:3 * blocksize],
                        (self.width, 1, self.in_out_dim))
        U = U * nn.sigmoid(G)

        B = lax.expand_dims(params[3 * blocksize:], (1, 2))
        return W, B, U


class CNF(nn.Module):
    """Adapted from the Pytorch implementation at:
    https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py
    """
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64

    @nn.compact
    def __call__(self, t, states):
        z, logp_z = states[:, :self.in_out_dim], states[:, self.in_out_dim:]
        W, B, U = HyperNetwork(self.in_out_dim, self.hidden_dim, self.width)(t)

        def dzdt(z):
            h = nn.tanh(vmap(jnp.matmul, (None, 0))(z, W) + B)
            return jnp.matmul(h, U).mean(0)

        dz_dt = dzdt(z)
        def sum_dzdt(z): return dzdt(z).sum(0)
        df_dz = jax.jacrev(sum_dzdt)(z)
        dlogp_z_dt = -1.0 * jnp.trace(df_dz, 0, 0, 2)

        return lax.concatenate((dz_dt, lax.expand_dims(dlogp_z_dt, (1,))), 1)


class Gen_CNF(nn.Module):
    """Negative CNF for jax's odeint."""
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64
    bool_neg: bool = True

    def setup(self) -> None:
        self.cnf = CNF(self.in_out_dim, self.hidden_dim,
                       self.width)
        if self.bool_neg:
            self.y0 = -1.
        else:
            self.y0 = 1.

    @nn.compact
    def __call__(self, t, states):
        outputs = self.cnf(self.y0 * t, states)
        return self.y0 * outputs


def get_batch_circles(num_samples):
    """Adapted from the Pytorch implementation at:

    """
    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    x = jnp.array(points, dtype=jnp.float32)
    logp_diff_t1 = jnp.zeros((num_samples, 1), dtype=jnp.float32)

    return lax.concatenate((x, logp_diff_t1), 1)


def get_batch_moons(num_samples):
    points, _ = make_moons(n_samples=num_samples, noise=0.05)
    x = jnp.array(points, dtype=jnp.float32)
    logp_diff_t1 = jnp.zeros((num_samples, 1), dtype=jnp.float32)

    return lax.concatenate((x, logp_diff_t1), 1)


def get_batch_scurve(num_samples):
    points, _ = make_s_curve(n_samples=num_samples, noise=0.05, random_state=0)
    x1 = jnp.array(points, dtype=jnp.float32)[:, :1]
    x2 = jnp.array(points, dtype=jnp.float32)[:, 2:]
    x = lax.concatenate((x1, x2), 1)
    logp_diff_t1 = jnp.zeros((num_samples, 1), dtype=jnp.float32)

    return lax.concatenate((x, logp_diff_t1), 1)


def multivariate_normal(z):
    """
    Log probability of multivariate_normal.
    """
    mean = jnp.array([0., 0.])
    z_m = z - mean
    cov = jnp.array([[0.1, 0.], [0., 0.1]])
    logz = -jnp.log((2 * jnp.pi)) + -0.5 * jnp.log(jnp.linalg.det(cov)) + - \
        0.5 * jnp.matmul(jnp.matmul(z_m.T, jnp.linalg.inv(cov)), z_m)
    return logz


def create_train_state(rng, learning_rate, in_out_dim, hidden_dim, width):
    """Creates initial 'TrainState'."""
    inputs = jnp.ones((1, 2))
    neg_cnf = Gen_CNF(in_out_dim, hidden_dim, width)
    params = neg_cnf.init(rng, jnp.array(10.), inputs)['params']
    set_params(params)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=neg_cnf.apply, params=params, tx=tx
    )


def set_params(params):
    # Convert all value of Params to certain constant
    params = unfreeze(params)
    # Get flattened-key: value list.
    flat_params = {'/'.join(k): v for k,
                   v in traverse_util.flatten_dict(params).items()}
    unflat_params = traverse_util.unflatten_dict(
        {tuple(k.split('/')): 0.1 * jnp.ones_like(v) for k, v in flat_params.items()})
    new_params = freeze(unflat_params)
    test_x = jnp.array([[0., 1.], [2., 3.], [4., 5.]])
    test_log_p = jnp.zeros((3, 1))
    test_inputs = lax.concatenate((test_x, test_log_p), 1)
    Gen_CNF().apply({'params': new_params}, jnp.array(0.), test_inputs)


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def train_step(state, batch, in_out_dim, hidden_dim, width, t0, t1):
    def p_z0(x): return scipy.stats.multivariate_normal.logpdf(x,
                                                               mean=jnp.array(
                                                                   [0., 0.]),
                                                               cov=jnp.array([[0.1, 0.], [0., 0.1]]))
    vmap_multi = jax.vmap(multivariate_normal, 0, 0)

    def loss_fn(params):
        def func(states, t): return Gen_CNF(in_out_dim, hidden_dim,
                                            width).apply({'params': params}, t, states)

        outputs = odeint(
            func,
            batch,
            -1.0 * jnp.array([t1, t0]),
            atol=1e-5,
            rtol=1e-5
        )

        z_t, logp_diff_t = outputs[:, :,
                                   :in_out_dim], outputs[:, :, in_out_dim:]
        z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
        logp_x = p_z0(z_t0) - lax.squeeze(logp_diff_t0, dimensions=(1,))
        loss = -logp_x.mean(0)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


def train(learning_rate, n_iters, batch_size, in_out_dim, hidden_dim, width, t0, t1, visual, dataset):
    """Train the model."""
    rng = jax.random.PRNGKey(0)
    state = create_train_state(
        rng, learning_rate, in_out_dim, hidden_dim, width)
    if dataset == "circles":
        def get_batch(num_samples): return get_batch_circles(num_samples)
    elif dataset == "moons":
        def get_batch(num_samples): return get_batch_moons(num_samples)
    elif dataset == "scurve":
        def get_batch(num_samples): return get_batch_scurve(num_samples)

    for itr in range(1, n_iters+1):
        batch = get_batch(batch_size)
        state, loss = train_step(
            state, batch, in_out_dim, hidden_dim, width, t0, t1)
        print("iter: %d, loss: %.2f" % (itr, loss))

    if visual is True:
        # Convert Params of Neg_CNF to CNF
        neg_params = state.params
        neg_params = unfreeze(neg_params)
        # Get flattened-key: value list.
        neg_flat_params = {
            '/'.join(k): v for k, v in traverse_util.flatten_dict(neg_params).items()}
        pos_flat_params = {key[6:]: jnp.array(
            np.array(neg_flat_params[key])) for key in list(neg_flat_params.keys())}
        pos_unflat_params = traverse_util.unflatten_dict(
            {tuple(k.split('/')): v for k, v in pos_flat_params.items()})
        pos_params = freeze(pos_unflat_params)

        jax.profiler.save_device_memory_profile("memory.prof")
        output = viz(neg_params, pos_params, in_out_dim,
                     hidden_dim, width, t0, t1, dataset)
        z_t_samples, z_t_density, logp_diff_t, viz_timesteps, target_sample, z_t1 = output
        create_plots(z_t_samples, z_t_density, logp_diff_t, t0,
                     t1, viz_timesteps, target_sample, z_t1, dataset)


def solve_dynamics(dynamics_fn, initial_state, t):
    def f(initial_state, t):
        return odeint(dynamics_fn, initial_state, t, atol=1e-5, rtol=1e-5)
    return f(initial_state, t)


def viz(neg_params, pos_params, in_out_dim, hidden_dim, width, t0, t1, dataset):
    """Adapted from PyTorch """
    viz_samples = 30000
    viz_timesteps = 41
    if dataset == "circles":
        def get_batch(num_samples): return get_batch_circles(num_samples)
    elif dataset == "moons":
        def get_batch(num_samples): return get_batch_moons(num_samples)
    elif dataset == "scurve":
        def get_batch(num_samples): return get_batch_scurve(num_samples)
    target_sample = get_batch(viz_samples)[:, :2]

    if not os.path.exists('results_%s/' % dataset):
        os.makedirs('results_%s/' % dataset)

    z_t0 = jnp.array(np.random.multivariate_normal(mean=np.array([0., 0.]),
                                                   cov=np.array(
                                                       [[0.1, 0.], [0., 0.1]]),
                                                   size=viz_samples))
    logp_diff_t0 = jnp.zeros((viz_samples, 1), dtype=jnp.float32)
    print(neg_params)
    print(pos_params)

    def func_pos(states, t): return Gen_CNF(in_out_dim, hidden_dim,
                                            width, False).apply({'params': pos_params}, t, states)
    output = solve_dynamics(func_pos, lax.concatenate(
        (z_t0, logp_diff_t0), 1), jnp.linspace(t0, t1, viz_timesteps))
    z_t_samples, _ = output[..., :2], output[..., 2:]

    # Generate evolution of density
    x = jnp.linspace(-1.5, 1.5, 100)
    y = jnp.linspace(-1.5, 1.5, 100)
    points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

    z_t1 = jnp.array(points, dtype=jnp.float32)
    logp_diff_t1 = jnp.zeros((z_t1.shape[0], 1), dtype=jnp.float32)
    def func_neg(states, t): return Gen_CNF(in_out_dim, hidden_dim,
                                            width).apply({'params': neg_params}, t, states)
    output = solve_dynamics(func_neg, lax.concatenate(
        (z_t1, logp_diff_t1), 1), -jnp.linspace(t1, t0, viz_timesteps))
    z_t_density, logp_diff_t = output[..., :2], output[..., 2:]

    return z_t_samples, z_t_density, logp_diff_t, viz_timesteps, target_sample, z_t1


def create_plots(z_t_samples, z_t_density, logp_diff_t, t0, t1, viz_timesteps, target_sample, z_t1, dataset):
    # Create plots for each timestep
    for (t, z_sample, z_density, logp_diff) in zip(
            tqdm(np.linspace(t0, t1, viz_timesteps)),
            z_t_samples, z_t_density, logp_diff_t
    ):
        fig = plt.figure(figsize=(12, 4), dpi=200)
        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)
        fig.suptitle(f'{t:.2f}s')

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title('Target')
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title('Samples')
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('Log Probability')
        ax3.get_xaxis().set_ticks([])
        ax3.get_yaxis().set_ticks([])

        ax1.hist2d(*jnp.transpose(target_sample), bins=300, density=True,
                   range=[[-1.5, 1.5], [-1.5, 1.5]])

        ax2.hist2d(*jnp.transpose(z_sample), bins=300, density=True,
                   range=[[-1.5, 1.5], [-1.5, 1.5]])

        def p_z0(x): return scipy.stats.multivariate_normal.logpdf(x,
                                                                   mean=jnp.array(
                                                                       [0., 0.]),
                                                                   cov=jnp.array([[0.1, 0.], [0., 0.1]]))
        logp = p_z0(z_density) - lax.squeeze(logp_diff, dimensions=(1,))
        ax3.tricontourf(*jnp.transpose(z_t1),
                        jnp.exp(logp), 200)

        plt.savefig(os.path.join('results_%s/' % dataset, f"cnf-viz-{int(t * 1000):05d}.jpg"),
                    pad_inches=0.2, bbox_inches='tight')
        plt.close()

    img, *imgs = [Image.open(f) for f in sorted(
        glob.glob(os.path.join('results_%s/' % dataset, f"cnf-viz-*.jpg")))]
    img.save(fp=os.path.join('results_%s/' % dataset, "cnf-viz.gif"), format='GIF', append_images=imgs,
             save_all=True, duration=250, loop=0)

    print('Saved visualization animation at {}'.format(
        os.path.join('results_%s/' % dataset, "cnf-viz.gif")))


if __name__ == '__main__':
    train(0.001, 100, 512, 2, 32, 64, 0., 10., True, 'scurve')
