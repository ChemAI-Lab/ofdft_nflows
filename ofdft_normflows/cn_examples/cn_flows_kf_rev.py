import matplotlib.pyplot as plt
from functools import partial
import jax
from typing import Any, Callable, Sequence, Optional, NewType
from jax import lax, random, vmap, scipy, numpy as jnp
from jax.experimental.ode import odeint
import jax.random as jrnd
import flax
from flax import linen as nn
import optax
from sklearn.datasets import make_s_curve

from cn_flows import Gen_CNF, neural_ode

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
    bool_neg: bool = False

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


@partial(jax.jit,  static_argnums=(2, 3, 4, 5,))
def neural_ode(params: Any, batch: Any, f: Callable, t0: float, t1: float, d_dim: int):
    # time as [t1 to t0] gives nans for the second term :/
    start_and_end_time = jnp.array([t0, t1])

    def _evol_fun(states, t):
        return f.apply(params, t, states)

    outputs = odeint(
        _evol_fun,
        batch,
        start_and_end_time,
        atol=1e-5,
        rtol=1e-5
    )
    z_t, logp_diff_t = outputs[:, :,
                               :d_dim], outputs[:, :, d_dim:]
    # z_t0, logp_diff_t0 = z_t[0], logp_diff_t[0]
    z_t1, logp_diff_t1 = z_t[-1], logp_diff_t[-1]
    # return lax.concatenate((z_t0, z_t1), 2), lax.concatenate((lax.expand_dims(logp_diff_t0, (1,)), lax.expand_dims(logp_diff_t1, (1,))), 1)
    return z_t1, logp_diff_t1
    # return outputs
# ---------------
# ---------------


def get_batch_scurve(num_samples):
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)
    while True:
        points, _ = make_s_curve(
            n_samples=num_samples, noise=0.05, random_state=0)
        x1 = jnp.array(points, dtype=jnp.float32)[:, :1]
        x2 = jnp.array(points, dtype=jnp.float32)[:, 2:]
        x = lax.concatenate((x1, x2), 1)
        logp_diff_t1 = jnp.zeros((num_samples, 1), dtype=jnp.float32)

        yield lax.concatenate((x, logp_diff_t1), 1)
# ---------------


def main(batch_size, epochs):

    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    model = Gen_CNF()
    test_inputs = lax.concatenate((jnp.ones((1, 2)), jnp.ones((1, 1))), 1)
    params = model.init(key, jnp.array(0.), test_inputs)

    from jax.tree_util import tree_map
    # params = jax.tree_util.tree_map(lambda x: 0.1 * jnp.ones_like(x), params)
    # print(params)

    def NODE(params, batch): return neural_ode(
        params, batch, model, 0., 10., 2)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    def log_p_z0(samples):
        d_dim = samples.shape[1]
        mean = jnp.zeros(d_dim)
        cov = 0.1*jnp.eye(d_dim)
        logp = jax.scipy.stats.multivariate_normal.logpdf(
            samples, mean=mean, cov=cov)
        return logp.reshape(samples.shape[0], 1)

    def loss(params, samples):
        zt0, logp_zt0 = NODE(params, samples)
        logp_x = log_p_z0(zt0) - logp_zt0
        # log_p_x = log_target_density(x)
        return -1.*jnp.mean(logp_x)
        # return jnp.linalg.norm(log_p_x - log_px)

    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    gen_batch = get_batch_scurve(batch_size)

    for i in range(epochs+1):

        batch = next(gen_batch)
        params, opt_state, loss_value = step(params, opt_state, batch)
        if i % 50 == 0:
            print(f'step {i}, loss: {loss_value}')

    png = jrnd.PRNGKey(1)
    _, key = jrnd.split(png)

    z0 = jrnd.multivariate_normal(key, mean=jnp.zeros(
        (2,)), cov=0.1*jnp.eye(2), shape=(2000,))
    logp_z0 = jnp.zeros((2000, 1))
    u_and_log_pu = lax.concatenate((z0, logp_z0), 1)

    model1 = Gen_CNF(bool_neg=True)
    zt, logp_zt = neural_ode(params, u_and_log_pu, model1, -10., 0., 2)

    plt.title('Samples from the NormFlow')
    plt.scatter(zt[:, 0], zt[:, 1])
    plt.savefig('Figures/CNF_twomoons.png')

    u0 = jnp.linspace(-3.5, 3.5, 100)
    u1 = jnp.linspace(-3.5, 3.5, 100)
    u0_, u1_ = jnp.meshgrid(u0, u1)
    zt = lax.concatenate(
        (jnp.expand_dims(u0_.ravel(), 1), jnp.expand_dims(u1_.ravel(), 1)), 1)
    logp_zt = jnp.zeros_like(zt[:, :1])
    zt_and_log_pzt = lax.concatenate((zt, logp_zt), 1)

    z0, logp_diff_z0 = neural_ode(params, zt_and_log_pzt, model, 0., 10., 2)
    logp_x = log_p_z0(z0) - logp_diff_z0

    plt.figure(0)
    plt.clf()
    plt.contour(u0_, u1_, logp_x.reshape(u0_.shape))
    plt.savefig('Figures/CNF_logP_twomoons.png')


if __name__ == '__main__':
    batch_size = 512
    epochs = 1000

    main(batch_size, epochs)
