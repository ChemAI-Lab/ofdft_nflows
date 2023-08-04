
from functools import partial
import jax
from typing import Any, Callable, Sequence, Optional, NewType, Tuple
from jax import lax, random, vmap, scipy, vjp, numpy as jnp
from jax.experimental.ode import odeint
import jax.random as jrnd

import flax
from flax import linen as nn
import optax

# file from https://huggingface.co/flax-community/NeuralODE_SDE/raw/main/train_cnf.py
# jax.config.update('jax_disable_jit', True)

a_initializer = jax.nn.initializers.normal()


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


class CNFRicky(nn.Module):
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


class Gen_CNFRicky(nn.Module):
    """Negative CNF for jax's odeint."""
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64
    bool_neg: bool = False

    def setup(self) -> None:
        self.cnf = CNFRicky(self.in_out_dim, self.hidden_dim,
                            self.width)
        if self.bool_neg:
            self.y0 = -1.
        else:
            self.y0 = 1.

    @nn.compact
    def __call__(self, t, states):
        outputs = self.cnf(self.y0 * t, states)
        return self.y0 * outputs


class FourierFeatures(nn.Module):
    num_features: int

    def setup(self):
        # Initialize the random Fourier matrix A
        self.ff_layer = nn.Dense(self.num_features, use_bias=False)

    @nn.compact
    def __call__(self, x):

        # Compute the Fourier features using the given input tensor x
        x = self.ff_layer(x)
        fourier_features = jnp.concatenate([jnp.cos(jnp.pi * x),
                                            jnp.sin(jnp.pi * x)], axis=-1)
        return fourier_features


class SimpleMLP(nn.Module):
    in_out_dims: Any
    features: Tuple[int]

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        # self.ff = FourierFeatures(self.features[0])
        self.layers = [nn.Dense(feat)
                       for feat in self.features]  # [1:]
        self.last_layer = nn.Dense(
            self.in_out_dims,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros)
        # kernel_init=jax.nn.initializers.constant(1E-2),
        # bias_init=jax.nn.initializers.constant(1E-2))

    @nn.compact
    def __call__(self, t, samples):
        # add an if statement to add batch dimension
        samples = samples[jnp.newaxis, :]
        # samples = self.ff(samples)
        z = lax.concatenate(
            (t*jnp.ones((samples.shape[0], 1)), samples), 1)

        for i, lyr in enumerate(self.layers):
            z = lyr(z)
            z = nn.tanh(z)
            # z = nn.softplus(1.*z) / \
            # 1.  # if it takes too long remove the 100.

        value = self.last_layer(z)
        return value[0]


class CNFSimpleMLP(nn.Module):
    """Adapted from the Pytorch implementation at:
    https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py
    """
    in_out_dim: Any
    features: Tuple[int]

    def setup(self):
        self.net = SimpleMLP(self.in_out_dim, self.features)

    @nn.compact
    def __call__(self, t, states):
        z, logp_z = states[:, :self.in_out_dim], states[:, self.in_out_dim:]

        def f(z): return self.net(t, z)
        df_dz = vmap(jax.jacrev(f))(z)
        dlogp_z_dt = -1.0 * jnp.trace(df_dz, 0, 1, 2)

        dz = vmap(f)(z)
        return lax.concatenate((dz, dlogp_z_dt[:, None]), 1)


class Gen_CNFSimpleMLP(nn.Module):
    """Negative CNF for jax's odeint."""
    in_out_dim: Any
    features: Tuple[int]
    bool_neg: bool = False

    def setup(self) -> None:
        self.cnf = CNFSimpleMLP(self.in_out_dim, self.features)
        if self.bool_neg:
            self.y0 = -1.
        else:
            self.y0 = 1.

    @nn.compact
    def __call__(self, t, states):
        outputs = self.cnf(self.y0 * t, states)
        return self.y0 * outputs


# @partial(jax.jit,  static_argnums=(2, 3, 4, 5,))
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


@partial(jax.jit,  static_argnums=(2, 3, 4, 5,))
def neural_ode_score(params: Any, batch: Any, f: Callable, t0: float, t1: float, d_dim: int):
    # time as [t1 to t0] gives nans for the second term :/
    start_and_end_time = jnp.array([t0, t1])

    @jax.jit
    def _evol_fn_i(params, t, state):
        state = lax.expand_dims(state, dimensions=(0,))
        state, score = state[:, :-d_dim], state[:, -d_dim:]
        def _f_div(state): return jnp.sum(
            f.apply(params, t, state)[:, -1:])

        def _f_dx(state): return f.apply(params, t, state)[:, :-1]

        div, grad_div = jax.value_and_grad(_f_div)(state[:, :-1])
        dx, _f_vjp = vjp(_f_dx, state[:, :-1])
        score_vjp = _f_vjp(score)[0]
        dscore = -score_vjp+grad_div  # check if the +grad_div
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
        atol=1e-5,
        rtol=1e-5
    )
    z_t, logp_diff_t, score_t = outputs[:, :,
                                        :d_dim], outputs[:, :, d_dim:d_dim+1], outputs[:, :, d_dim+1:]
    # z_t0, logp_diff_t0 = z_t[0], logp_diff_t[0]
    z_t1, logp_diff_t1, score_t1 = z_t[-1], logp_diff_t[-1], score_t[-1]
    # return lax.concatenate((z_t0, z_t1), 2), lax.concatenate((lax.expand_dims(logp_diff_t0, (1,)), lax.expand_dims(logp_diff_t1, (1,))), 1)
    return z_t1, logp_diff_t1, score_t1
    # return outputs


if __name__ == '__main__':
    import jax.random as jrnd
    from jax.tree_util import tree_map
    from jax import value_and_grad, vjp

    import jax.random as jrnd
    import distrax
    from distrax import Normal, MultivariateNormalDiag

    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    in_out_dim = 2

    # dist_prior = Normal(0., 1.)
    mean = jnp.zeros((in_out_dim,))
    cov = jnp.ones((in_out_dim,))
    prior_dist = MultivariateNormalDiag(mean, cov)
    x0 = prior_dist.sample(seed=key, sample_shape=(4,))
    log_prob = prior_dist.log_prob(x0)
    score = vmap(jax.jacrev(lambda x:
                            prior_dist.log_prob(x)))(x0)
    x0 = lax.concatenate((x0, log_prob[:, None], score), 1)
    print(x0)

    # init model
    model_rev = Gen_CNFSimpleMLP(in_out_dim, (512, 512,), bool_neg=False)
    model_fwd = Gen_CNFSimpleMLP(in_out_dim, (512, 512,), bool_neg=True)
    test_inputs = x_test = jnp.ones((1, in_out_dim+1+score.shape[1]))
    t = jnp.array(0.)
    params = model_fwd.init(key, jnp.array(0.), test_inputs)

    @jax.jit
    def NODE_fwd_w_score(params, batch): return neural_ode_score(
        params, batch, model_fwd, 0., 1., in_out_dim)

    # log prob
    # @jax.jit
    def log_prob_wscore(params, samples):
        zt, logp_zt, score = NODE_fwd_w_score(params, samples)
        return logp_zt, (zt, score)

    logp, (x, score) = log_prob_wscore(params, x0)
    print('ODE score')
    print('x:', x)
    print('logp:', logp)
    print('score:', score)

    # assert 0
   # AD SCORE
    print('AD score')

    @jax.jit
    def NODE_fwd(params, batch): return neural_ode(
        params, batch, model_fwd, 0., 1., in_out_dim)

    def log_prob(params, samples):
        zt, logp_zt = NODE_fwd(params, samples[None])
        return jnp.sum(logp_zt), zt.ravel()

    x0 = x0[:, :-in_out_dim]
    logp, x = vmap(log_prob, in_axes=(None, 0))(params, x0)
    # print(logp)
    print('x:', x)
    print('logp:', logp)

    score_ad, _ = vmap(jax.grad(log_prob, argnums=(
        1,), has_aux=True), in_axes=(None, 0))(params, x0)
    print('x:', x)
    print('logp:', logp)
    print('score:', score_ad)
