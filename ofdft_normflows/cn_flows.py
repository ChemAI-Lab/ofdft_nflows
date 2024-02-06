import jax
from typing import Any, Callable, Tuple
from jax import lax, vmap, vjp
from jax import numpy as jnp 
from jax.experimental.ode import odeint
import jax.random as jrnd

import flax
from flax import linen as nn


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
        self.layers = [nn.Dense(feat)
                       for feat in self.features]  # [1:]
        self.last_layer = nn.Dense(
            self.in_out_dims,
        kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros)    

    @nn.compact
    def __call__(self, t, samples):
        
        samples = samples[jnp.newaxis, :]
        
        z = lax.concatenate(
            (t*jnp.ones((samples.shape[0], 1)), samples), 1)

        for i, lyr in enumerate(self.layers):
            z = lyr(z)
            z = nn.tanh(z)

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

