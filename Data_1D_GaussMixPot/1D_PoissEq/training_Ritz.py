import os
import argparse
from functools import partial
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)
import jax
import jax.numpy as jnp
from jax import jit, lax, value_and_grad, vmap, hessian, jit
import jax.random as jrnd

import flax.linen as nn
from flax.linen import jit as jit_flax
from flax.training import checkpoints
from flax.training import checkpoints
from flax.linen.dtypes import promote_dtype
from jax._src import prng

import optax

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

NGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]


@jit_flax
class SwishVec(nn.Module):
    """SiLU function with a SINGLE learnable parameter per feature

    Args:
        nn (_type_): _description_
    """
    features: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        kernel = self.param('beta',
                            lambda rng, shape, dtype: jnp.ones(
                                shape, dtype=dtype),
                            (self.features,),
                            self.param_dtype)
        inputs, kernel = promote_dtype(inputs, kernel, dtype=self.dtype)
        y = vmap(lambda x, b: x*b, in_axes=(0, None))(inputs, kernel)
        y = nn.sigmoid(y)
        y = vmap(lambda x, y: x*y, in_axes=(0, 0))(inputs, y)
        return y

# Define the Neural Network


@jit_flax
class MLP(nn.Module):
    n_layers: int
    n_neurons: int
    act: str = 'tanh'
    out_dims: int = 1

    def setup(self):
        self.act_f = self.act
        self.layers_ = [self.n_neurons for i in range(self.n_layers)]
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(i)
                       for i in self.layers_]
        if self.act_f == 'tanh':
            self.f_act = nn.tanh
        elif self.act_f == 'sigmoid':
            self.f_act = nn.sigmoid
        elif self.act_f == 'softplus':
            self.f_act = nn.softplus
        elif self.act_f == 'gelu':
            self.f_act = nn.gelu
        self.last_lyr = nn.Dense(self.out_dims)

    @nn.compact
    def __call__(self, x):
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            x = self.f_act(x)
        x = self.last_lyr(x)
        return x
# Define the Neural Network


@jit_flax
class MLPSw(nn.Module):
    n_layers: int
    n_neurons: int
    act: str = 'swish'
    out_dims: int = 1

    def setup(self):
        self.act_f = self.act
        self.layers_ = [self.n_neurons for i in range(self.n_layers)]
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(i)
                       for i in self.layers_]
        self.f_act = [SwishVec(self.n_neurons)
                      for i in range(self.n_layers)]
        self.last_lyr = nn.Dense(self.out_dims)

    @nn.compact
    def __call__(self, x):
        z = x
        for i, (lyr, sw) in enumerate(zip(self.layers, self.f_act)):
            x = lyr(x)
            x = z + sw(x)
            z = x
        x = self.last_lyr(x)
        return x


@jit
def _normal_dist(x):
    # d = jnp.sqrt(x**2)
    sigma, mu = 1.0, 0
    g = jnp.exp(-((x - mu)**2) / (2.0 * sigma**2))
    # g = 1/(jnp.sqrt(2*jnp.pi*x**3 + 1e-6)) * jnp.exp(-(x-1)**2/(2*x + 1e-6))
    return g


@jit
def mse(y_true, y_pred):
    return jnp.linalg.norm(y_true - y_pred)
    # return jnp.mean(jnp.square(y_true - y_pred))

# Define the left side of the equation


@partial(jit, static_argnums=(2,))
def _laplacian(x: Array, params: Any, model: nn.Module):
    hes_ = hessian(model.apply, argnums=1)(params, x[jnp.newaxis])
    hes_ = jnp.squeeze(hes_, axis=(0, 2, 4))
    hes_ = jnp.einsum('...ii', hes_)
    return hes_


laplacian = vmap(_laplacian, in_axes=(0, None, None))


@partial(jit, static_argnums=(2,))
def _force(x: Array, params: Any, model: nn.Module):
    hes_ = jax.jacrev(model.apply, argnums=1)(params, x[jnp.newaxis])
    return hes_


force = vmap(_force, in_axes=(0, None, None))


def train(act, n_neurons, layers,
          epochs: int = 2000, batch_size: int = 512,
          bool_load_params: bool = False):
    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    x_min = -10.5
    x_max = 10.5
    CKPT_DIR = f"ckpt_adam_{layers}-{n_neurons}_{act}"
    cwd = 'Poisson_Equation_Ritz'
    CKPT_DIR = os.path.join(cwd, CKPT_DIR)
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    # Y = normal_dist(x0)

    x = jnp.ones((1, 1))
    if not act == 'swish':
        model = MLP(layers, n_neurons, act)
    else:
        model = MLPSw(layers, n_neurons, act)
    params = model.init(key, x)

    if bool_load_params:
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=CKPT_DIR, target=params, step=0)
        params = restored_state

    @jit
    def normal_dist(x): return jnp.exp(
        jax.scipy.stats.norm.logpdf(x, loc=0., scale=1.))

    @jit
    def f_loss(params, grid):
        # batch, batch_bc = grid
        grid, grid_bc = grid
        x, y = grid
        # d_dx = laplacian(x, params, model)
        # d_dx = vmap(jax.hessian(model.apply, argnums=(1)),
        # in_axes=(None, 0))(params, x)
        d_dx = force(x, params, model)
        d_dx = jnp.reshape(d_dx, y.shape)
        d_dx = 0.5*d_dx*d_dx
        # l0 = mse(d_dx, y)
        l0 = d_dx - (y*model.apply(params, x))

        x_bc, y_bc = grid_bc
        y_bc_pred = model.apply(params, x_bc)
        # l1 = mse(y_bc_pred, y_bc)
        l1 = y_bc_pred**2  # - y_bc
        # + jnp.trapz(l1.ravel(), x_bc.ravel())
        return jnp.mean(l0)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=3E-4),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(
            f_loss, has_aux=False)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    def batches_generator(key: prng.PRNGKeyArray, batch_size: int):
        while True:
            _, key = jrnd.split(key)
            # x0 = 1.5*jrnd.normal(key, shape=(int(batch_size/2),))
            # x1 = jrnd.uniform(key, shape=(int(batch_size/2), ),
            #                   minval=x_min, maxval=x_max)
            # x = lax.concatenate((x0, x1), 0)[:, None]
            x = jnp.linspace(x_min, x_max, batch_size)[:, None]
            y = -4*jnp.pi*normal_dist(x)  # check sign

            # _, key = jrnd.split(key)
            # x0_bc = jrnd.uniform(key, shape=(
            #     int(batch_size/2), 1), minval=x_min, maxval=-5.)
            # _, key = jrnd.split(key)
            # x1_bc = jrnd.uniform(key, shape=(
            #     int(batch_size/2), 1), minval=5., maxval=x_max)
            # x01_bc = lax.concatenate((x0_bc, x1_bc), 0)[:, None]
            x0_bc = jnp.linspace(x_min, -5., int(batch_size/2))[:, None]
            x1_bc = jnp.linspace(5., x_max, int(batch_size/2))[:, None]
            x01_bc = lax.concatenate((x0_bc, x1_bc), 0)
            y_bc = jnp.zeros_like(x01_bc)

            yield ((x, y), (x01_bc, y_bc))

    gen_batches = batches_generator(key, batch_size)
    loss0 = jnp.inf
    ema = optax.ema(decay=0.99)
    ema_state = ema.init(0.)
    for i in range(epochs+1):
        batch = next(gen_batches)
        params, opt_state, loss_value = step(params, opt_state, batch)
        ema_updates, ema_state = ema.update(loss_value, ema_state)
        if i % 100 == 0:
            print(f'{i}: loss: {loss_value:.4f}', ema_state.ema)

            # assert 0

        if loss_value < loss0:
            params0 = params
            loss0 = loss_value
            checkpoints.save_checkpoint(
                ckpt_dir=CKPT_DIR, target=params, step=0, overwrite=True)

    # Plotting
    x0 = jnp.linspace(x_min, x_max, 1024)[:, None]
    y_nn = model.apply(params0, x0)
    y_nn = y_nn.reshape(x0.shape)

    d_dx = laplacian(x0, params0, model)
    _, axs = plt.subplots(1, 2)
    axs[0].plot(x0, normal_dist(x0), label=r'$\rho(x)$')
    axs[0].legend()
    axs[1].plot(x0, y_nn, label=r'$V_{H}(x)$')
    axs[1].plot(x0, d_dx, label=r'$\nabla^{2}V_{H}(x)$')
    axs[1].plot(x0, -4*jnp.pi*normal_dist(x0), label=r'$-4\pi\rho(x)$')
    axs[1].legend()
    # plt.plot(x_,y_nn)
    plt.tight_layout()
    plt.savefig(f'nn_poiss_eq_{layers}-{n_neurons}_{act}.png')


def main():
    parser = argparse.ArgumentParser(description="SINDy training")

    parser.add_argument("--act",  type=str,   default='tanh',   help="act fun")

    parser.add_argument("--layers",   type=int,
                        default=8,   help="number of layers")

    parser.add_argument("--n_neurons",    type=int,
                        default=64,      help="number of neurons per layer")

    args = parser.parse_args()

    act = args.act
    n_neurons = args.n_neurons
    layers = args.layers

    train(act, n_neurons, layers)


if __name__ == "__main__":
    main()
