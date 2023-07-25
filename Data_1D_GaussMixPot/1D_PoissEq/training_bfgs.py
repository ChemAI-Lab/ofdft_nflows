import os
import argparse
import jax, matplotlib,flax,optax
import jax.numpy as jnp
from jax import jit,lax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import flax.linen as nn
from flax.linen import jit as jit_flax 
from flax.training import checkpoints
from jax import value_and_grad, vmap, hessian, grad, jit, random
from functools import partial
from flax.training import checkpoints
from flax.linen.dtypes import promote_dtype

from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

import jaxopt
from jaxopt import BFGS, LBFGS
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

#Define the Neural Network
@jit_flax
class MLP(nn.Module):
    n_layers: int
    n_neurons: int
    act:str = 'tanh'
    out_dims: int = 1
    def setup(self):
        self.act_f = self.act
        print(self.n_layers)
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
    def __call__(self,x):
        for i,lyr in enumerate(self.layers):
            x = lyr(x)
            x = self.f_act(x)
        x = self.last_lyr(x)
        return x
#Define the Neural Network
@jit_flax
class MLPSw(nn.Module):
    n_layers: int
    n_neurons: int
    act:str = 'swish'
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
    def __call__(self,x):
        for i,(lyr,sw) in enumerate(zip(self.layers,self.f_act)):
            x = lyr(x)
            x = sw(x)
        x = self.last_lyr(x)
        return x
@jit
def _normal_dist(x):
    # d = jnp.sqrt(x**2)
    sigma, mu = 1.0, 0
    g = jnp.exp(-((x - mu)**2) / (2.0 * sigma**2))
    #g = 1/(jnp.sqrt(2*jnp.pi*x**3 + 1e-6)) * jnp.exp(-(x-1)**2/(2*x + 1e-6))
    return g
@jit
def mse(y_true, y_pred):
    return jnp.mean(jnp.square(y_true - y_pred))
 
#Define the left side of the equation 
@partial(jit, static_argnums=(2,))
def _laplacian(x:Array,params:Any, model: nn.Module):
    hes_ = hessian(model.apply,argnums=1)(params,x[jnp.newaxis])
    hes_ = jnp.squeeze(hes_,axis=(0,2,4))
    hes_ = jnp.einsum('...ii',hes_)
    return hes_
laplacian = vmap(_laplacian, in_axes=(0,None,None))

def train(act,n_neurons,layers):
    x_min = -10.5
    x_max = 10.5
    CKPT_DIR = f"ckpt_bfgs_{layers}-{n_neurons}_{act}"
    cwd = 'Poisson_Equation'
    CKPT_DIR = os.path.join(cwd, CKPT_DIR)
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    #Y = normal_dist(x0)

    x = jnp.ones((1, 1))
    if not act == 'swish': 
        model = MLP(layers,n_neurons,act)
    else:
        model = MLPSw(layers,n_neurons,act)
    params = model.init(random.PRNGKey(42),x)
    @jit
    def f_loss(params, grid):
        # batch, batch_bc = grid
        grid, grid_bc = grid
        x,y = grid
        d_dx = laplacian(x, params, model)
        l0 = mse(d_dx,y)
	 
        x_bc, y_bc = grid_bc
        y_bc_pred = model.apply(params, x_bc)
        l1 = mse(y_bc_pred,y_bc)
        return l0 + l1
   
    batch_size = 512 
    normal_dist = lambda x:jnp.exp(jax.scipy.stats.norm.logpdf(x,loc=0.,scale=1.))
    x = jnp.linspace(x_min,x_max,batch_size)[:,None]
    y = jax.scipy.stats.norm.logpdf(x,loc=0.,scale=1.)#normal_dist(x)
    y = jnp.exp(y)
    grid = (x,-4*jnp.pi*y)
    x0 = jnp.linspace(-8.5,-1.5,int(batch_size/2))
    x1 = jnp.linspace(1.5,8.5,int(batch_size/2))
    x_bc = jax.lax.concatenate((x0,x1),0)[:,None]
    y_bc = jnp.zeros_like(x)
    grid = (grid,(x_bc,y_bc))    

    max_iter = 3500
    solver = LBFGS(f_loss, maxiter=max_iter, tol=.0001, history_size=25)

    res = solver.run(params, grid=grid)
    params0 = res.params
    print(res.state)
    checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=params, step=0, overwrite=True)
    #Plotting 
    x0 = jnp.linspace(x_min,x_max,512)[:,None]  
    y_nn = model.apply(params0, x0)
    y_nn = y_nn.reshape(x0.shape)
   
    d_dx = laplacian(x0, params0, model)
    _,axs = plt.subplots(1,2)
    axs[0].plot(x0,normal_dist(x0),label=r'$\rho(x)$')
    axs[0].legend()
    axs[1].plot(x0,y_nn,label=r'$V_{H}(x)$')
    axs[1].plot(x0,d_dx,label=r'$\nabla^{2}V_{H}(x)$')
    axs[1].plot(x0,-4*jnp.pi*normal_dist(x0),label=r'$-4\pi\rho(x)$')
    axs[1].legend()
    # plt.plot(x_,y_nn)
    plt.tight_layout()
    plt.savefig(f'nn_poiss_eq_{layers}-{n_neurons}_{act}.png') 

def main():
    parser = argparse.ArgumentParser(description = "SINDy training")

    parser.add_argument("--act",  type = str,   default = 'tanh',   help = "act fun")

    parser.add_argument("--layers",   type = int,   default = 3,   help = "number of layers")

    parser.add_argument("--n_neurons",    type = int,   default = 16,      help = "number of neurons per layer")

    args = parser.parse_args()
    
    act = args.act
    n_neurons = args.n_neurons
    layers = args.layers

    train(act,n_neurons,layers)

if __name__ == "__main__":
    main()
