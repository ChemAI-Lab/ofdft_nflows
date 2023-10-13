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


class MLP(nn.Module):
    features: Sequence[int]

    def setup(self) -> None:
        self.layers = [
            # nn.Dense(feat, kernel_init=normal_init, bias_init=normal_init)
            nn.Dense(feat)
            for feat in self.features
        ]
        self.act = nn.tanh

    @nn.compact
    def __call__(self, t, state) -> Any:

        z = lax.concatenate((state, lax.expand_dims(t, (0, 1))), 1)
        for i, lyr in enumerate(self.layers[:-1]):
            z = lyr(z)
            z = self.act(z)  # nn.silu(z)  # jnp.tanh(z)  # nn.sigmoid(z)
        y = self.layers[-1](z)
        return y


# class ODENET(nn.Module):
#     features:Sequence[int]
#     t0: float = 1.
#     t1: float = 0.

#     def setup(self) -> None:
#         self.start_and_end_time = jnp.array([self.t0, self.t1])
#         self.nn_dynamics = MLP(self.features)

from flax.core import lift
from flax.core import Scope, init, apply, nn as core_nn

def lift_ode(fn,):
    def odeint_wrapper():
        return odeint(func,y0,t,*args)
    
    def lift.pack(
        
    )
        


def main():
    import jax.random as jrnd

    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    mlp = MLP([10, 1])
    x = jnp.ones((1, 1))
    t = jnp.array(1.)
    # x_and_t = lax.concatenate((x, t), 1)
    params = mlp.init(key, t, x)
    print(params)


if __name__ == '__main__':
    main()
