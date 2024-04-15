from typing import Any, Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap, jacrev, random
from jax.experimental.ode import odeint


import flax
from flax import linen as nn


@jax.custom_jvp
def safe_sqrt(x):
    return jnp.sqrt(x)


@safe_sqrt.defjvp
def safe_sqrt_jvp(primals, tangents):
    x = primals[0]
    x_dot = tangents[0]
    # print(x[0])
    primal_out = safe_sqrt(x)
    tangent_out = 0.5 * x_dot / jnp.where(x > 0, primal_out, jnp.inf)
    return primal_out, tangent_out
    # https://github.com/cagrikymk/JAX-ReaxFF/blob/master/jaxreaxff/forcefield.py


'''
Adaptation of 
Equivariant Flows: sampling configurations for
multi-body systems with symmetric energies
https://ml4physicalsciences.github.io/2019/files/NeurIPS_ML4PS_2019_79.pdf

'''


# @nn.jit
class NN(nn.Module):
    in_out_dims: Any
    features: Tuple[int]

    def setup(self):
        self.layers = [nn.Dense(feat)
                       for feat in self.features]  # [1:]
        self.last_layer = nn.Dense(1,
        kernel_init=jax.nn.initializers.zeros,
        bias_init=jax.nn.initializers.zeros)
        # self.nuclei = self.xyz_nuclei[:,None]

    @nn.compact
    def __call__(self, t, xi_norm, zi_one_hot):

        z = jnp.hstack((t, xi_norm, zi_one_hot))
        # z = xi_norm
        for i, lyr in enumerate(self.layers):
            z = lyr(z)
            z = nn.tanh(z)
        z = self.last_layer(z)
        return z


# @nn.jit
class RadialMLP(nn.Module):
    in_out_dims: Any
    features: Tuple[int]
    xyz_nuclei: Any
    z_one_hot: Any

    def setup(self):
        self.nuclei = self.xyz_nuclei[:, None]
        self.z_ = self.z_one_hot

    @nn.compact
    def __call__(self, t, samples):
        vmap_radialblock = nn.vmap(NN,
                                   variable_axes={'params': None, },
                                   split_rngs={'params': False, },
                                   in_axes=(None, 0, 0))(self.in_out_dims, self.features)

        z = lax.expand_dims(samples, dimensions=(0,)) - self.nuclei
        z_norm = jnp.linalg.norm(z, axis=-1)
        x = vmap_radialblock(t, z_norm, self.z_)
        x = jnp.einsum('ijk,ij->k', z, x)
        return x


class EqvFlow(nn.Module):
    """Equivariant Flows: sampling configurations for 
        multi-body systems with symmetric energies
    """
    in_out_dim: Any
    features: Tuple[int]
    xyz_nuclei: Any
    z_one_hot: Any

    def setup(self):
        self.net = RadialMLP(self.in_out_dim, self.features,
                             self.xyz_nuclei, self.z_one_hot)

    @nn.compact
    def __call__(self, t, states):
        z, logp_z = states[:, :self.in_out_dim], states[:, self.in_out_dim:]

        def f(z): return self.net(t, z)
        df_dz = vmap(jax.jacrev(f))(z)
        dlogp_z_dt = -1.0 * jnp.trace(df_dz, 0, 1, 2)

        dz = vmap(f)(z)
        return lax.concatenate((dz, dlogp_z_dt[:, None]), 1)


class Gen_EqvFlow(nn.Module):
    """Negative CNF for jax's odeint."""
    in_out_dim: Any
    features: Tuple[int]
    xyz_nuclei: Any
    z_one_hot: Any
    bool_neg: bool = False

    def setup(self) -> None:
        self.cnf = EqvFlow(self.in_out_dim, self.features,
                           self.xyz_nuclei, self.z_one_hot)
        if self.bool_neg:
            self.y0 = -1.
        else:
            self.y0 = 1.

    @nn.compact
    def __call__(self, t, states):
        outputs = self.cnf(self.y0 * t, states)
        return self.y0 * outputs


def main():
    import jax
    import jax.random as jrnd
    from jax import grad, value_and_grad
    from promolecular_distrax import ProMolecularDensity
    from jax_ode import neural_ode, neural_ode_score
    from functionals import weizsacker

    d = 3
    mu = jnp.array([[-1., 0., 0.], [1., 0., 0.]])
    # probs = jnp.array([0.75,.25])
    z_atoms = jnp.array([1., 1.])
    # +1 because [0, num_classes) -> 0
    n_atoms_type = jnp.unique(z_atoms).shape[0]+1
    z_one_hot = jax.nn.one_hot(z_atoms, n_atoms_type)

    dist = ProMolecularDensity(z_atoms, mu)

    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)
    z = dist._sample_n(key, 2)
    log_pz = dist.log_prob(z)
    score_z = dist.score(z)
    state = jnp.column_stack((z, log_pz))
    state_wscore = jnp.column_stack((state, score_z))
    print(state_wscore)

    model_fwd = Gen_EqvFlow(d, (64, 64, ),
                            xyz_nuclei=mu, z_one_hot=z_one_hot, bool_neg=True, )
    model_rev = Gen_EqvFlow(d, (64, 64, ),
                            xyz_nuclei=mu, z_one_hot=z_one_hot, bool_neg=False, )

    params = model_fwd.init(key,  jnp.array(0.), state)
    # print(params)

    print('initial', state)
    x, logpx = neural_ode(params, state, model_fwd, 0., 1., d)
    x_logpx = lax.concatenate((x, logpx), 1)
    print('fwd', x_logpx)

    z_tst, log_pz_test = neural_ode(params, x_logpx, model_rev, -1., 0., d)
    z_logpz_tst = lax.concatenate((z_tst, log_pz_test), 1)
    print('rev', z_logpz_tst)
    print('----------------------------------------------------------------')
    # test with a functional
    prior_dist = ProMolecularDensity(z_atoms, mu)
    _, key = jrnd.split(key)

    z = prior_dist._sample_n(key, 512)
    logp_z = prior_dist.log_prob(z)
    score_z = prior_dist.score(z)
    state = jnp.column_stack((z, logp_z))
    state_wscore = jnp.column_stack((state, score_z))

    @jit
    def loss(params, batch):
        x, logp_x, score = neural_ode_score(
            params, batch, model_fwd, 0., 1., d)
        return jnp.mean(weizsacker(jnp.exp(logp_x), score, 1))

    print(jax.value_and_grad(loss)(params, state_wscore))


if __name__ == '__main__':
    main()


# if __name__ == '__main__':

#     import jax.random as jrnd
#     from jax import grad
#     from distrax import MultivariateNormalDiag
#     from functionals import weizsacker

#     png = jrnd.PRNGKey(0)
#     _, key = jrnd.split(png)

#     d = 1
#     prior_dist = MultivariateNormalDiag(
#         jnp.zeros(d), 1.*jnp.ones(d))
#     z = prior_dist._sample_n(key, 2)
#     logp_z = prior_dist.log_prob(z)
#     score_z = vmap(grad(prior_dist.log_prob))(z)
#     state = jnp.column_stack((z, logp_z[:, None]))
#     state_wscore = jnp.column_stack((state, score_z))
#     print(state_wscore)

#     model_fwd = FullODENet(d, (512, 512, 512,), True, True)
#     params = model_fwd.init(key, state_wscore[:1])
#     state_x = model_fwd.apply(params, state_wscore)

#     print(state_x)

#     model_rev = FullODENet(d, (512, 512, 512,), False, True)
#     _ = model_rev.init(key, state_wscore[:1])
#     state_z1 = model_rev.apply(params, state_x)
#     print(state_z1)

#     z = prior_dist._sample_n(key, 512)
#     logp_z = prior_dist.log_prob(z)
#     score_z = vmap(grad(prior_dist.log_prob))(z)
#     state = jnp.column_stack((z, logp_z[:, None]))
#     state_wscore = jnp.column_stack((state, score_z))

#     @jit
#     def loss(params, batch):
#         x_logp_x_score = model_fwd.apply(params, batch)
#         score = x_logp_x_score[:, -d:]
#         logp_x = x_logp_x_score[:, d:d+1]
#         return jnp.mean(weizsacker(jnp.exp(logp_x), score, 1))

#     print(jax.value_and_grad(loss)(params, state_wscore))
