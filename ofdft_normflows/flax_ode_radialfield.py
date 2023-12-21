from typing import Any, Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit,lax, vmap, jacrev, random
from jax.experimental.ode import odeint

import flax
import flax
from flax import linen as nn

@jax.custom_jvp
def safe_sqrt(x):
  return jnp.sqrt(x)

@safe_sqrt.defjvp
def safe_sqrt_jvp(primals, tangents):
  x = primals[0]
  x_dot = tangents[0]
  #print(x[0])
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

@nn.jit
class NN(nn.Module):
    in_out_dims: Any
    features: Tuple[int]
        
    def setup(self):
        self.layers = [nn.Dense(feat)
                       for feat in self.features]  # [1:]
        self.last_layer = nn.Dense(1,)
            #  kernel_init=jax.nn.initializers.zeros,
             #bias_init=jax.nn.initializers.zeros)
        # self.nuclei = self.xyz_nuclei[:,None]

    @nn.compact
    def __call__(self, t, xi_norm):
        
        z = jnp.hstack((t,xi_norm))
        # z = xi_norm
        for i, lyr in enumerate(self.layers):
            z = lyr(z)
            z = nn.tanh(z)
        z = self.last_layer(z)
        return z

@nn.jit
class RadialVecField(nn.Module):
    in_out_dims: Any
    features: Tuple[int]
    xyz_nuclei: Any
    
    def setup(self):
        self.nuclei = self.xyz_nuclei[:,None]
    
    @nn.compact
    def __call__(self,t,samples):
        vmap_radialblock = nn.vmap(NN,
                        variable_axes={'params':None,}, 
                        split_rngs={'params': False, },
                        in_axes=(None,0))(self.in_out_dims,self.features)
        
        z = lax.expand_dims(samples,dimensions=(0,)) - self.nuclei
        z_norm = jnp.linalg.norm(z, axis=-1)
        x = vmap_radialblock(t,z_norm)
        x = jnp.einsum('ijk,ij->k',z,x)
        return x
        
            
    
@nn.jit
class ODEBlock(nn.Module):
    
    """ODE block which contains odeint"""
    in_out_dims: Any
    features: Tuple[int]
    bool_neg: Any = True #True(Fwd dynamics), False (reverse dynamics)
    xyz_nuclei: Any = None
    tol: Any = 1E-6

    @nn.compact
    def __call__(self, states, params):
        ode_fun = RadialVecField(self.in_out_dims, self.features, self.xyz_nuclei)
        if self.bool_neg:
            t0 = 1
            t_grid = jnp.array([0.,1.])
        else:
            t0 = -1
            t_grid = jnp.array([-1.,.0])

        # f_ode = lambda params, x, t: t0*ode_fun.apply(params,t0*t,x)
        @jit
        def f_ode(params,states,t):
            x,logp_x = states[:self.in_out_dims], states[self.in_out_dims:]
            def f(x): return ode_fun.apply(params,t0*t,x)
            dz = f(x)
            df_dz = jacrev(f)(x)
            dlogp_z_dt = -1. * jnp.trace(df_dz)#, 0, 0, 1)
            # return lax.concatenate((lax.expand_dims(dz,dimensions=(1,)), dlogp_z_dt), 1)#self.t0
            return t0*jnp.append(dz,dlogp_z_dt)
            
        _, final_state = odeint(partial(f_ode, {'params': params}),
                                         states, t_grid,
                                         rtol=self.tol, atol=self.tol)

        return final_state

@nn.jit
class ODEBlockwScore(nn.Module):
    """ODE block which contains odeint"""
    in_out_dims: Any
    features: Tuple[int]
    bool_neg: Any = True #True(Fwd dynamics), False (reverse dynamics)
    xyz_nuclei: Any = None
    tol: Any = 1E-5

    @nn.compact
    def __call__(self, states, params):
        ode_fun = RadialVecField(self.in_out_dims, self.features, self.xyz_nuclei)
        if self.bool_neg:
          t0 = 1
          t_grid = jnp.array([0.,1.])
        else:
          t0 = -1
          t_grid = jnp.array([-1.,0.])

        # f_ode = lambda params, x, t: t0*ode_fun.apply(params,t0*t,x)
        @jit
        def _f_ode(params,states,t):
            x,logp_x = states[:self.in_out_dims], states[self.in_out_dims:]
            def f(x): return ode_fun.apply(params,t,x)
            dz = f(x)
            df_dz = jacrev(f)(x)
            dlogp_z_dt = -1. * jnp.trace(df_dz)#, 0, 0, 1)
            # return lax.concatenate((lax.expand_dims(dz,dimensions=(1,)), dlogp_z_dt), 1)#self.t0
            return dz,dlogp_z_dt

        @jit
        def f_ode(params,state,t):
            state, score = state[:-self.in_out_dims], state[-self.in_out_dims:]
            dx_and_dlopz, _f_vjp = jax.vjp(lambda state: _f_ode(params,state,t0*t), state)
            dx, dlopz = dx_and_dlopz
            (vjp_all,) = _f_vjp((score,1.))
            score_vjp,grad_div = vjp_all[:-1], vjp_all[-1]
            dscore = -score_vjp + grad_div    
            return t0*jnp.append(jnp.append(dx, dlopz),dscore)
            
        _, final_state = odeint(partial(f_ode, {'params': params}),
                                         states, t_grid,
                                         rtol=self.tol, atol=self.tol)

        return final_state

   
@nn.jit
class ODEBlockVmap(nn.Module):
    """Apply vmap to ODEBlock"""
    in_out_dims: Any
    features: Tuple[int]
    bool_neg: Any = True #True(Fwd dynamics), False (reverse dynamics)
    bool_wscore: bool = False
    xyz_nuclei: Any = None
    tol: Any = 1E-5
  
    @nn.compact
    def __call__(self, x, params):
        if self.bool_wscore:
            vmap_odeblock = nn.vmap(ODEBlockwScore,
                                    variable_axes={'params': 0, 'nfe': None},
                                    split_rngs={'params': True, 'nfe': False},
                                    in_axes=(0, None))        
        elif self.bool_wscore == False:
            vmap_odeblock = nn.vmap(ODEBlock,
                                    variable_axes={'params': 0, 'nfe': None},
                                    split_rngs={'params': True, 'nfe': False},
                                    in_axes=(0, None))

        return vmap_odeblock(in_out_dims=self.in_out_dims,
                             features=self.features,
                             bool_neg=self.bool_neg,xyz_nuclei=self.xyz_nuclei,
                            tol=self.tol, name='odeblock')(x, params)
        
@nn.jit
class FullODENet(nn.Module):
    in_out_dims: Any
    features: Tuple[int]
    bool_neg: Any = True #True(Fwd dynamics), False (reverse dynamics)
    bool_wscore: bool = False
    xyz_nuclei: Any = None
    tol: Any = 1E-5

    @nn.compact
    def __call__(self, inputs):
        # ode_func = SimpleVecField(self.in_out_dims,self.features)
        ode_func = RadialVecField(self.in_out_dims,self.features,self.xyz_nuclei)
        init_fn = lambda rng, x: ode_func.init(random.split(rng)[-1], 0.,jnp.ones((1)))['params']
        ode_func_params = self.param('ode_func', init_fn, jnp.ones(self.in_out_dims))
        x = ODEBlockVmap(in_out_dims=self.in_out_dims,
                            features=self.features,
                            bool_neg=self.bool_neg,bool_wscore=self.bool_wscore,
                            xyz_nuclei=self.xyz_nuclei,
                            tol=self.tol)(inputs, ode_func_params)
        return x

if __name__ == '__main__':

    import jax.random as jrnd
    from jax import grad
    from distrax import MultivariateNormalDiag
    from functionals import weizsacker
    
    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)
    
    xyz = jnp.array([[0., 0., -1.4008538753/2], [0., 0., 1.4008538753/2]])

    # fwd and rev test
    dimension = 3
    prior_dist = MultivariateNormalDiag(jnp.zeros(dimension), 1.*jnp.ones(dimension))
    z = prior_dist._sample_n(key,2)
    logp_z = prior_dist.log_prob(z)
    score_z = vmap(grad(prior_dist.log_prob))(z)
    state = jnp.column_stack((z,logp_z[:,None]))
    state_wscore = jnp.column_stack((state,score_z))
    print(state_wscore)
    print('---')
    
    model_fwd = FullODENet(dimension,(512,512,512,),True,True,xyz_nuclei=xyz)
    params = model_fwd.init(key,state_wscore[:1])
    state_x = model_fwd.apply(params,state_wscore)
    
    print(state_x)
    
    model_rev = FullODENet(dimension,(512,512,512,),False,True,xyz_nuclei=xyz)
    _ = model_rev.init(key,state_wscore[:1])
    # _ = model_rev.init(key,state[:1])
    state_z1 = model_rev.apply(params,state_x)
    print(state_z1)
    # assert 0

    # grad test
    z = prior_dist._sample_n(key,128)
    logp_z = prior_dist.log_prob(z)
    score_z = vmap(grad(prior_dist.log_prob))(z)
    state = jnp.column_stack((z,logp_z[:,None]))
    state_wscore = jnp.column_stack((state,score_z))
    
    @jit 
    def loss(params,batch):
        x_logp_x_score = model_fwd.apply(params,batch)
        score = x_logp_x_score[:,-dimension:]
        logp_x = x_logp_x_score[:,dimension:dimension+1]
        return jnp.mean(weizsacker(jnp.exp(logp_x),score, 1))
    
    
    print(loss(params,state_wscore))
    print(jax.value_and_grad(loss)(params,state_wscore))
    
    assert 0
    
    
    
    
    
    
    