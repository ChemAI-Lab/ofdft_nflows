from typing import Any, Callable
from functools import partial

import jax
from jax import grad
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

    Parameters
    ----------
    params : Any
        _description_
    X : Array
        _description_
    fun : callable
        _description_

    Returns
    -------
    jax.Array
        _description_
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
    """
    Function that evaluates the score of the model. Using jax.jacrev to compute the gradient of the function
    and then reshaping the output to the original shape of the input and then vmapping the function 
    to evaluate the score for each element in the input.

    Parameters
    ----------
    params : Any
        Parameters of the model.
    X : Array
        X values to evaluate the score.
    fun : callable
        Function to evaluate the score.

    Returns
    -------
    jax.Array
        The score of the model.
    """    
    @jit
    def _score(params: Any, xi: Array):
        score_ = jax.jacrev(fun, argnums=1)(params, xi[jnp.newaxis])
        return jnp.reshape(score_, xi.shape[0])

    v_score = vmap(_score, in_axes=(None, 0))
    return v_score(params, X)

def batch_generator(key: prng.PRNGKeyArray, batch_size: int, prior_dist: Callable):
    """
    Generator that yields batches of samples from the prior distribution.

    Parameters
    ----------
    key : prng.PRNGKeyArray
        Key to generate random numbers.
    batch_size : int
        Size of the batch.
    prior_dist : Callable
        Prior distribution.

    """    
    
    v_score = jax.vmap(jax.grad(lambda x:
                              prior_dist.log_prob(x).sum()))
    while True:
        _, key = jrnd.split(key)
        samples = prior_dist.sample(seed=key, sample_shape=batch_size)
        logp_samples = prior_dist.log_prob(samples)
        score = v_score(samples)
        samples0 = lax.concatenate(
            (samples, logp_samples, score), 1)

        _, key = jrnd.split(key)
        samples = prior_dist.sample(seed=key, sample_shape=batch_size)
        logp_samples = prior_dist.log_prob(samples)
        score = v_score(samples)
        samples1 = lax.concatenate(
            (samples, logp_samples, score), 1)

        yield lax.concatenate((samples0, samples1), 0)

def batche_generator_1D(key: prng.PRNGKeyArray, batch_size: int, prior_dist: Callable):
    """
    Generator that yields batches of samples from the prior distribution.

    Parameters
    ----------
    key : prng.PRNGKeyArray
        Key to generate random numbers.
    batch_size : int
        Size of the batch.
    prior_dist : Callable
        Prior distribution.

    """    
    v_score = vmap(jax.jacrev(lambda x:
                              prior_dist.log_prob(x)))
    # v_score = jax.vmap(jax.grad(lambda x:
    #                           prior_dist.log_prob(x).sum()))
    while True:
        _, key = jrnd.split(key)
        samples = prior_dist.sample(seed=key, sample_shape=batch_size)
        logp_samples = prior_dist.log_prob(samples)
        score = v_score(samples)
        samples0 = lax.concatenate(
            (samples, logp_samples[:,None], score), 1)

        _, key = jrnd.split(key)
        samples = prior_dist.sample(seed=key, sample_shape=batch_size)
        logp_samples = prior_dist.log_prob(samples)
        score = v_score(samples)
        samples1 = lax.concatenate(
            (samples, logp_samples[:,None], score), 1)

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
        
def correlation_polarization_correction(
    e_tilde_PF: float, 
    den: Array, 
    clip_cte: float = 1e-30
):
    r"""Spin polarization correction to a correlation functional using eq 2.75 from
    Carsten A. Ullrich, "Time-Dependent Density-Functional Theory".

    Parameters
    ----------
    e_tilde_PF: Float[Array, "spin grid"]
        The paramagnetic/ferromagnetic energy contributions on the grid, to be combined.

    rho: Float[Array, "spin grid"]
        The electronic density of each spin polarization at each grid point.

    clip_cte:
        float, defaults to 1e-30
        Small constant to avoid numerical issues when dividing by rho.

    Returns
    ----------
    e_tilde: Float[Array, "grid"]
        The ready to be integrated electronic energy density.
    """

    log_rho = jnp.log2(jnp.clip(den.sum(axis=1), a_min=clip_cte))
    # assert not jnp.isnan(log_rho).any() and not jnp.isinf(log_rho).any()
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_rho / 3.0

    zeta = jnp.where(den.sum(axis=1) > clip_cte, (den[:, 0] - den[:, 1]) / (den.sum(axis=1)), 0.0)

    def fzeta(z):
        zm = 2 ** (4 * jnp.log2(1 - z) / 3)
        zp = 2 ** (4 * jnp.log2(1 + z) / 3)
        return (zm + zp - 2) / (2 * (2 ** (1 / 3) - 1))

    A_ = 0.016887
    alpha1 = 0.11125
    beta1 = 10.357
    beta2 = 3.6231
    beta3 = 0.88026
    beta4 = 0.49671

    ars = 2 ** (jnp.log2(alpha1) + log_rs)
    brs_1_2 = 2 ** (jnp.log2(beta1) + log_rs / 2)
    brs = 2 ** (jnp.log2(beta2) + log_rs)
    brs_3_2 = 2 ** (jnp.log2(beta3) + 3 * log_rs / 2)
    brs2 = 2 ** (jnp.log2(beta4) + 2 * log_rs)

    alphac = 2 * A_ * (1 + ars) * jnp.log(1 + (1 / (2 * A_)) / (brs_1_2 + brs + brs_3_2 + brs2))
    # assert not jnp.isnan(alphac).any() and not jnp.isinf(alphac).any()

    fz = fzeta(zeta) #jnp.round(fzeta(zeta), int(math.log10(clip_cte)))
    z4 = zeta**4 #jnp.round(2 ** (4 * jnp.log2(jnp.clip(zeta, a_min=clip_cte))), int(math.log10(clip_cte)))

    e_tilde = (
        e_tilde_PF[:, 0]
        + alphac * (fz / (grad(grad(fzeta))(0.0))) * (1 - z4)
        + (e_tilde_PF[:, 1] - e_tilde_PF[:, 0]) * fz * z4
    )
    # assert not jnp.isnan(e_tilde).any() and not jnp.isinf(e_tilde).any()

    return e_tilde

def one_hot_encode(z: Array) -> Array:
    """
    One hot encode the input array.

    Parameters
    ----------
    z : Array
        Array to one hot encode.

    Returns
    -------
    Array
        One hot encoded array.
    """
    z_u = jnp.unique(z)
    zz = []
    for zi in z:
        for i,zui in enumerate(z_u):
            if zui == zi:
                zz.append(i)
    n_atoms_type = jnp.unique(z).shape[0]
    z_one_hot = jax.nn.one_hot(zz,n_atoms_type)    
    
    return z_one_hot 

def coordinates(mol_name: str, BOHR: float = 1.8897259886 ) -> Array:
    
    if mol_name == 'H2':
        Ne = 2
        atoms = ['H', 'H']
        coords = jnp.array([[0., 0., -1.4008538753/2], 
                          [0., 0., 1.4008538753/2]])*BOHR
        z = jnp.array([1, 1])
        return Ne,atoms,z,coords
    elif mol_name == 'LiH':
        Ne = 4 
        atoms = ['Li', 'H']
        coords = jnp.array([[0., 0., -1.5949/2], 
                          [0., 0., 1.5949/2]])*BOHR 
        z = jnp.array([3, 1])
        return Ne,atoms,z,coords
    elif mol_name == 'H2O':
        Ne = 10
        atoms = ['O', 'H', 'H']
        coords = jnp.array([[0.0,	0.0,	0.1189120],
                        [0.0,	0.7612710,	-0.4756480],
                        [0.0,	-0.7612710,	-0.4756480]])*BOHR
        z = jnp.array([8, 1, 1])
        return Ne,atoms,z,coords
    elif mol_name == 'CH4':
        Ne = 10
        atoms = ['C', 'H', 'H', 'H', 'H']
        coords = jnp.array([[0.0,	0.0,	0.0],
                        [0.0,	0.0,	1.09],
                        [1.03,	0.0,	-0.363],
                        [-0.515,	-0.889165,	-0.363],
                        [-0.515,	0.889165,	-0.363]])*BOHR
        z = jnp.array([6, 1, 1, 1, 1])
        return Ne,atoms,z,coords
    elif mol_name == 'C6H6':
        Ne = 42
        atoms = ['C','C','C','C','C','C','H','H','H','H','H','H']
        coords = jnp.array([[-0.6984022192, 1.2096794375, 0.0001298085],
                        [0.6983971652, 1.2096794375, 0.0001298085],
                        [1.3968148123, 0.0000100819, 0.0001298085],
                        [0.6983970907, -1.2096631450, -0.0001577731],
                        [-0.6984347101, -1.2097236297, -0.0001117213],
                        [-1.3967427862, -0.0000316846, -0.0000531302],
                        [2.4989957176, 0.0000352657, 0.0001433155],
                        [1.2492115488, 2.1643070022, 0.0000608098],
                        [-1.2497352350, 2.1640355443, 0.0000873078],
                        [1.2495192781, -2.1641211865, -0.0004652006],
                        [-1.2494488074, -2.1642720181, -0.0003281198],
                        [-2.4988922798, 0.0006052813, -0.0002941369]])*BOHR
        z = jnp.array([6.,6.,6.,6.,6.,6.,1.,1.,1.,1.,1.,1.])
        return Ne,atoms,z,coords
    elif mol_name == 'C27H46O': 
        Ne = 216
        atoms = ['O','C','C','C','C','C','C','C','C','C','C',
             'C','C','C','C','C','C','C','C','C','C','C',
             'C','C','C','C','C','C','H','H','H','H','H',
             'H','H','H','H','H','H','H','H','H','H','H',
             'H','H','H','H','H','H','H','H','H','H','H',
             'H','H','H','H','H','H','H','H','H','H','H',
             'H','H','H','H','H','H','H','H']
        coords = jnp.array([[-7.6344919379, 0.4597041902, 0.7112354681],
                        [1.1080808630, 0.4597041902, 0.7112354681],
                        [0.2530014327, -0.8231352339, 0.7112354681],
                        [-1.1599473252, -0.6092757486, 1.2632436184],
                        [-1.8846097300, 0.4851364826, 0.4266238676],
                        [2.4346065739, -0.1264931117, 0.1695905830],
                        [0.4237780146, 1.4699206954, -0.2202286076],
                        [-3.3372747943, 0.7709975410, 0.9822345823],
                        [1.1534637485, -1.8591349538, 1.3772546332],
                        [-1.0194512849, 1.7601151269, 0.2271214576],
                        [2.5719892229, -1.4735213405, 0.9177695847],
                        [-1.9616876023, -1.9189024556, 1.2281160480],
                        [3.6706406723, 0.7518022510, 0.2998092579],
                        [1.3056592864, 1.0858012443, 2.1142034561],
                        [-4.0659099987, -0.5409527880, 1.3058897583],
                        [-4.1355207853, 1.5407857692, -0.1122402103],
                        [-3.4345565189, -1.7244824608, 1.3955388362],
                        [-3.2883872448, 1.6217121992, 2.2742973586],
                        [-5.5600747780, -0.4657287448, 1.5202105354],
                        [-5.6342682740, 1.6532127465, 0.1635040190],
                        [4.9369027993, 0.0920796760, -0.2857409192],
                        [-6.2593559003, 0.2808578004, 0.3875037320],
                        [3.4346199063, 2.0977159171, -0.3972787505],
                        [4.8597222067, -0.3194422440, -1.7619595329],
                        [6.1183294840, -1.0351005847, -2.2632694976],
                        [6.0683625521, -1.3844022946, -3.7635312418],
                        [5.8475871083, -0.1546334232, -4.6458023942],
                        [4.9580276609, -2.4067800912, -4.0202930776],
                        [0.1320531554, -1.1546414208, -0.3341704111],
                        [-1.0961065494, -0.2887741191, 2.3091814564],
                        [-2.0071557590, 0.0577891770, -0.5822428933],
                        [2.2692308941, -0.3566230816, -0.8919397659],
                        [0.4024909308, 1.0868866903, -1.2485637383],
                        [0.9326111775, 2.4332063069, -0.2550387088],
                        [0.9000265464, -2.8793549884, 1.0717133503],
                        [1.0798371703, -1.8121549020, 2.4694535032],
                        [-0.9810389048, 2.3462405294, 1.1499104917],
                        [-1.4838623070, 2.4059394028, -0.5267205531],
                        [3.2317842936, -1.3989125509, 1.7896882295],
                        [2.9883545370, -2.2408492672, 0.2555163717],
                        [-1.6014174036, -2.5802367118, 2.0254337925],
                        [-1.7980739690, -2.4474317007, 0.2806843164],
                        [3.8879474770, 0.9526354546, 1.3563558574],
                        [1.8492781899, 2.0336135308, 2.0610078045],
                        [0.3578137484, 1.3050839833, 2.6129243364],
                        [1.8665627197, 0.4298627984, 2.7876605918],
                        [-4.0090347972, 1.0378379064, -1.0807548937],
                        [-3.7338443144, 2.5534931648, -0.2361852128],
                        [-4.0082696014, -2.6236692902, 1.6068637872],
                        [-2.9172441072, 2.6329927094, 2.0790411399],
                        [-4.2766956463, 1.7447530584, 2.7297022229],
                        [-2.6449795846, 1.1683767430, 3.0353862870],
                        [-6.0051645049, -1.4649829974, 1.6125803553],
                        [-5.7715116059, 0.0256833216, 2.4782609823],
                        [-5.8206294845, 2.3008198064, 1.0291274274],
                        [-6.1301728674, 2.1483087541, -0.6807530020],
                        [5.7878953777, 0.7733743373, -0.1540084782],
                        [5.1800431538, -0.7964447756, 0.3105080195],
                        [-6.2070878208, -0.3024479474, -0.5399782530],
                        [2.9407650699, 1.9873110720, -1.3676590681],
                        [4.3895302067, 2.6050134027, -0.5818686450],
                        [2.8705286939, 2.7872275940, 0.2335427866],
                        [4.0058535847, -0.9856353615, -1.9055309667],
                        [4.6772717683, 0.5793866627, -2.3581606350],
                        [-8.0203071775, -0.4225513077, 0.8470435952],
                        [6.2790374269, -1.9486777435, -1.6774593986],
                        [6.9885972888, -0.3917002684, -2.0822296102],
                        [7.0293119718, -1.8396620422, -4.0335362510],
                        [6.5721859735, 0.6309863795, -4.4078721938],
                        [5.9798893020, -0.4194041327, -5.7008780697],
                        [4.8401657335, 0.2599872029, -4.5419245327],
                        [5.0564586193, -3.2709821863, -3.3550073831],
                        [5.0182590552, -2.7761853866, -5.0502175442],
                        [3.9582362826, -1.9813780480, -3.8886221559]])*BOHR
  
        z = jnp.array([8.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,
                    6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,
                    1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                    1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                    1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                    1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                    1.,1.,1.,1.,1.,1.])
        return Ne,atoms,z,coords


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
