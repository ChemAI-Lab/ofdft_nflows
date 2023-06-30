# ofdft_normflows
Normalizing flows for orbital-free DFT


Ricky tips:
sample from a Gaussian and try to max the log rho for samples under a gaussian to check normalization

CNF in Flax
https://huggingface.co/flax-community/NeuralODE_SDE/blob/main/train_cnf.py


correct definition of the CNFs

'''python

    model_rev = CNF(2, (200, 200,), bool_neg=False)
    model_fwd = CNF(2, (200, 200,), bool_neg=True)
    # model_rev = Gen_CNFRicky(2, bool_neg=False)
    # model_fwd = Gen_CNFRicky(2, bool_neg=True)
    test_inputs = lax.concatenate((jnp.ones((1, 2)), jnp.ones((1, 1))), 1)
    params = model_rev.init(key, jnp.array(0.), test_inputs)

    @jax.jit
    def NODE_rev(params, batch): return neural_ode(
        params, batch, model_rev, -10., 0., 2)

    @jax.jit
    def NODE_fwd(params, batch): return neural_ode(
        params, batch, model_fwd, 0., 10., 2)
'''