import jax
import jax.numpy as jnp
from jax import vmap

from functools import partial

from sing.sde import SDE

from typing import Tuple, Any

class InputSignals:
    """
    Represent inputs to latent dynamics
    """
    def __init__(self, v):
        """
        Params:
        --------------
        v: shape (n_trials, T, n_inputs) array, known input signals v(t)
        """
        self.v = v

    def update_input_effect(self, fn: SDE, t_grid: jnp.array, marginal_params: Tuple[jnp.array], inputs: jnp.array, gp_post: dict[str, jnp.array], drift_params: dict[str, Any], jitter: float = 1e-4):
        """
        For the model dx(t) = {f(x(t)) + B v(t)} dt + dW(t), computes closed-form update for input effect matrix B.

        Params:
        --------------
        fn: the prior SDE
        t_grid: a shape (T) array, the time grid \tau
        marginal_params: a tuple containing
            - ms: a shape (batch_size, T, D) array, the marginal means E[x_i]
            - Ss: a shape (batch_size, T, D, D) array, the marginal covariances Var(x_i)
            - SSs: a shape (batch_size, T-1, D, D) array, the covariance between consecutive states x_i and x_{i+1}, Cov(x_i, x_{i+1})
        inputs: a shape (batch_size, T, n_inputs) array, the model inouts
        gp_post: for SING-GP, a dictionary containing the parameters of the variational posterior over inducing points, including 
            - q_u_mu: (D, n_inducing)
            - q_u_sigma: (D, n_inducing, n_inducing)
            None if drift is modeled as deterministic 
        drift_params: a dictionary containing the parameters of the prior drift
        jitter: jitter when updating the input effect matrix     
        """
        
        def _int_outer_dynamics_inputs(t_grid, fn, marginal_params, inputs, drift_params, gp_post):
            """Computes sum((m_{t+1} - m{t}) - Delta_t * E[f(x_t)]) v(t)^T)"""
            ms, Ss, _ = marginal_params
            del_t = t_grid[1:] - t_grid[:-1] # (T-1,)
            E_f = vmap(partial(fn.f, drift_params, gp_post=gp_post))(t_grid[:-1], ms[:-1], Ss[:-1]) # (T-1, D)
            ms_diffs = ms[1:] - ms[:-1] # (T-1, D)
            outer = vmap(jnp.outer)(ms_diffs - del_t[...,None] * E_f, inputs[:-1]) # (T-1, D, I)
            return outer.sum(0) # (D, I)

        def _int_outer_inputs(t_grid, inputs):
            """Computes Delta_t * sum(v(t) v(t)^T) from t=0 to t=T-1 (last timestep excluded)."""
            del_t = t_grid[1:] - t_grid[:-1] # (T-1,)
            outer_prod = vmap(jnp.outer)(inputs[:-1], inputs[:-1])
            return (del_t[:,None,None] * outer_prod).sum(0) # (I, I)
        
        n_inputs = inputs.shape[-1]
        outer_inputs_term = vmap(partial(_int_outer_inputs, t_grid))(inputs).sum(0) # (I, I)
        outer_dynamics_inputs_term = vmap(partial(_int_outer_dynamics_inputs, t_grid, fn, drift_params=drift_params, gp_post=gp_post))(marginal_params, inputs).sum(0) # (K, I)
        B = jnp.linalg.solve(outer_inputs_term + jitter * jnp.eye(n_inputs), outer_dynamics_inputs_term.T).T
        return B