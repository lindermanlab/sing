"""
Contains helper functions specific to the SING variational inference algorithm
See sing/sing.py
"""

import jax
import jax.numpy as jnp
from jax import vmap, lax, value_and_grad
from jax.scipy.linalg import solve_triangular

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from sing.sde import SDE

from functools import partial
from typing import Tuple, NamedTuple, Any

# --------------------- Functions for parameter conversion (parallelized) -------------------
class GaussianPotential(NamedTuple):
  """
  Represents a Gaussian potential phi(x_i, x_j).
  """
  J_diag_1: jnp.array
  J_diag_2: jnp.array
  h_1: jnp.array
  h_2: jnp.array
  J_lower_diag: jnp.array
  log_Z: jnp.array

def combine(a: GaussianPotential, b: GaussianPotential):
  return vmap(combine_elt)(a, b)

def combine_elt(a: GaussianPotential, b: GaussianPotential):
    """
    Combines two Gaussian potentials according to the binary associative operator described in Hu et al., 2025.
    """
    J_a_1, J_a_2, h_a_1, h_a_2, J_a_lower, log_Z_a = a
    J_b_1, J_b_2, h_b_1, h_b_2, J_b_lower, log_Z_b = b
    D = J_a_1.shape[0]

    # Condition step
    J_c = J_a_2 + J_b_1
    h_c = h_a_2 + h_b_1

    # Precompute terms
    sqrt_Jc = jnp.linalg.cholesky(J_c)
    trm1 = solve_triangular(sqrt_Jc, h_c, lower=True)
    trm2 = solve_triangular(sqrt_Jc, J_a_lower, lower=True)
    trm3 = solve_triangular(sqrt_Jc, J_b_lower.T, lower=True)

    local_logZ = 0.5 * D * jnp.log(2.0 * jnp.pi)
    local_logZ += -jnp.sum(jnp.log(jnp.diag(sqrt_Jc)))  # sum these terms only to get approx log|J|
    local_logZ += 0.5 * jnp.dot(trm1.T, trm1)

    J_p_1 = J_a_1 - jnp.dot(trm2.T, trm2)
    J_p_2 = J_b_2 - jnp.dot(trm3.T, trm3)
    J_p_lower = -jnp.dot(trm3.T, trm2)
    h_p_1 = h_a_1 - jnp.dot(trm2.T, trm1)
    h_p_2 = h_b_2 - jnp.dot(trm3.T, trm1)
    log_Z_p = log_Z_a + log_Z_b + local_logZ

    return GaussianPotential(J_p_1, J_p_2, h_p_1, h_p_2, J_p_lower, log_Z_p)

def compute_log_normalizer_parallel(precision_diag_blocks: jnp.array, precision_lower_diag_blocks: jnp.array, linear_potential: jnp.array, jitter=1e-3):
    """
    Implements the parallelized log normalizer computation from Hu et al., 2025 for a linear, Gaussian Markov chain.

    Params:
    ------------
    precision_diag_blocks: a (T, D, D) array, the D x D diagonal blocks of the precision matrix
    precision_lower_diag_blocks: a (T-1, D, D) array, the D x D lower-diagonal blocks of the precision matrix
    linear_potential: a (T, D) array, the precision-weighted mean 
    jitter: jitter added to the diagonal block of the prevision matrix

    Returns:
    ------------
    log Z: the log normalizer of the linear, Gaussian Markov chain
    """
    # Build the initial potentials
    def construct_potential(p_diag, p_lower_diag, linear_potential):
        return GaussianPotential(jnp.zeros_like(p_diag), p_diag, jnp.zeros_like(linear_potential), linear_potential, p_lower_diag, 0.)

    dim = precision_diag_blocks.shape[1]
    precision_diag_blocks = precision_diag_blocks + jitter * jnp.eye(dim)[None, :, :]

    # Pad lower diag blocks with zero for the initial potential
    precision_lower_diag_blocks_pad = jnp.concatenate((jnp.zeros((1, dim, dim)), precision_lower_diag_blocks), axis=0)

    # Pad everything with zeros at the end to integrate out the last variable
    precision_diag_blocks_pad = jnp.concatenate((precision_diag_blocks, jnp.zeros((1, dim, dim))), axis=0)
    precision_lower_diag_blocks_pad = jnp.concatenate((precision_lower_diag_blocks_pad, jnp.zeros((1, dim, dim))), axis=0)
    linear_potential_pad = jnp.concatenate((linear_potential, jnp.zeros((1, dim))), axis=0)

    # Construct elements
    elems = vmap(construct_potential)(precision_diag_blocks_pad, precision_lower_diag_blocks_pad, linear_potential_pad)

    # Perform the parallel associative scan
    scanned = lax.associative_scan(combine, elems)
    log_normalizer = scanned.log_Z[-1]
    return log_normalizer

def natural_to_mean_params(natural_params: dict[str, jnp.array]):
    """
    Parallel conversion between natural and mean parameters in a linear, Gaussian Markov chain.
    Works by differentiating the log-normalizer. 

    Params:
    ------------
    natural_params: a dictionary representing the natural parameters, containing
        - J: a shape (T, D, D) array
        - L: a shape (T-1, D, D) array
        - h: a shape (T, D) array

    Returns:
    ------------
    log_normalizer: the log normalizer of the linear, Gaussian Markov chain
    Ex: a shape (T, D) array, the marginal means E[x_i]
    ExxT: a shape (T, D, D) array, the marginal second moments E[x_i x_i^T]
    ExxnT: a shape (T-1, D, D) array, the mixed moment between consecutive states x_i and x_{i+1}, E[x_i x_{i+1}^T]
    """
    precision_diag_blocks, precision_lower_diag_blocks, linear_potential = (-2)*natural_params['J'], (-1)*natural_params['L'], natural_params['h']

    # Take gradients of compute_log_normalizer_parallel to get mean parameters.
    f = value_and_grad(compute_log_normalizer_parallel, argnums=(0, 1, 2))
    log_normalizer, grads = f(precision_diag_blocks, precision_lower_diag_blocks, linear_potential)

    # Correct for the -1/2 J -> J implementation
    ExxT = -2 * grads[0]
    ExxnT = -grads[1]
    Ex = grads[2]
    return log_normalizer, Ex, ExxT, ExxnT

def natural_to_marginal_params(natural_params: dict[str, jnp.array]):
    """
    Parallel conversion between natural and marginal parameters in a linear, Gaussian Markov chain.

    Params:
    ------------
    natural_params: a dictionary representing the natural parameters, containing
        - J: a shape (T, D, D) array
        - L: a shape (T-1, D, D) array
        - h: a shape (T, D) array

    Returns:
    ------------
    marginal_params: a tuple, containing
        - ms: a shape (T, D) array, the marginal means E[x_i]
        - Ss: a shape (T, D, D) array, the marginal covariances Var(x_i)
        - SSs: a shape (T-1, D, D) array, the covariance between consecutive states x_i and x_{i+1}, Cov(x_i, x_{i+1})
    log_normalizer: the log normalizer of the linear, Gaussian Markov chain
    """
    log_normalizer, Ex, ExxT, ExxnT = vmap(natural_to_mean_params)(natural_params)
    marginal_params = vmap(mean_params_to_pairwise_marginals)(Ex, ExxT, ExxnT)
    return marginal_params, log_normalizer

def mean_params_to_pairwise_marginals(Ex: jnp.array, ExxT: jnp.array, ExxnT: jnp.array):
    """
    Computes pairwise marginals from mean parameters.
    """
    T = Ex.shape[0]
    outer = vmap(lambda x1, x2: x1 @ x2.T)
    return Ex, ExxT - outer(Ex[:, :, None], Ex[:, :, None]), ExxnT - outer(Ex[1:, :, None], Ex[:(T-1), :, None])

def pairwise_marginals_to_mean_params(marginal_params: Tuple[jnp.array]):
    """
    Inverse of mean_params_to_pairwise_marginals.
    """
    T = marginal_params[0].shape[0]
    outer = vmap(lambda x1, x2: x1 @ x2.T)
    Ex = marginal_params[0]
    ExxT = marginal_params[1] + outer(Ex[:, :, None], Ex[:, :, None])
    ExxnT = marginal_params[2] + outer(Ex[1:, :, None], Ex[:(T-1), :, None])
    return Ex, ExxT, ExxnT

# ------------------------------- Helper functions for variational EM ---------------------------
def compute_gaussian_entropy(natural_params: dict[str, jnp.array], marginal_params: dict[str, jnp.array], log_normalizer: float):
    """
    Computes the entropy, -E_p[log p(x)], of a multivariate Gaussian with block tridiagonal precision matrix.

    Params:
    ------------
    natural_params: a dictionary representing the natural parameters of the multivariate Gaussian, containing
        - J: a shape (T, D, D) array
        - L: a shape (T-1, D, D) array
        - h: a shape (T, D) array
    marginal_params: a tuple representing the marginal parameters of the multivariate Gaussian, containing
        - ms: a shape (T, D) array, the marginal means E[x_i]
        - Ss: a shape (T, D, D) array, the marginal covariances Var(x_i)
        - SSs: a shape (T-1, D, D) array, the covariance between consecutive states x_i and x_{i+1}, Cov(x_i, x_{i+1})
    log_normalizer: the log normalizer of the multivariate Gaussian

    Returns:
    ------------
    ent: the entropy of the multivariate Gaussian distribution
    """
    Ex, ExxT, ExxnT = pairwise_marginals_to_mean_params(marginal_params)
    inner = lambda x1, x2: jnp.trace(x1.T @ x2)
    tr_term = vmap(inner)(natural_params['J'], ExxT).sum()
    tr_term += vmap(inner)(natural_params['L'], ExxnT).sum()
    tr_term += (Ex * natural_params['h']).sum()
    return log_normalizer - tr_term

def compute_neg_CE_initial(m0: jnp.array, S0: jnp.array, mu0: jnp.array, V0: jnp.array):
    """
    Computes the negative cross-entropy
    E_{q(x0)}[log p(x0)], where q(x0) = N(x0 | m0, S0).
    """
    p = tfd.MultivariateNormalFullCovariance(loc = mu0, covariance_matrix = V0)
    q = tfd.MultivariateNormalFullCovariance(loc = m0, covariance_matrix = S0)
    return -q.cross_entropy(p)

def compute_neg_CE_single(fn: SDE, gp_post: dict[str, jnp.array], drift_params: dict[str, Any], t: float, del_t: float, mt: jnp.array, mt_next: jnp.array, St: jnp.array, St_next: jnp.array, SS: jnp.array, input_t: jnp.array, input_effect: jnp.array, sigma=1.):
    """
    Computes the negative cross-entropy
    E_f[[E_q[log p(x_{i+1}|x_i)]] 
    for a single i between 0 and T-1.
    The expectation over E_f can be ignored if the drift of the prior SDE is not a GP.

    Params:
    ------------
    fn: the prior SDE 
    gp_post: for SING-GP, a dictionary containing the parameters of the variational posterior over inducing points, including 
        - q_u_mu: (D, n_inducing)
        - q_u_sigma: (D, n_inducing, n_inducing)
        None if drift is modeled as deterministic
    drift_params: a dictionary containing the parameters of the prior SDE drift
    t: \tau_i, the time which the cross-entropy is computed
    del_t:\tau_{i+1} - \tau_i, the difference between time \tau_i and the subsequent timestep
    mt: a shape (D) array, the mean at time \tau_i
    mt_next: a shape (D) array, the mean at time \tau_{i+1}
    St: a shape (D, D) array, the covariance at time \tau_i
    St_next: a shape (D, D) array, the covariance at time \tau_{i+1}
    SS: a shape (D, D) array, the covariance between the states at times \tau_i and \tau_{i+1}
    input_t: a shape (n_inputs) array, the input at time \tau_i
    input_effect: a shape (D, n_inputs) array, the linear mapping from inputs to the latent space
    sigma: the noise scale of the prior SDE

    Return:
    ------------ 
    neg_CE: the negative cross-entropy
    """
    Ef = fn.f(drift_params, t, mt, St, gp_post)
    Eff = fn.ff(drift_params, t, mt, St, gp_post)
    Edfdx = fn.dfdx(drift_params, t, mt, St, gp_post)
    
    const = -0.5 * len(mt) * jnp.log(2 * jnp.pi * del_t * (sigma**2))
    trm = jnp.trace(St_next + jnp.outer(mt_next, mt_next))
    trm += jnp.trace(St + jnp.outer(mt, mt))
    trm += -2 * jnp.trace(SS + jnp.outer(mt, mt_next))
    trm += (del_t)**2 * Eff
    trm += -2 * del_t * jnp.trace(jnp.outer(Ef, mt_next) + Edfdx @ SS)
    trm += 2 * del_t * jnp.trace(jnp.outer(Ef, mt) + Edfdx @ St)

    trm += (del_t)**2 * ((input_effect @ input_t)**2).sum() 
    trm += -2 * del_t * jnp.dot(input_effect @ input_t, mt_next - mt - del_t * Ef)

    trm *= -1.0 / (2 * del_t * sigma**2)
    return const + trm

def compute_neg_CE(t_grid: jnp.array, fn: SDE, gp_post: dict[str, jnp.array], drift_params: dict[str, Any], init_params: dict[str, jnp.array], ms: jnp.array, Ss: jnp.array, SSs: jnp.array, inputs: jnp.array, input_effect: jnp.array, sigma=1.):
    """
    Computes total expected negative cross entropy, -E_f[E_q[log p(x_0,...,x_T)].
    """
    neg_CE_init = compute_neg_CE_initial(ms[0], Ss[0], init_params['mu0'], init_params['V0'])
    neg_CE_rest = vmap(partial(compute_neg_CE_single, fn, gp_post, drift_params, input_effect=input_effect, sigma=sigma))(t_grid[:-1], t_grid[1:] - t_grid[:-1], ms[:-1], ms[1:], Ss[:-1], Ss[1:], SSs, inputs[:-1]).sum()
    return neg_CE_init + neg_CE_rest

def update_init_params(m0: jnp.array, S0: jnp.array):
    """
    Updates the initial mean mu0 and covariance V0  of the prior

    Params:
    ------------
    m0: a shape (D) array, the initial mean of the variational posterior
    S0: a shape (D, D) array, the initial covariance of the variational posterior

    Return:
    ------------ 
    init_params: a dictionary containing
        - mu0: a shape (D) array, the initial mean of the prior
        - V0: a shape (D, D) array, the initial covariance of the prior
    """
    mu0, V0 = m0, S0
    init_params = {'mu0': mu0, 'V0': V0}
    return init_params

# -------------------- Mini-batching helpers -------------------------
def subset_batches(args: list, batch_inds: jnp.array):
    return [jax.tree_util.tree_map(lambda x: x[batch_inds], arg) for arg in args]

def fill_batches(args: list, batch_args: list, batch_inds: jnp.array):
    return [jax.tree_util.tree_map(lambda x, y: x.at[batch_inds].set(y), arg, batch_arg) for (arg, batch_arg) in zip(args, batch_args)]

# ---------------------- Parameter conversion functions (sequential, not used) --------------------

# Code from Dynamax (https://probml.github.io/dynamax/) for converting between natural parameters and mean parameters in a linear, Gaussian Markov chain
def block_tridiag_mvn_expectations(precision_diag_blocks: jnp.array, precision_lower_diag_blocks: jnp.array, linear_potential: jnp.array):
    f = value_and_grad(block_tridiag_mvn_log_normalizer, argnums=(0, 1, 2), has_aux=True)
    (log_normalizer, _), grads = f(precision_diag_blocks, precision_lower_diag_blocks, linear_potential)

    # Correct for the -1/2 J -> J implementation
    ExxT = -2 * grads[0]
    ExxnT = -grads[1]
    Ex = grads[2]
    return log_normalizer, Ex, ExxT, ExxnT

def block_tridiag_mvn_log_normalizer(precision_diag_blocks: jnp.array, precision_lower_diag_blocks: jnp.array, linear_potential: jnp.array, jitter=1e-3):
    # Shorthand names
    J_diag = precision_diag_blocks + jitter * jnp.eye(precision_diag_blocks.shape[-1])[None, :, :]
    J_lower_diag = precision_lower_diag_blocks
    h = linear_potential

    # Extract dimensions
    num_timesteps, dim = J_diag.shape[:2]

    # Pad the L's with one extra set of zeros for the last predict step
    J_lower_diag_pad = jnp.concatenate((J_lower_diag, jnp.zeros((1, dim, dim))), axis=0)

    def marginalize(carry, t):
        Jp, hp, lp = carry

        # Condition
        Jc = J_diag[t] + Jp
        hc = h[t] + hp

        sqrt_Jc = jnp.linalg.cholesky(Jc)
        trm1 = solve_triangular(sqrt_Jc, hc, lower=True)
        trm2 = solve_triangular(sqrt_Jc, J_lower_diag_pad[t].T, lower=True)
        log_Z = 0.5 * dim * jnp.log(2 * jnp.pi)
        log_Z += -jnp.sum(jnp.log(jnp.diag(sqrt_Jc)))  # sum these terms only to get approx log|J|
        log_Z += 0.5 * jnp.dot(trm1.T, trm1)
        Jp = -jnp.dot(trm2.T, trm2)
        hp = -jnp.dot(trm2.T, trm1)

        new_carry = Jp, hp, lp + log_Z
        return new_carry, (Jc, hc)

    # Initialize
    Jp0 = jnp.zeros((dim, dim))
    hp0 = jnp.zeros((dim,))
    (_, _, log_Z), (filtered_Js, filtered_hs) = lax.scan(marginalize, (Jp0, hp0, 0), jnp.arange(num_timesteps))
    return log_Z, (filtered_Js, filtered_hs)