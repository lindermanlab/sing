import jax
import jax.numpy as jnp
from jax import vmap

from functools import partial
from sing.sde import SparseGP

from sing.utils.general_helpers import discretize_sde_on_grid
from sing.sde import SDE

from typing import Optional, Any

def initialize_zs(D: int, zs_lim: float, num_per_dim: int):
    """
    Initializes inducing points on a num_per_dim^K grid between -zs_lim and zs_lim.

    Params:
    -------------
    D: latent dimension
    zs_lim: positive scalar, bound on each axis of grid
    num_per_dim: number of inducing points per latent dimension

    Returns:
    -------------
    zs: a shape (num_inducing_points, D) array of inducing points, where num_inducing_points = num_per_dim^K
    """
    assert zs_lim > 0, "zs_lim must be positive"
    
    zs_per_dim = jnp.linspace(-zs_lim, zs_lim, num_per_dim)
    all_zs_per_dim = [zs_per_dim for _ in range(D)]
    zs = jnp.stack(jnp.meshgrid(*all_zs_per_dim), axis=-1).reshape(-1, D)
    return zs

def initialize_params_pca(D: int, ys: jnp.array):
    """
    Initialize output parameters C, d, R with PCA for a Gaussian observation model, y ~ N(Cx+d, R).

    Params:
    -------------
    D: latent dimension
    ys: a shape (n_trials, T, N) array

    Returns:
    -------------
    C_init: a shape (N, D) array, the initialized C (affine mapping)
    d_init: a shape (N,) array, the initialized d (offset)
    R_init: a shape (N,) array, the initialized R (diagonal covariance)
    x0_init_params: a shape (n_trials, D) array, estimated initial latent states 
    """
    ys_stacked = jnp.vstack(ys) # (total_n_samps, N)
    d_init = ys_stacked.mean(0)
    ys_centered = ys_stacked - d_init
    U, S, Vt = jnp.linalg.svd(ys_centered, full_matrices=False)
    C_init = Vt[:D].T # (N, D)

    # initialize output covariance R
    xs_pca = ys_centered @ C_init # (total_n_samps, D)
    ys_recon = xs_pca @ C_init.T + d_init 
    residual = ys_stacked - ys_recon 
    R_init = (residual**2).sum(0) / len(residual) # (N,)

    output_params_init = {'C': C_init, 'd': d_init, 'R': R_init}

    # intialize x0_init_params
    x0_init_params = ((ys - d_init) @ C_init)[:,0] # (n_trials, D) 

    return output_params_init, x0_init_params

def linearize_prior(fn: SDE, gp_post: dict[str, jnp.array], drift_params: dict[str, Any], t_grid: jnp.array, ms: jnp.array, Ss: jnp.array):
    """
    Function for linearizing a nonlinear SDE according to statistical linearization
    i.e. computing
    argmin_{A(t), b(t)} E_{q(x(t))}[||A(\tau_i) x_i + b(\tau_i) - f(x_i, \tau_i)||], i = 0, ..., T
    where f(x, t) is the drift of the nonlinear SDE and q(x_i) is a multivariate Gaussian distribution
    See Verma et al., 2024.
    
    Params:
    -------------
    fn: the SDE to be linearized
    gp_post: for SING-GP, a dictionary containing the parameters of the variational posterior over inducing points, including 
        - q_u_mu: (D, n_inducing)
        - q_u_sigma: (D, n_inducing, n_inducing)
        None if drift is modeled as deterministic
    drift_params: a dictionary containing the parameters of the SDE
    t_grid: a shape (T) array, the grid on which the SDE is discretized
    ms: a shape (T, D) array, the marginal means E[x_i]
    Ss: a shape (T, D, D) array, the marginal covariances Var(x_i)
    
    Return:
    -------------
    a linear SDE of the form dx(t) = A(t)x(t) + b(t) + L dW(t)
    As: a shape (T, D, D) array, the linear transition matrices of the linear SDE  
    bs: a shape (T, D) array, the offset vectors of the linear SDE
    """
    As = vmap(partial(fn.dfdx, drift_params, gp_post=gp_post))(t_grid, ms, Ss)
    bs = vmap(partial(fn.f, drift_params, gp_post=gp_post))(t_grid, ms, Ss) - vmap(lambda x1, x2: x1 @ x2)(As, ms)
    return As, bs

def initialize_params(fn: SDE, drift_params: dict[str, Any], init_params: dict[str, jnp.array], t_grid: jnp.array, sigma: float = 1.):
    """
    Initializes parameters to perform inference.

    Params:
    -------------
    fn: the prior SDE
    drift_params: dictionary containing the parameters of the prior SDE drift
    init_params: a dictionary containing
        - mu0: a shape (D) array, the initial mean of the prior
        - V0: a shape (D, D) array, the initial covariance of the prior
    t_grid: a shape (T) array, the time grid \tau on which to discretize process
    sigma: the noise scale  of the prior SDE
    
    Return:
    -------------
    natural_params: a dictionary containing the parameters of the variational posterior over the latents
        - J: a shape (T, D, D) array
        - L: a shape (T-1, D, D) array
        - h: a shape (T, D) array
    gp_post: a dictionary containing the parameters of the variational posterior over the sparse inducing points
        - q_u_mu: a shape (n_inducing) array, the posterior mean over inducing points
        - q_u_sigma: a shape (n_inducing, n_inducing) array, the posterior variance over inducing points
    None if fn is not a SparseGP
    """

    D = fn.latent_dim
    T = t_grid.shape[0]
    del_t = t_grid[1] - t_grid[0]

    # Initialize dynamics variational parameters
    if isinstance(fn, SparseGP):
        n_inducing = fn.zs.shape[0]
        gp_post = {
        'q_u_mu': jnp.zeros((D, n_inducing)),
        'q_u_sigma': jnp.eye(n_inducing)[None].repeat(D, axis=0)
        }
    else:
        gp_post = None
    
    # Initialize variational posterior by performing statistical linearization wrt N(0, I)
    # TODO: it is probably better to linearize about the initial state of the prior i.e. x0
    ms = jnp.zeros((T, D))
    Ss = (jnp.eye(D)[None, :, :]).repeat(T, axis=0)
    SSs = jnp.zeros((T-1, D, D))
    As, bs = linearize_prior(fn, gp_post, drift_params, t_grid[:-1], ms[:-1], Ss[:-1])
    
    # set posterior covariance = sigma^2 * identity (same as prior covariance)
    L = sigma * jnp.eye(D)
    As_SSM, bs_SSM, Qs_SSM = discretize_sde_on_grid(t_grid, As, bs, L)
    bs_SSM = jnp.concat([init_params['mu0'][None, :], bs_SSM], axis=0)
    Qs_SSM = jnp.concat([init_params['V0'][None, :], Qs_SSM], axis=0)
    J, L, h = get_natural_params(As_SSM, bs_SSM, Qs_SSM)
    natural_params = {
      'J': J, # (T, D, D)
      'L': L, # (T-1, D, D)
      'h': h # (T, K)
    }

    # initialize marginal_params as a tuple of empty arrays
    marginal_params = (jnp.zeros((T, D)), jnp.zeros((T, D, D)), jnp.zeros((T-1, D, D)))
    
    return natural_params, marginal_params, gp_post

def get_natural_params(A, b, Q):
    """
    Function for computing the natural parameters from SSM parameters
    
    Params:
    -------------
    A: a shape (T - 1, D, D) array of transition matrices
    b: a shape (T, D) array of offsets, with b[0] being the initial mean
    Q: a shape (T, D, D) array of noise covariance matrices, with Q[0] being the initial covariance
    """

    # Diagonal blocks of precision matrix
    invQ = jnp.linalg.inv(Q)
    cross = vmap(lambda A_t, invQ_tp1: A_t.T @ invQ_tp1 @ A_t)(A, invQ[1:])                                
    J_diag = invQ.at[:-1].add(cross)

    # Lower diagonal blocks of precision matrix
    J_lower_diag = -vmap(lambda invQ_tp1, A_t: invQ_tp1 @ A_t)(invQ[1:], A)

    # Linear potential
    h_base = vmap(lambda invQ_t, b_t: invQ_t @ b_t)(invQ, b)
    h_corr = -vmap(lambda A_t, invQ_tp1, b_tp1: A_t.T @ (invQ_tp1 @ b_tp1))(A, invQ[1:], b[1:]) # shape (d−1, n)
    h = h_base.at[:-1].add(h_corr)

    return (-1/2)*J_diag, (-1)*J_lower_diag, h