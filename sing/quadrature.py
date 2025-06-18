import jax
import jax.numpy as jnp
from jax import vmap
from numpy.polynomial.hermite_e import hermegauss

from typing import Callable

class GaussHermiteQuadrature:
    """
    An object for integration with respect to a D-dimensional Gaussian distribution
    """
    def __init__(self, D: int, n_quad: int = 10):
        """
        D: dimension of the input space
        n_quad: number of quadrature points per dimension; total number of quadrature points is n_quad^D

        NOTE: GaussHermiteQuadrature is only recommended for low-dimensional (1D, 2D, 3D) input spaces
        """
        self.weights, self.unit_sigmas = self.compute_weights_and_sigmas(D, n_quad)

    def compute_weights_and_sigmas(self, D: int, n_quad: int):
        """
        Computes weights and sigma-points for Gauss-Hermite quadrature.
        """
        samples_1d, weights_1d = jnp.array(hermegauss(n_quad)) # weights are proportional to standard Gaussian density
        weights_1d /= weights_1d.sum() # normalize weights 
        weights_rep = [weights_1d for _ in range(D)]
        samples_rep = [samples_1d for _ in range(D)]
        weights = jnp.stack(jnp.meshgrid(*weights_rep), axis=-1).reshape(-1, D).prod(axis=1)
        unit_sigmas = jnp.stack(jnp.meshgrid(*samples_rep), axis=-1).reshape(-1, D)
        return weights, unit_sigmas
    
    def gaussian_int(self, fn: Callable[jnp.array, jnp.array], m: jnp.array, S: jnp.array):
        """
        Approximates E[f(x)] wrt x ~ N(m, S) with Gauss-Hermite quadrature.
        """
        sigmas = m + (jnp.linalg.cholesky(S) @ self.unit_sigmas[...,None]).squeeze(-1)
        return jnp.tensordot(self.weights, vmap(fn)(sigmas), axes=(0, 0)) # same shape as output of f