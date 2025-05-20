"""PDE-related helper utilities for PINNs."""

from __future__ import annotations

from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx


def laplacian(model: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """Compute the Laplacian of ``model`` at ``x``.

    This uses automatic differentiation to evaluate the trace of the Hessian of
    the model output with respect to ``x``. ``x`` can be a single point of shape
    ``(dim,)`` or a batch of points ``(batch, dim)``.
    """

    def single_point(xx):
        def scalar_fn(xp):
            return jnp.sum(model(xp))

        hess = jax.hessian(scalar_fn)(xx)
        return jnp.trace(hess)

    return jax.vmap(single_point)(x) if x.ndim > 1 else single_point(x)


def sample_interior(key: jax.random.PRNGKey, domain_min: jnp.ndarray, domain_max: jnp.ndarray, num: int) -> jnp.ndarray:
    """Sample ``num`` points uniformly from the interior of a hyper-rectangle."""
    dim = domain_min.shape[0]
    pts = jax.random.uniform(key, (num, dim))
    return domain_min + (domain_max - domain_min) * pts


def sample_boundary(key: jax.random.PRNGKey, domain_min: jnp.ndarray, domain_max: jnp.ndarray, num: int) -> jnp.ndarray:
    """Sample ``num`` points uniformly from the boundary of a hyper-rectangle."""
    dim = domain_min.shape[0]
    pts = jax.random.uniform(key, (num, dim))
    face = jax.random.randint(key, (num,), 0, dim)
    pts = domain_min + (domain_max - domain_min) * pts
    pts = pts.at[jnp.arange(num), face].set(domain_min[face])
    return pts
