"""Simple Physics-Informed Neural Network utilities."""

from __future__ import annotations

from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx

from . import pdes


def make_mlp(in_dim: int = 1, width: int = 32, depth: int = 2, key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
    """Construct a simple fully connected network."""
    layers = []
    k1 = key
    k1, k2 = jax.random.split(k1)
    layers.append(eqx.nn.Linear(in_dim, width, key=k1))
    layers.append(eqx.nn.Lambda(jax.nn.tanh))
    for _ in range(depth - 1):
        k1, k2 = jax.random.split(k2)
        layers.append(eqx.nn.Linear(width, width, key=k1))
        layers.append(eqx.nn.Lambda(jax.nn.tanh))
    layers.append(eqx.nn.Linear(width, 1, key=k2))
    return eqx.nn.Sequential(layers)


def pinn_residual(model: eqx.Module, points: jnp.ndarray, rhs: Callable[[jnp.ndarray], jnp.ndarray]):
    """Compute the PDE residual ``Δu(x) - rhs(x)`` at ``points``."""

    lap = pdes.laplacian(model, points)
    return lap - rhs(points)


def boundary_loss(model: eqx.Module, points: jnp.ndarray, exact: Callable[[jnp.ndarray], jnp.ndarray]):
    """Squared error on boundary conditions."""
    preds = jax.vmap(model)(points)
    return jnp.mean((preds - exact(points)) ** 2)


def interior_loss(model: eqx.Module, points: jnp.ndarray, rhs: Callable[[jnp.ndarray], jnp.ndarray]):
    """Squared residual of the PDE inside the domain."""
    res = jax.vmap(lambda x: pinn_residual(model, x, rhs))(points)
    return jnp.log(jnp.mean(res ** 2))
