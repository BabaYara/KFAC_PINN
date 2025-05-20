"""Simple Physics-Informed Neural Network utilities."""

from __future__ import annotations

from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx


def make_mlp(width: int = 32, depth: int = 2, key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
    """Construct a simple fully connected network."""
    layers = []
    k1 = key
    for _ in range(depth):
        k1, k2 = jax.random.split(k1)
        layers.append(eqx.nn.Linear(width, width, key=k1))
        layers.append(jax.nn.tanh)
    layers.append(eqx.nn.Linear(width, 1, key=k2))
    return eqx.nn.Sequential(layers)


def pinn_residual(model: eqx.Module, points: jnp.ndarray, rhs: Callable[[jnp.ndarray], jnp.ndarray]):
    """Compute the PDE residual ``L(model)(x) - rhs(x)`` at ``points``."""

    def laplace(u, x):
        grads = jax.grad(lambda xx: u(xx).sum())(x)
        return jax.grad(lambda xx: grads.sum())(x)

    return laplace(model, points) - rhs(points)


def boundary_loss(model: eqx.Module, points: jnp.ndarray, exact: Callable[[jnp.ndarray], jnp.ndarray]):
    """Squared error on boundary conditions."""
    preds = model(points)
    return jnp.mean((preds - exact(points)) ** 2)


def interior_loss(model: eqx.Module, points: jnp.ndarray, rhs: Callable[[jnp.ndarray], jnp.ndarray]):
    """Squared residual of the PDE inside the domain."""
    res = pinn_residual(model, points, rhs)
    return jnp.mean(res ** 2)
