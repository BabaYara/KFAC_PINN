"""Simple Physics-Informed Neural Network utilities."""

from __future__ import annotations

from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx


def make_mlp(width: int = 32, depth: int = 2, key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
    layers = []
    for _ in range(depth):
        layers.append(eqx.nn.Linear(width, width, key=key))
        layers.append(jax.nn.tanh)
    layers.append(eqx.nn.Linear(width, 1, key=key))
    return eqx.nn.Sequential(layers)


def pinn_residual(model: eqx.Module, points: jnp.ndarray, rhs: Callable[[jnp.ndarray], jnp.ndarray]):
    def laplace(u, x):
        grads = jax.grad(lambda xx: model(xx).sum())(x)
        return jax.grad(lambda xx: grads.sum())(x)

    return laplace(model, points) - rhs(points)
