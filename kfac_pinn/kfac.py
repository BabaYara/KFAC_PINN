"""Minimal KFAC optimizer implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Any, NamedTuple
import equinox as eqx


class KFACState(NamedTuple):
    step: int
    a: Any
    g: Any


def _update_stats(a, g, stats, decay=0.95):
    return jax.tree_util.tree_map(
        lambda s, x: decay * s + (1.0 - decay) * x, stats, jax.tree_util.tree_map(jnp.mean, (a, g))
    )


class KFAC(eqx.Module):
    lr: float = 1e-3
    damping: float = 1e-3

    def init(self, model, batch):
        a = jax.tree_util.tree_map(jnp.zeros_like, batch)
        g = jax.tree_util.tree_map(jnp.zeros_like, model)
        return KFACState(step=0, a=a, g=g)

    def step(self, model, loss_fn, batch, state: KFACState):
        def closure(params, inputs):
            return loss_fn(eqx.combine(model, params), inputs)

        grads = jax.grad(closure)(eqx.filter(model, eqx.is_array), batch)

        a = jax.tree_util.tree_map(lambda x: x.T @ x, batch)
        g = jax.tree_util.tree_map(lambda x: x.T @ x, grads)

        new_a = _update_stats(a, g, state.a)
        new_g = _update_stats(g, a, state.g)

        precond = jax.tree_util.tree_map(
            lambda gg, aa, d: jnp.linalg.inv(gg + d * jnp.eye(gg.shape[0]))
            @ (aa + d * jnp.eye(aa.shape[0])),
            new_g,
            new_a,
            self.damping,
        )

        updates = jax.tree_util.tree_map(lambda p, u: p - self.lr * u, model, precond)
        new_model = eqx.combine(model, updates)

        new_state = KFACState(step=state.step + 1, a=new_a, g=new_g)
        return new_model, new_state
