"""Minimal KFAC optimizer implementation.

This is a very small optimiser that maintains a running diagonal estimate of the
Fisher information matrix.  It is **not** a full KFAC implementation but it
captures the flavour of preconditioning gradients with approximate curvature
information.  The optimiser is designed to be used with the simple
Physics‑Informed Neural Network examples in this repository.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Any, NamedTuple
import equinox as eqx


class KFACState(NamedTuple):
    """State for :class:`KFAC`."""

    step: int
    fisher: Any


def _update_fisher(fisher, grads, decay: float):
    """Exponential moving average of squared gradients."""
    return jax.tree_util.tree_map(
        lambda f, g: decay * f + (1.0 - decay) * (g ** 2), fisher, grads
    )


class KFAC(eqx.Module):
    """A tiny diagonal KFAC-like optimiser."""

    lr: float = 1e-3
    damping: float = 1e-3
    decay: float = 0.95

    def init(self, model):
        params = eqx.filter(model, eqx.is_array)
        fisher = jax.tree_util.tree_map(jnp.zeros_like, params)
        return KFACState(step=0, fisher=fisher)

    def step(self, model, loss_fn, batch, state: KFACState):
        params, static = eqx.partition(model, eqx.is_array)

        def closure(p):
            m = eqx.combine(static, p)
            return loss_fn(m, batch)

        loss, grads = jax.value_and_grad(closure)(params)

        new_fisher = _update_fisher(state.fisher, grads, self.decay)
        precond_grads = jax.tree_util.tree_map(
            lambda g, f: g / (jnp.sqrt(f) + self.damping), grads, new_fisher
        )

        new_params = jax.tree_util.tree_map(
            lambda p, g: p - self.lr * g, params, precond_grads
        )
        new_model = eqx.combine(static, new_params)

        new_state = KFACState(step=state.step + 1, fisher=new_fisher)
        return new_model, new_state
