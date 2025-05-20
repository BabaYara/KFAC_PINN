"""Training helpers for Physics-Informed Neural Networks."""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from .kfac import KFAC, KFACState

@eqx.filter_jit 
def train_step(model: eqx.Module, opt: KFAC, loss_fn: Callable[[eqx.Module, Tuple[jnp.ndarray, ...]], jnp.ndarray], batch, state: KFACState):
    """Apply one optimisation step."""
    return opt.step(model, loss_fn, batch, state)


@eqx.filter_jit 
def train(model: eqx.Module, opt: KFAC, loss_fn: Callable[[eqx.Module, Tuple[jnp.ndarray, ...]], jnp.ndarray], data_iter: Iterable, steps: int):
    """Simple training loop.

    Parameters
    ----------
    model:
        The neural network to optimise.
    opt:
        Instance of :class:`KFAC`.
    loss_fn:
        Function taking ``(model, batch)`` and returning a scalar loss.
    data_iter:
        Iterable yielding batches of training data.
    steps:
        Number of optimisation steps.
    """
    state = opt.init(model)
    for step, batch in zip(range(steps), data_iter):
        model, state = train_step(model, opt, loss_fn, batch, state)
    return model, state
