"""PINN-specific KFAC implementation.

This module implements a Kronecker-Factored Approximate Curvature optimiser
geared towards Physics-Informed Neural Networks as described in
``Tex/2025_05_20_e8fffb9338419e358febg.tex``. The implementation
closely follows Algorithm~1 in that document and maintains separate
Kronecker factors for the PDE operator term and the boundary term.

The optimiser is intentionally written with clarity in mind. It currently
supports sequential MLP models composed of ``equinox.nn.Linear`` layers
and activation functions.  The implementation should be considered an
initial version -- several parts of the algorithm such as the forward
Laplacian propagation are simplified.  Nevertheless, all four Kronecker
factors are tracked and used to precondition gradients.

"""
from __future__ import annotations

from typing import Callable, List, NamedTuple, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .pdes import forward_laplacian


class LayerFactors(NamedTuple):
    A_omega: jnp.ndarray
    B_omega: jnp.ndarray
    A_boundary: jnp.ndarray
    B_boundary: jnp.ndarray
    mw: jnp.ndarray
    mb: jnp.ndarray


class PINNKFACState(NamedTuple):
    step: int
    layers: Tuple[LayerFactors, ...]


class PINNKFAC(eqx.Module):
    """KFAC optimiser for Physics-Informed Neural Networks."""

    lr: float = 1e-2
    damping: float = 1e-3
    decay: float = 0.95
    momentum: float = 0.9

    def init(self, model: eqx.Module) -> PINNKFACState:
        factors: List[LayerFactors] = []
        for layer in _linear_layers(model):
            m, n = layer.out_features, layer.in_features
            factors.append(
                LayerFactors(
                    A_omega=jnp.eye(n),
                    B_omega=jnp.eye(m),
                    A_boundary=jnp.eye(n),
                    B_boundary=jnp.eye(m),
                    mw=jnp.zeros_like(layer.weight),
                    mb=jnp.zeros_like(layer.bias),
                )
            )
        return PINNKFACState(step=0, layers=tuple(factors))

    # ------------------------------------------------------------------
    def step(
        self,
        model: eqx.Module,
        rhs_fn: Callable[[jnp.ndarray], jnp.ndarray],
        bc_fn: Callable[[jnp.ndarray], jnp.ndarray],
        interior: jnp.ndarray,
        boundary: jnp.ndarray,
        state: PINNKFACState,
    ) -> Tuple[eqx.Module, PINNKFACState]:
        """Perform one optimiser step.

        Parameters
        ----------
        model: eqx.Module
            Neural network to optimise.
        rhs_fn, bc_fn:
            Functions defining the PDE right-hand side and boundary
            conditions.
        interior, boundary: jnp.ndarray
            Sampled interior and boundary coordinates.
        state: PINNKFACState
            Optimiser state.
        """

        params, static = eqx.partition(model, eqx.is_array)

        def interior_loss(p):
            m = eqx.combine(static, p)
            lap = jax.vmap(lambda x: forward_laplacian(m, x[None, :]))(interior)
            res = lap.squeeze() - rhs_fn(interior)
            return 0.5 * jnp.mean(res**2)

        def boundary_loss(p):
            m = eqx.combine(static, p)
            preds = jax.vmap(m)(boundary)
            res = preds.squeeze() - bc_fn(boundary)
            return 0.5 * jnp.mean(res**2)

        loss_fn = lambda p: interior_loss(p) + boundary_loss(p)

        loss_val, grads = jax.value_and_grad(loss_fn)(params)

        acts_i, deltas_i = _factor_terms(model, params, interior, rhs_fn, True)
        acts_b, deltas_b = _factor_terms(model, params, boundary, bc_fn, False)

        new_layers = []
        for lf, a_i, d_i, a_b, d_b in zip(
            state.layers, acts_i, deltas_i, acts_b, deltas_b
        ):
            A_om = self.decay * lf.A_omega + (1 - self.decay) * (a_i.T @ a_i) / a_i.shape[0]
            B_om = self.decay * lf.B_omega + (1 - self.decay) * (d_i.T @ d_i) / d_i.shape[0]
            A_bd = self.decay * lf.A_boundary + (1 - self.decay) * (a_b.T @ a_b) / a_b.shape[0]
            B_bd = self.decay * lf.B_boundary + (1 - self.decay) * (d_b.T @ d_b) / d_b.shape[0]

            new_layers.append(
                LayerFactors(A_om, B_om, A_bd, B_bd, lf.mw, lf.mb)
            )
        state = PINNKFACState(step=state.step + 1, layers=tuple(new_layers))

        updates = []
        new_layers = []
        lin_indices = [i for i, l in enumerate(model.layers) if isinstance(l, eqx.nn.Linear)]
        layer_idx = len(state.layers) - 1
        for i in reversed(lin_indices):
            layer = model.layers[i]
            lf = state.layers[layer_idx]
            Aw = lf.A_omega + self.damping * jnp.eye(lf.A_omega.shape[0])
            Gw = lf.B_omega + self.damping * jnp.eye(lf.B_omega.shape[0])
            Aw_b = lf.A_boundary + self.damping * jnp.eye(lf.A_boundary.shape[0])
            Gw_b = lf.B_boundary + self.damping * jnp.eye(lf.B_boundary.shape[0])

            eig_A, UA = jnp.linalg.eigh(Aw)
            eig_G, UG = jnp.linalg.eigh(Gw)
            eig_Ab, UAb = jnp.linalg.eigh(Aw_b)
            eig_Gb, UGb = jnp.linalg.eigh(Gw_b)

            gw = grads.layers[i].weight
            gb = grads.layers[i].bias

            gw = UA.T @ gw @ UG
            gw = gw / (eig_A[:, None] * eig_G[None, :] + eig_Ab[:, None] * eig_Gb[None, :])
            gw = UA @ gw @ UG.T

            gb = UG.T @ gb
            gb = gb / (eig_G + eig_Gb)
            gb = UG @ gb

            mw = self.momentum * lf.mw + gw
            mb = self.momentum * lf.mb + gb

            updates.insert(0, (mw, mb))
            new_layers.insert(0, LayerFactors(lf.A_omega, lf.B_omega, lf.A_boundary, lf.B_boundary, mw, mb))
            layer_idx -= 1

        def apply_updates(p, alpha):
            for idx, (mw, mb) in zip(lin_indices, updates):
                w = p.layers[idx].weight - alpha * mw
                b = p.layers[idx].bias - alpha * mb
                p = eqx.tree_at(lambda q, i=idx: q.layers[i].weight, p, w)
                p = eqx.tree_at(lambda q, i=idx: q.layers[i].bias, p, b)
            return p

        alphas = self.lr * (2.0 ** jnp.arange(0, -5, -1))
        losses = jax.vmap(lambda a: loss_fn(apply_updates(params, a)))(alphas)
        best_alpha = alphas[jnp.argmin(losses)]
        params = apply_updates(params, best_alpha)

        state = PINNKFACState(step=state.step, layers=tuple(new_layers))
        new_model = eqx.combine(static, params)
        return new_model, state


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _linear_layers(model: eqx.Module) -> List[eqx.nn.Linear]:
    if not isinstance(model, eqx.nn.Sequential):
        raise ValueError("Only Sequential models supported")
    return [l for l in model.layers if isinstance(l, eqx.nn.Linear)]


def _forward_cache(model: eqx.Module, x: jnp.ndarray):
    acts = []
    pre = []
    phi = []
    h = x
    layers = list(model.layers)
    i = 0
    while i < len(layers):
        layer = layers[i]
        if isinstance(layer, eqx.nn.Linear):
            acts.append(h)
            s = layer(h)
            pre.append(s)
            next_is_act = i + 1 < len(layers) and isinstance(layers[i + 1], eqx.nn.Lambda)
            if next_is_act:
                fn = layers[i + 1].fn
                phi.append(jax.vmap(jax.grad(fn))(s))
                h = fn(s)
                i += 1
            else:
                phi.append(jnp.ones_like(s))
                h = s
        else:
            h = layer(h)
        i += 1
    return h, acts, pre, phi


def _factor_terms(model, params, pts, fn, interior: bool):
    m = eqx.combine(eqx.partition(model, eqx.is_array)[1], params)
    y, acts, pre, phi = _forward_cache(m, pts)
    if interior:
        lap = jax.vmap(lambda x: forward_laplacian(m, x[None, :]))(pts).squeeze()
        res = lap - fn(pts)
    else:
        res = y.squeeze() - fn(pts)
    grad_out = res / pts.shape[0]
    deltas = []
    g = grad_out[:, None]
    layer_idx = len(acts) - 1
    for i in reversed(range(len(model.layers))):
        layer = model.layers[i]
        if isinstance(layer, eqx.nn.Linear):
            phi_l = phi[layer_idx]
            g = g * phi_l
            deltas.insert(0, g)
            g = g @ layer.weight.T
            layer_idx -= 1
        elif isinstance(layer, eqx.nn.Lambda):
            continue
    return acts, deltas
