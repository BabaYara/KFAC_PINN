"""Kronecker-Factored Approximate Curvature (KFAC) optimizer.

This is a minimal but complete KFAC implementation for fully connected
networks built with ``equinox.nn.Linear`` layers. It computes Kronecker
factor updates for each layer using exponential moving averages of the
input activations and the backpropagated gradients. The resulting
preconditioned gradients are used to update the parameters.

The implementation is intentionally simple and aimed at small
Physics-Informed Neural Network (PINN) models.  It supports models
constructed using :class:`equinox.nn.Sequential` with ``Linear`` layers
and arbitrary activation functions wrapped in ``equinox.nn.Lambda``.
"""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


class LayerState(NamedTuple):
    """Running estimates of Kronecker factors for one layer."""

    A: jnp.ndarray  # Activation covariance
    G: jnp.ndarray  # Gradient covariance


class KFACState(NamedTuple):
    """Optimizer state."""

    step: int
    layers: Tuple[LayerState, ...]


class KFAC(eqx.Module):
    """
    .. warning::
        This is a basic, generic KFAC optimizer implementation.
        It is **NOT** intended or suitable for Physics-Informed Neural Networks (PINNs)
        that involve differential operators. This implementation does not handle
        the specific requirements for such problems, such as augmented state
        propagation for derivative tracking or separate Kronecker factors for
        PDE residual and boundary condition terms.

        For PINN applications requiring KFAC, please use the `PINNKFAC` class
        located in `kfac_pinn.pinn_kfac.py`. The `PINNKFAC` optimizer is
        specifically designed according to the relevant literature for these
        use cases (e.g., as described in Dangel et al., 2024, "KFAC for PINNs").

    Kronecker-Factored Approximate Curvature optimizer.
    """

    lr: float = 1e-3
    damping: float = 1e-3
    decay: float = 0.95

    def init(self, model: eqx.Module) -> KFACState:
        """Initialise optimiser state for ``model``."""
        factors: List[LayerState] = []
        for layer in _linear_layers(model):
            A = jnp.eye(layer.in_features)
            G = jnp.eye(layer.out_features)
            factors.append(LayerState(A, G))
        return KFACState(step=0, layers=tuple(factors))

    # ------------------------------------------------------------------
    # Core optimisation step
    # ------------------------------------------------------------------
    @eqx.filter_jit
    def step(
        self,
        model: eqx.Module,
        loss_fn: Callable[[eqx.Module, Tuple[jnp.ndarray, ...]], jnp.ndarray],
        batch: Tuple[jnp.ndarray, ...],
        state: KFACState,
    ) -> Tuple[eqx.Module, KFACState]:
        """Apply one KFAC optimisation step."""

        x, *extra = batch
        params, static = eqx.partition(model, eqx.is_array)

        def forward(p):
            m = eqx.combine(static, p)
            return _forward_with_cache(m, x)

        value_and_aux, param_grads = jax.value_and_grad(forward, has_aux=True)(params)
        pred, aux_data = value_and_aux
        acts, phi_primes = aux_data

        loss = loss_fn(eqx.combine(static, params), batch)
        grad_out = jax.grad(lambda y: loss_fn(eqx.combine(static, params), (y, *extra)))(pred)

        new_layers = []
        new_params = params
        g = grad_out
        layer_idx = len(acts) - 1
        for i in reversed(range(len(model.layers))):
            layer = model.layers[i]
            if isinstance(layer, eqx.nn.Linear):
                act = acts[layer_idx]
                phi = phi_primes[layer_idx]
                g = g * phi
                A = (act.T @ act) / act.shape[0]
                G = (g.T @ g) / g.shape[0]

                old_state = state.layers[layer_idx]
                new_A = self.decay * old_state.A + (1.0 - self.decay) * A
                new_G = self.decay * old_state.G + (1.0 - self.decay) * G

                Aw = new_A + self.damping * jnp.eye(new_A.shape[0])
                Gw = new_G + self.damping * jnp.eye(new_G.shape[0])

                w_grad = (g.T @ act) / g.shape[0]
                b_grad = jnp.mean(g, axis=0)

                w_grad = jax.scipy.linalg.solve(Gw, w_grad, assume_a="pos")
                w_grad = jax.scipy.linalg.solve(Aw.T, w_grad.T, assume_a="pos").T
                b_grad = jax.scipy.linalg.solve(Gw, b_grad, assume_a="pos")

                new_weight = layer.weight - self.lr * w_grad
                new_bias = layer.bias - self.lr * b_grad

                new_params = eqx.tree_at(
                    lambda p, i=i: p.layers[i].weight, new_params, new_weight
                )
                new_params = eqx.tree_at(
                    lambda p, i=i: p.layers[i].bias, new_params, new_bias
                )

                new_layers.insert(0, LayerState(new_A, new_G))
                g = jnp.dot(g, layer.weight)
                layer_idx -= 1
            elif isinstance(layer, eqx.nn.Lambda):
                continue
            else:
                raise ValueError("Only Sequential models with Linear layers are supported")

        new_model = eqx.combine(static, new_params)
        new_state = KFACState(step=state.step + 1, layers=tuple(new_layers))
        return new_model, new_state


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def _linear_layers(model: eqx.Module) -> List[eqx.nn.Linear]:
    """Return all ``Linear`` layers inside ``model`` in order."""
    if not isinstance(model, eqx.nn.Sequential):
        raise ValueError("KFAC only supports Sequential models")
    return [layer for layer in model.layers if isinstance(layer, eqx.nn.Linear)]


def _forward_with_cache(
    model: eqx.Module, x: jnp.ndarray
) -> Tuple[jnp.ndarray, List[jnp.ndarray], List[jnp.ndarray]]:
    """Forward pass storing activations and activation derivatives."""
    acts: List[jnp.ndarray] = []
    phi_primes: List[jnp.ndarray] = []
    h = x
    layers = list(model.layers)
    num = len(layers)
    i = 0
    while i < num:
        layer = layers[i]
        if isinstance(layer, eqx.nn.Linear):
            acts.append(h)
            h = layer(h)
            next_is_act = i + 1 < num and isinstance(layers[i + 1], eqx.nn.Lambda)
            if next_is_act:
                act_fn = layers[i + 1].fn
                deriv = jax.vmap(lambda t: jax.grad(lambda u: jnp.sum(act_fn(u)))(t))(h)
                phi_primes.append(deriv)
                h = act_fn(h)
                i += 1  # skip activation layer
            else:
                phi_primes.append(jnp.ones_like(h))
        else:
            h = layer(h)
        i += 1
    return h, acts, phi_primes
