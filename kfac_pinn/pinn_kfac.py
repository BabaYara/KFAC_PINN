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

from typing import Callable, List, NamedTuple, Tuple, Any # Added Any

import equinox as eqx
import jax
import jax.numpy as jnp

from .pdes import forward_laplacian


# Augmented state for propagating (value, d/dx, d/dy, laplacian)
class AugmentedState(NamedTuple):
    value: jnp.ndarray  # Batched: (batch, features)
    d_dx: jnp.ndarray   # Batched: (batch, features)
    d_dy: jnp.ndarray   # Batched: (batch, features)
    laplacian: jnp.ndarray # Batched: (batch, features)

    @classmethod
    def from_coords(cls, coords: jnp.ndarray) -> "AugmentedState":
        """
        Initializes AugmentedState from input coordinates.
        coords: (batch, dim) - expected dim=2 for (x,y)
        """
        batch_size, dim = coords.shape
        if dim != 2:
            # For now, strict 2D. Generalize later if needed.
            raise ValueError(f"Expected 2D coordinates for AugmentedState, got {dim}D.")

        value = coords
        
        # For input x (coords[:, 0]), dx/dx = 1, dx/dy = 0. Lap(x) = 0
        # For input y (coords[:, 1]), dy/dx = 0, dy/dy = 1. Lap(y) = 0
        # So, d_dx applied to (x,y) vector results in (1,0) for each sample.
        # And d_dy applied to (x,y) vector results in (0,1) for each sample.
        
        # d_dx_coords has shape (batch_size, 2)
        # first col is d(x_coord)/dx, second is d(y_coord)/dx
        d_dx_val = jnp.zeros_like(coords)
        d_dx_val = d_dx_val.at[:, 0].set(1.0) 
        
        # d_dy_coords has shape (batch_size, 2)
        # first col is d(x_coord)/dy, second is d(y_coord)/dy
        d_dy_val = jnp.zeros_like(coords)
        d_dy_val = d_dy_val.at[:, 1].set(1.0)
        
        # laplacian_coords has shape (batch_size, 2)
        # first col is lap(x_coord), second is lap(y_coord)
        laplacian_val = jnp.zeros_like(coords)

        return cls(value, d_dx_val, d_dy_val, laplacian_val)

    def concatenate_components(self, per_sample_norm: bool = False) -> jnp.ndarray:
        """
        Concatenates all S components.
        Output: (batch_size * S, features)
        If per_sample_norm is True, this indicates that the norm will be taken per sample later.
        """
        # value, d_dx, d_dy, laplacian all have shape (batch, features)
        # Stack them to create S "virtual" samples per original sample.
        # (S, batch, features) then transpose and reshape.
        s_components = [self.value, self.d_dx, self.d_dy, self.laplacian]
        num_s_components = len(s_components) # S=4 for 2D
        
        concatenated = jnp.concatenate(s_components, axis=0) # (S * batch, features)
        return concatenated

    @property
    def num_s_components(self):
        return 4 # value, d_dx, d_dy, laplacian

def _propagate_linear_augmented(aug_state: AugmentedState, layer: eqx.nn.Linear) -> AugmentedState:
    """Propagates AugmentedState through an eqx.nn.Linear layer."""
    # Bias only affects the 'value' component. Derivatives of bias are zero.
    new_value = aug_state.value @ layer.weight.T + layer.bias
    new_d_dx = aug_state.d_dx @ layer.weight.T
    new_d_dy = aug_state.d_dy @ layer.weight.T
    new_laplacian = aug_state.laplacian @ layer.weight.T
    
    return AugmentedState(new_value, new_d_dx, new_d_dy, new_laplacian)

def _propagate_activation_augmented(
    aug_state_pre_activation: AugmentedState, 
    activation_fn: Callable, 
    vmap_activation_fn_grad: Callable, # vmapped jax.grad(activation_fn)
    vmap_activation_fn_grad_grad: Callable # vmapped jax.grad(jax.grad(activation_fn))
) -> AugmentedState:
    """
    Propagates AugmentedState through an element-wise activation function.
    Assumes activation_fn operates element-wise on `aug_state_pre_activation.value`.
    """
    s_val = aug_state_pre_activation.value # (batch, features_in)
    s_dx = aug_state_pre_activation.d_dx   # (batch, features_in)
    s_dy = aug_state_pre_activation.d_dy   # (batch, features_in)
    s_lap = aug_state_pre_activation.laplacian # (batch, features_in)

    # These grads are element-wise for each feature
    sigma_prime_s = vmap_activation_fn_grad(s_val)     # (batch, features_in)
    sigma_prime_prime_s = vmap_activation_fn_grad_grad(s_val) # (batch, features_in)

    val_out = activation_fn(s_val) # (batch, features_in)
    d_dx_out = sigma_prime_s * s_dx
    d_dy_out = sigma_prime_s * s_dy
    
    # Element-wise computation for laplacian propagation
    # sigma''(s) * ( (d_dx s)^2 + (d_dy s)^2 )
    # All terms are (batch, features_in)
    term_sum_sq_derivs = sigma_prime_prime_s * (jnp.square(s_dx) + jnp.square(s_dy))
    lap_out = sigma_prime_s * s_lap + term_sum_sq_derivs 
    
    return AugmentedState(val_out, d_dx_out, d_dy_out, lap_out)


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

        def interior_loss_for_grads(p_tree): # This is L_Omega for main KFAC gradients
            m = eqx.combine(static, p_tree)
            # Compute final laplacian using standard forward_laplacian for actual loss calculation
            lap_standard = jax.vmap(lambda x_single: forward_laplacian(m, x_single[None, :]))(interior).squeeze()
            res_standard = lap_standard - rhs_fn(interior)
            return 0.5 * jnp.mean(res_standard**2)

        def boundary_loss_for_grads(p_tree):
            m = eqx.combine(static, p_tree)
            preds = jax.vmap(m)(boundary)
            res = preds.squeeze() - bc_fn(boundary)
            return 0.5 * jnp.mean(res**2)

        loss_fn_for_grads = lambda p_tree: interior_loss_for_grads(p_tree) + boundary_loss_for_grads(p_tree)

        loss_val, grads = jax.value_and_grad(loss_fn_for_grads)(params)
        
        m_eval = eqx.combine(static, params) # Model with current parameters

        # Boundary factor calculation (standard KFAC)
        y_b, acts_b_std, pre_b_std, phi_b_std = _standard_forward_cache(m_eval, boundary)
        res_b = y_b.squeeze() - bc_fn(boundary) # Residual for boundary loss
        grad_out_b = res_b / boundary.shape[0]  # dL_boundary / dy_pred_boundary
        deltas_b_std = _standard_backward_pass(m_eval, pre_b_std, phi_b_std, grad_out_b)
        
        # Interior factor calculation (augmented KFAC)
        aug_factors_i = _augmented_factor_terms(m_eval, params, interior, rhs_fn)

        new_layers = []
        for i, lf in enumerate(state.layers):
            a_i_aug, d_i_aug = aug_factors_i[i] 
            
            A_om_update = (a_i_aug.T @ a_i_aug) / a_i_aug.shape[0]
            B_om_update = (d_i_aug.T @ d_i_aug) / d_i_aug.shape[0]

            A_om = self.decay * lf.A_omega + (1 - self.decay) * A_om_update
            B_om = self.decay * lf.B_omega + (1 - self.decay) * B_om_update
            
            a_b_std = acts_b_std[i] 
            d_b_std = deltas_b_std[i] 
            A_bd_update = (a_b_std.T @ a_b_std) / a_b_std.shape[0] 
            B_bd_update = (d_b_std.T @ d_b_std) / d_b_std.shape[0] 
            A_bd = self.decay * lf.A_boundary + (1 - self.decay) * A_bd_update
            B_bd = self.decay * lf.B_boundary + (1 - self.decay) * B_bd_update
            
            new_layers.append(
                LayerFactors(A_om, B_om, A_bd, B_bd, lf.mw, lf.mb)
            )
        state = PINNKFACState(step=state.step + 1, layers=tuple(new_layers))

        updates = []
        new_layers_post_grad = [] # Renamed to avoid confusion with new_layers for factors
        lin_indices = [k for k, l_obj in enumerate(model.layers) if isinstance(l_obj, eqx.nn.Linear)]
        layer_idx_kfac = len(state.layers) - 1 # Index for KFAC state's layers tuple
        
        for i_grad_idx in reversed(lin_indices): # i_grad_idx is the actual model layer index
            # Find corresponding KFAC layer_factor
            # This assumes _linear_layers(model) used for init matches iteration here.
            # The lf should correspond to the model.layers[i_grad_idx]
            # The state.layers are ordered same as _linear_layers output.
            # So if lin_indices are [idx_lin0, idx_lin1, ...], then state.layers[0] is for model.layers[idx_lin0]
            # This reversed loop means we need to map i_grad_idx to its position in lin_indices.
            
            # Correct mapping from model's linear layer index to KFAC state layer index
            kfac_layer_index_for_current_grad = lin_indices.index(i_grad_idx)
            
            # The loop for `lf` should be based on `layer_idx_kfac` which counts down
            # matching the reversed `lin_indices`.
            # No, layer_idx_kfac is correct as it's used with reversed(lin_indices).

            lf = state.layers[layer_idx_kfac] # This should be correct if layer_idx_kfac counts down.

            Aw = lf.A_omega + self.damping * jnp.eye(lf.A_omega.shape[0])
            Gw = lf.B_omega + self.damping * jnp.eye(lf.B_omega.shape[0])
            Aw_b = lf.A_boundary + self.damping * jnp.eye(lf.A_boundary.shape[0])
            Gw_b = lf.B_boundary + self.damping * jnp.eye(lf.B_boundary.shape[0])

            eig_A, UA = jnp.linalg.eigh(Aw)
            eig_G, UG = jnp.linalg.eigh(Gw)
            eig_Ab, UAb = jnp.linalg.eigh(Aw_b)
            eig_Gb, UGb = jnp.linalg.eigh(Gw_b)

            # Grads are w.r.t. params PyTree structure
            # grads.layers[k] where k is index in model.layers if it's Sequential
            # If model is a general PyTree, grads will match that structure.
            # Assuming model is Sequential and grads.layers[i_grad_idx] is the correct grad
            
            current_layer_grads = grads.layers[i_grad_idx] # grads for model.layers[i_grad_idx]
            gw = current_layer_grads.weight
            gb = current_layer_grads.bias
            
            gw_kfac = UA.T @ gw @ UG
            gw_kfac = gw_kfac / (eig_A[:, None] * eig_G[None, :] + eig_Ab[:, None] * eig_Gb[None, :])
            gw_kfac = UA @ gw_kfac @ UG.T

            gb_kfac = UG.T @ gb # Bias grad preconditioning by B_omega + B_boundary
            gb_kfac = gb_kfac / (eig_G + eig_Gb) # Element-wise division
            gb_kfac = UG @ gb_kfac

            mw_new = self.momentum * lf.mw + gw_kfac
            mb_new = self.momentum * lf.mb + gb_kfac

            updates.insert(0, (mw_new, mb_new))
            # For the new state, update mw and mb
            new_layers_post_grad.insert(0, LayerFactors(lf.A_omega, lf.B_omega, lf.A_boundary, lf.B_boundary, mw_new, mb_new))
            layer_idx_kfac -= 1


        def apply_updates(p_apply, alpha_apply):
            updated_params = p_apply
            for i_update, (mw_u, mb_u) in enumerate(updates):
                # lin_indices gives the actual index in model.layers
                actual_model_layer_idx = lin_indices[i_update] 
                
                # Path to weight and bias for the specific layer in the PyTree
                weight_path = lambda tree: tree.layers[actual_model_layer_idx].weight
                bias_path = lambda tree: tree.layers[actual_model_layer_idx].bias

                new_weight = weight_path(updated_params) - alpha_apply * mw_u
                new_bias = bias_path(updated_params) - alpha_apply * mb_u
                
                updated_params = eqx.tree_at(weight_path, updated_params, new_weight)
                updated_params = eqx.tree_at(bias_path, updated_params, new_bias)
            return updated_params

        alphas_search = self.lr * (2.0 ** jnp.arange(0, -5, -1))
        # loss_fn_for_grads uses the new parameter tree 'p'
        losses_search = jax.vmap(lambda alpha_s: loss_fn_for_grads(apply_updates(params, alpha_s)))(alphas_search)
        
        best_alpha = alphas_search[jnp.argmin(losses_search)]
        final_params = apply_updates(params, best_alpha)

        # Create the final PINNKFACState with updated momentum terms
        final_state = PINNKFACState(step=state.step, layers=tuple(new_layers_post_grad)) # Use layers with updated mw, mb
        new_model = eqx.combine(static, final_params)
        return new_model, final_state


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _linear_layers(model: eqx.Module) -> List[eqx.nn.Linear]:
    if not isinstance(model, eqx.nn.Sequential):
        raise ValueError("Only Sequential models supported")
    return [l for l in model.layers if isinstance(l, eqx.nn.Linear)]


# Renamed from _forward_cache to avoid confusion
def _standard_forward_cache(model: eqx.Module, x: jnp.ndarray):
    """Standard forward pass, collects activations and pre-activations."""
    acts = []  # To store h (inputs to linear layers)
    pre = []   # To store s (outputs of linear layers, pre-activation)
    phi = []   # To store sigma'(s) (derivatives of activation functions)
    
    h = x # Current activation
    
    _activation_fn_grad_vmap = None
    _activation_fn_lambda = None

    for layer_obj in model.layers:
        if isinstance(layer_obj, eqx.nn.Lambda):
            try:
                layer_obj.fn(jnp.array(0.0))
                _activation_fn_lambda = layer_obj.fn
            except Exception: 
                 print("Warning: Could not directly get scalar fn for activation grad. Assuming direct grad works.")
                 _activation_fn_lambda = lambda val: layer_obj.fn(val[None])[0] 

            if _activation_fn_lambda:
                _activation_fn_grad_vmap = jax.vmap(jax.grad(_activation_fn_lambda))
            break 

    layers = list(model.layers)
    i = 0
    while i < len(layers):
        layer = layers[i]
        if isinstance(layer, eqx.nn.Linear):
            acts.append(h) 
            s = layer(h)   
            pre.append(s)
            
            if i + 1 < len(layers) and isinstance(layers[i+1], eqx.nn.Lambda):
                if _activation_fn_grad_vmap is not None: # Check explicitly for None
                    phi.append(_activation_fn_grad_vmap(s))
                else: 
                    phi.append(jnp.ones_like(s)) 
                h = layers[i+1](s) 
                i += 1 
            else: 
                phi.append(jnp.ones_like(s)) 
                h = s
        else: 
            h = layer(h) 
        i += 1
        
    return h, acts, pre, phi

def _standard_backward_pass(model: eqx.Module, pre_activations: List[jnp.ndarray], act_derivatives: List[jnp.ndarray], grad_output: jnp.ndarray) -> List[jnp.ndarray]:
    """Standard backward pass to compute deltas for KFAC factors."""
    deltas = []
    # Ensure g has shape (batch_size, num_outputs_of_model)
    if grad_output.ndim == 1:
        g = grad_output[:, None] 
    else:
        g = grad_output
    
    linear_layer_indices = [i for i, l_obj in enumerate(model.layers) if isinstance(l_obj, eqx.nn.Linear)]
    
    for idx_in_model_lists in reversed(range(len(linear_layer_indices)))):
        # model_layer_idx = linear_layer_indices[idx_in_model_lists] # Actual index in model.layers
        layer_object = _linear_layers(model)[idx_in_model_lists] # Get the specific Linear layer object

        phi_l = act_derivatives[idx_in_model_lists] 
        
        g_s_l = g * phi_l 
        deltas.insert(0, g_s_l) 
        
        g = g_s_l @ layer_object.weight 
        
    return deltas


def _augmented_forward_cache(model: eqx.Module, initial_aug_state: AugmentedState) -> Tuple[List[AugmentedState], List[AugmentedState], AugmentedState]:
    """
    Performs a forward pass propagating the AugmentedState.
    Returns:
        - aug_input_acts: List of AugmentedState, inputs to each Linear layer.
        - aug_pre_acts: List of AugmentedState, outputs of each Linear layer (pre-activation).
        - final_aug_output: AugmentedState after the last layer.
    """
    aug_input_acts: List[AugmentedState] = []
    aug_pre_acts: List[AugmentedState] = []
    
    current_aug_state = initial_aug_state

    _activation_fn_obj = None 
    _vmap_activation_fn_grad = None
    _vmap_activation_fn_grad_grad = None
    
    for layer_obj in model.layers:
        if isinstance(layer_obj, eqx.nn.Lambda):
            scalar_fn_for_grad = None
            try: 
                layer_obj.fn(jnp.array(0.0)) 
                scalar_fn_for_grad = layer_obj.fn
            except: 
                try:
                    scalar_fn_for_grad = lambda x_scalar: layer_obj.fn(jnp.array([x_scalar]))[0]
                except:
                    pass 

            if scalar_fn_for_grad:
                _activation_fn_obj = layer_obj.fn 
                _vmap_activation_fn_grad = jax.vmap(jax.grad(scalar_fn_for_grad))
                _vmap_activation_fn_grad_grad = jax.vmap(jax.grad(jax.grad(scalar_fn_for_grad)))
            else: 
                _activation_fn_obj = None
                _vmap_activation_fn_grad = None
                _vmap_activation_fn_grad_grad = None
                print("Warning: Could not determine scalar activation function for augmented derivatives. Activations may not propagate correctly.")
            break 

    layers = list(model.layers)
    i = 0
    while i < len(layers):
        layer = layers[i]
        if isinstance(layer, eqx.nn.Linear):
            aug_input_acts.append(current_aug_state)
            current_aug_state = _propagate_linear_augmented(current_aug_state, layer)
            aug_pre_acts.append(current_aug_state) 

            if i + 1 < len(layers) and isinstance(layers[i+1], eqx.nn.Lambda):
                if _activation_fn_obj and _vmap_activation_fn_grad and _vmap_activation_fn_grad_grad:
                    current_aug_state = _propagate_activation_augmented(
                        current_aug_state, 
                        _activation_fn_obj, 
                        _vmap_activation_fn_grad,
                        _vmap_activation_fn_grad_grad
                    )
                i += 1 
        
        elif isinstance(layer, eqx.nn.Lambda): 
            if _activation_fn_obj and _vmap_activation_fn_grad and _vmap_activation_fn_grad_grad:
                 current_aug_state = _propagate_activation_augmented(
                    current_aug_state, 
                    _activation_fn_obj, 
                    _vmap_activation_fn_grad,
                    _vmap_activation_fn_grad_grad
                )
            else:
                 current_aug_state = AugmentedState(
                     layer(current_aug_state.value), 
                     layer(current_aug_state.d_dx), 
                     layer(current_aug_state.d_dy), 
                     layer(current_aug_state.laplacian)
                 )
        else: 
            current_aug_state = AugmentedState(
                layer(current_aug_state.value), 
                jnp.zeros_like(current_aug_state.d_dx), 
                jnp.zeros_like(current_aug_state.d_dy),
                jnp.zeros_like(current_aug_state.laplacian)
            )
        i += 1
        
    return aug_input_acts, aug_pre_acts, current_aug_state


def _augmented_factor_terms(
    model_eval: eqx.Module, 
    params: Any, # params of the model
    interior_pts: jnp.ndarray, 
    rhs_fn: Callable[[jnp.ndarray], jnp.ndarray]
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Calculates factor contributions (A_omega, B_omega) using augmented states.
    B_omega is now based on gradients of the interior loss L_Omega.
    """
    
    initial_aug_state = AugmentedState.from_coords(interior_pts)
    
    aug_input_acts_per_layer_values, aug_pre_acts_per_layer_values, final_aug_output_for_loss = \
        _augmented_forward_cache(model_eval, initial_aug_state)

    # Define a function that computes the interior loss L_Omega based on a list of 
    # augmented pre-activations (S_out_l). This function will be differentiated.
    def compute_interior_loss_from_s_out_list(
        s_out_aug_list_for_grad: List[AugmentedState], # VARS for jax.grad.
        model_for_grad_b: eqx.Module, 
        initial_input_to_network: AugmentedState, 
        interior_pts_for_loss_calc: jnp.ndarray, 
        rhs_fn_for_loss_calc: Callable 
    ):
        
        _act_fn_b, _vmap_grad_act_b, _vmap_grad_grad_act_b = None, None, None
        for lo_b in model_for_grad_b.layers: 
            if isinstance(lo_b, eqx.nn.Lambda):
                sfn_grad_b = None
                try: lo_b.fn(jnp.array(0.0)); sfn_grad_b = lo_b.fn
                except: 
                    try: sfn_grad_b = lambda xs: lo_b.fn(jnp.array([xs]))[0]
                    except: pass
                if sfn_grad_b:
                    _act_fn_b = lo_b.fn
                    _vmap_grad_act_b = jax.vmap(jax.grad(sfn_grad_b))
                    _vmap_grad_grad_act_b = jax.vmap(jax.grad(jax.grad(sfn_grad_b)))
                break
        
        current_z_l = initial_input_to_network # This is Z_in_0
        
        linear_layer_indices_in_model = [i for i, layer in enumerate(model_for_grad_b.layers) if isinstance(layer, eqx.nn.Linear)]
        
        if len(s_out_aug_list_for_grad) != len(linear_layer_indices_in_model):
            raise ValueError("Mismatch: len(s_out_aug_list_for_grad) != num linear layers in model.")

        for k_linear_idx in range(len(linear_layer_indices_in_model)):
            # S_out_k is the k-th variable we are differentiating against.
            s_out_k = s_out_aug_list_for_grad[k_linear_idx] 
            
            model_idx_of_linear_k = linear_layer_indices_in_model[k_linear_idx]
            
            z_k = s_out_k # Default if no activation follows
            if model_idx_of_linear_k + 1 < len(model_for_grad_b.layers) and \
               isinstance(model_for_grad_b.layers[model_idx_of_linear_k + 1], eqx.nn.Lambda):
                if _act_fn_b and _vmap_grad_act_b and _vmap_grad_grad_act_b:
                    z_k = _propagate_activation_augmented(
                        s_out_k, _act_fn_b, 
                        _vmap_grad_act_b, _vmap_grad_grad_act_b
                    )
            current_z_l = z_k 
            
        final_lap_val_batch = current_z_l.laplacian.squeeze(-1) if current_z_l.laplacian.ndim == 2 else current_z_l.laplacian
        
        res_interior = final_lap_val_batch - rhs_fn_for_loss_calc(interior_pts_for_loss_calc)
        loss_val = 0.5 * jnp.mean(jnp.square(res_interior)) 
        return loss_val
    # --- End of compute_interior_loss_from_s_out_list ---
    
    grad_fn_B_final = jax.grad(compute_interior_loss_from_s_out_list, argnums=0)
    
    grads_for_b_omega_list_of_aug_states = grad_fn_B_final(
        aug_pre_acts_per_layer_values, 
        model_eval,                    
        initial_aug_state,             
        interior_pts,                  
        rhs_fn                         
    )
    
    factor_contributions = []
    linear_layers_list = _linear_layers(model_eval) # Ensure this list is available
    for idx, lin_layer_obj in enumerate(linear_layers_list): 
        z_in_l_minus_1_aug = aug_input_acts_per_layer[idx] # Use `aug_input_acts_per_layer` from first fwd pass
        a_contrib = z_in_l_minus_1_aug.concatenate_components() 
        
        grad_S_out_l_components = grads_for_b_omega_list_of_aug_states[idx]
        b_contrib = grad_S_out_l_components.concatenate_components() 
        
        factor_contributions.append((a_contrib, b_contrib))
        
    return factor_contributions


# Original _factor_terms, to be kept for boundary conditions (non-interior)
# Renamed to _standard_factor_terms for clarity.
def _standard_factor_terms(model_eval_std: eqx.Module, params_std: Any, pts_std: jnp.ndarray, fn_boundary_val_std: Callable):
    # This function is now only for boundary terms.
    # model_eval_std should already be combined with params_std by the caller if needed.
    # Or, params_std is passed and combined here. Let's assume model_eval_std is ready.
    
    y_pred, acts_std, pre_std, phi_std = _standard_forward_cache(model_eval_std, pts_std)
    
    res_boundary = y_pred.squeeze() - fn_boundary_val_std(pts_std)
    grad_out_boundary = res_boundary / pts_std.shape[0] 
    
    deltas_std = _standard_backward_pass(model_eval_std, pre_std, phi_std, grad_out_boundary)
    
    return acts_std, deltas_std
