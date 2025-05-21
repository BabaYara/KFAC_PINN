"""PINN-specific KFAC implementation.

This module implements a Kronecker-Factored Approximate Curvature optimiser
geared towards Physics-Informed Neural Networks as described in
``Tex/2025_05_20_e8fffb9338419e358febg.tex``. The implementation
closely follows Algorithm~1 in that document and maintains separate
Kronecker factors for the PDE operator term and the boundary term.

The optimiser is intentionally written with clarity in mind. It currently
supports sequential MLP models composed of ``equinox.nn.Linear`` layers
and activation functions. The augmented state propagation (e.g., for value,
gradient, and Laplacian components) is implemented directly for efficiency
with common second-order PDEs. For arbitrary PDE operators, this would involve
generalizing the augmented state and using general Taylor-mode AD, as detailed
in the referenced papers. Nevertheless, all four Kronecker factors are tracked
and used to precondition gradients.

"""
from __future__ import annotations

from typing import Callable, List, NamedTuple, Tuple, Any # Added Any

import equinox as eqx
import jax
import jax.numpy as jnp


# Augmented state for propagating (value, spatial_derivatives, laplacian)
class AugmentedState(NamedTuple):
    """
    Represents the augmented state propagated through the network.
    Its current structure (value, spatial_derivatives, laplacian) is specialized
    for PDEs requiring the Laplacian and first-order spatial derivatives.
    For general PDEs (as per Zeinhofer et al. Sec 3.3), this state would
    need to hold all S Taylor coefficients, and propagation would use
    general Taylor-mode Automatic Differentiation (AD) rules.
    """
    value: jnp.ndarray  # Batched: (batch, features)
    spatial_derivatives: List[jnp.ndarray] # List of d arrays, each (batch, features)
    laplacian: jnp.ndarray # Batched: (batch, features)

    @classmethod
    def from_coords(cls, coords: jnp.ndarray) -> "AugmentedState":
        """
        Initializes AugmentedState from input coordinates.
        coords: (batch, dim) - where dim is the number of spatial dimensions.
        """
        batch_size, dim = coords.shape
        value = coords # Input features are the coordinates themselves.

        spatial_derivatives_list = []
        for i in range(dim):
            # For the i-th spatial derivative (d/dx_i) applied to coords (x_0, ..., x_{d-1}),
            # the result is a vector that is 1 at the i-th component and 0 otherwise.
            # This should have shape (batch, dim) matching `value`.
            deriv_component = jnp.zeros_like(coords)
            deriv_component = deriv_component.at[:, i].set(1.0)
            spatial_derivatives_list.append(deriv_component)
        
        # Laplacian of input coordinates (x_i) is 0.
        # This should have shape (batch, dim) matching `value`.
        laplacian_val = jnp.zeros_like(coords)

        return cls(value, spatial_derivatives_list, laplacian_val)

    def concatenate_components(self, per_sample_norm: bool = False) -> jnp.ndarray:
        """
        Concatenates all S components.
        Output: (batch_size * S, features)
        If per_sample_norm is True, this indicates that the norm will be taken per sample later.
        """
        # self.value: (batch, features)
        # self.spatial_derivatives: List of d arrays, each (batch, features)
        # self.laplacian: (batch, features)
        # Stack them to create S "virtual" samples per original sample.
        s_components = [self.value] + self.spatial_derivatives + [self.laplacian]
        
        # All components should have the same (batch, features) shape.
        # Concatenate along a new leading axis (axis=0 for concatenate) then reshape,
        # or concatenate along existing batch axis if that's the goal.
        # The current KFAC code expects (S * batch, features).
        concatenated = jnp.concatenate(s_components, axis=0) # (S * batch, features)
        return concatenated

    @property
    def num_s_components(self):
        # 1 (value) + d (spatial_derivatives) + 1 (laplacian)
        # d is the number of spatial dimensions, derived from len of spatial_derivatives list.
        if not isinstance(self.spatial_derivatives, list):
            # This case might occur if the object is not properly initialized (e.g. during type checking)
            # or if it's an intermediate state in a JIT context where list structure is opaque.
            # However, for a NamedTuple, fields should be present.
            # Consider how this is used. If it's for sizing KFAC factors, it must be known at init.
            # For now, assume spatial_derivatives is always a list when this is called on a valid instance.
            raise ValueError("spatial_derivatives is not a list, cannot determine num_s_components.")
        return 1 + len(self.spatial_derivatives) + 1

def _propagate_linear_augmented(aug_state: AugmentedState, layer: eqx.nn.Linear) -> AugmentedState:
    """
    Propagates AugmentedState through an eqx.nn.Linear layer.
    This implementation is specialized for an AugmentedState with (value, spatial_derivatives, laplacian).
    A general version would propagate all S Taylor coefficients using Taylor-mode AD rules for linear layers.
    """
    # Bias only affects the 'value' component. Derivatives of bias are zero.
    new_value = aug_state.value @ layer.weight.T + layer.bias
    # Propagate each spatial derivative component
    new_spatial_derivatives = [s_deriv @ layer.weight.T for s_deriv in aug_state.spatial_derivatives]
    new_laplacian = aug_state.laplacian @ layer.weight.T # Laplacian propagation rule for linear layer
    
    return AugmentedState(new_value, new_spatial_derivatives, new_laplacian)

def _propagate_activation_augmented(
    aug_state_pre_activation: AugmentedState, 
    activation_fn: Callable, 
    vmap_activation_fn_grad: Callable, # vmapped jax.grad(activation_fn)
    vmap_activation_fn_grad_grad: Callable # vmapped jax.grad(jax.grad(activation_fn))
) -> AugmentedState:
    """
    Propagates AugmentedState through an element-wise activation function.
    Assumes activation_fn operates element-wise on `aug_state_pre_activation.value`.
    This implementation is specialized for an AugmentedState with (value, spatial_derivatives, laplacian)
    and uses specific propagation rules (e.g., Faà di Bruno's formula for Laplacian).
    A general version would propagate all S Taylor coefficients using general Taylor-mode AD rules for activations.
    """
    s_val = aug_state_pre_activation.value # (batch, features_in)
    s_spatial_derivatives = aug_state_pre_activation.spatial_derivatives # List of d arrays, each (batch, features_in)
    s_lap = aug_state_pre_activation.laplacian # (batch, features_in)

    # These grads are element-wise for each feature
    sigma_prime_s = vmap_activation_fn_grad(s_val)     # (batch, features_in)
    sigma_prime_prime_s = vmap_activation_fn_grad_grad(s_val) # (batch, features_in)

    val_out = activation_fn(s_val) # (batch, features_in)
    
    # Propagate each spatial derivative component
    # d_xk_out = sigma'(s) * s_xk
    new_spatial_derivatives_out = [sigma_prime_s * s_deriv for s_deriv in s_spatial_derivatives]
    
    # Element-wise computation for laplacian propagation (Eq. 7b in Dangel et al.)
    # lap_out = sigma'(s) * lap_s + sigma''(s) * sum_k (d_xk s)^2
    # All terms are (batch, features_in)
    sum_sq_spatial_derivs = sum(jnp.square(s_deriv) for s_deriv in s_spatial_derivatives)
    term_sum_sq_derivs = sigma_prime_prime_s * sum_sq_spatial_derivs
    lap_out = sigma_prime_s * s_lap + term_sum_sq_derivs 
    
    return AugmentedState(val_out, new_spatial_derivatives_out, lap_out)


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
    # damping: float = 1e-3 # Removed
    damping_omega: float = 1e-3
    damping_boundary: float = 1e-3
    damping_kfac_star_model: float = 1e-3
    decay: float = 0.95

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
            # Compute final laplacian using _augmented_forward_cache
            initial_aug_state_for_loss = AugmentedState.from_coords(interior)
            # The first two returned values are lists of intermediate states, not needed here.
            _, _, final_aug_output_for_loss = _augmented_forward_cache(m, initial_aug_state_for_loss)
            lap_from_aug = final_aug_output_for_loss.laplacian
            # Ensure lap_from_aug is squeezed if it has an extra dimension, e.g. (batch, 1) -> (batch,)
            if lap_from_aug.ndim == 2 and lap_from_aug.shape[-1] == 1:
                lap_from_aug = lap_from_aug.squeeze(-1)
            
            res_standard = lap_from_aug - rhs_fn(interior)
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
        state = PINNKFACState(step=state.step + 1, layers=tuple(new_layers)) # Factors A, B updated

        # KFAC* computations
        lin_indices = [k for k, l_obj in enumerate(model.layers) if isinstance(l_obj, eqx.nn.Linear)]

        current_preconditioned_delta_parts = [] # List of {'weight': gw_kfac, 'bias': gb_kfac}
        previous_update_parts = []            # List of {'weight': lf.mw, 'bias': lf.mb} (delta_{t-1})
        
        # Loop through linear layers in ascending order of model definition
        for kfac_idx, model_layer_idx in enumerate(lin_indices):
            lf = state.layers[kfac_idx] # KFAC factors (A,B) and momentum (mw, mb) from previous step or factor update
            
            # Damping for preconditioning (same as before, G_lambda = G + lambda*I)
            # The factors lf.A_omega etc. are EMA of undamped A_l A_l^T.
            # Damping for preconditioning is added here.
            # The KFAC* quadratic model uses G + lambda_quad*I, where lambda_quad is self.damping.
            # This means Gv products should use factors without this preconditioning damping.
            # However, the preconditioned gradient Delta_t = (G + lambda_precond*I)^-1 g_t.
            # Let's assume the existing preconditioning damping `self.damping` is lambda_precond.
            # The KFAC* paper's lambda is lambda_quad. For now, let's assume they are the same `self.damping`.

            # Factors for preconditioning, including damping for stable inversion
            Aw_damped = lf.A_omega + self.damping_omega * jnp.eye(lf.A_omega.shape[0])
            Gw_damped = lf.B_omega + self.damping_omega * jnp.eye(lf.B_omega.shape[0])
            Aw_b_damped = lf.A_boundary + self.damping_boundary * jnp.eye(lf.A_boundary.shape[0])
            Gw_b_damped = lf.B_boundary + self.damping_boundary * jnp.eye(lf.B_boundary.shape[0])

            eig_A, UA = jnp.linalg.eigh(Aw_damped)
            eig_G, UG = jnp.linalg.eigh(Gw_damped)
            eig_Ab, UAb = jnp.linalg.eigh(Aw_b_damped)
            eig_Gb, UGb = jnp.linalg.eigh(Gw_b_damped)
            
            current_layer_grads = grads.layers[model_layer_idx] # grads for model.layers[model_layer_idx]
            gw = current_layer_grads.weight
            gb = current_layer_grads.bias
            
            # Compute preconditioned gradient for this layer (Delta_t components)
            gw_kfac_layer = UA.T @ gw @ UG
            # Denominator for KFAC preconditioning: (eig_A tensor eig_G) + (eig_Ab tensor eig_Gb)
            # This is effectively (G_omega + G_boundary + damping*I)^-1 applied to grads
            precond_denominator = (eig_A[:, None] * eig_G[None, :]) + \
                                  (eig_Ab[:, None] * eig_Gb[None, :]) # Damping already in eigs
            gw_kfac_layer = gw_kfac_layer / precond_denominator
            gw_kfac_layer = UA @ gw_kfac_layer @ UG.T

            gb_kfac_layer = UG.T @ gb 
            bias_precond_denominator = eig_G + eig_Gb # Damping already in eigs
            gb_kfac_layer = gb_kfac_layer / bias_precond_denominator
            gb_kfac_layer = UG @ gb_kfac_layer
            
            current_preconditioned_delta_parts.append({'weight': gw_kfac_layer, 'bias': gb_kfac_layer})
            previous_update_parts.append({'weight': lf.mw, 'bias': lf.mb}) # lf.mw, lf.mb is delta_{t-1}

        # Helper to build a params-like PyTree from parts list
        def _build_tree_from_parts(parts_list_local, template_params_local, linear_indices_local):
            zero_params = jax.tree_map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x, template_params_local)
            filled_tree = zero_params
            if len(parts_list_local) != len(linear_indices_local):
                raise ValueError(f"Parts list length {len(parts_list_local)} != linear indices length {len(linear_indices_local)}")

            for k_idx, p_dict in enumerate(parts_list_local):
                model_l_idx = linear_indices_local[k_idx]
                
                # Ensure that the target in the template is actually a Linear layer structure
                # This check is more for safety, lin_indices should guarantee this.
                if not hasattr(template_params_local.layers[model_l_idx], 'weight') or \
                   not hasattr(template_params_local.layers[model_l_idx], 'bias'):
                    raise ValueError(f"Target layer {model_l_idx} in template is not a Linear layer.")

                weight_path = lambda tree: tree.layers[model_l_idx].weight
                bias_path = lambda tree: tree.layers[model_l_idx].bias
                
                filled_tree = eqx.tree_at(weight_path, filled_tree, p_dict['weight'])
                filled_tree = eqx.tree_at(bias_path, filled_tree, p_dict['bias'])
            return filled_tree

        # Build PyTrees for KFAC* computations
        # params is the PyTree of current model parameters theta_t
        current_preconditioned_delta_tree = _build_tree_from_parts(current_preconditioned_delta_parts, params, lin_indices) # Delta_t
        previous_update_tree = _build_tree_from_parts(previous_update_parts, params, lin_indices) # delta_t
        
        # Compute Gramian-vector products.
        # compute_gramian_vector_product expects kfac_factors_layers to be the *undamped* factors for G.
        # state.layers contains EMA of (A_l A_l^T) etc. These are appropriate for G.
        g_Delta_parts = compute_gramian_vector_product(current_preconditioned_delta_tree, model, state.layers)
        g_delta_parts = compute_gramian_vector_product(previous_update_tree, model, state.layers)
        
        g_Delta_tree = _build_tree_from_parts(g_Delta_parts, params, lin_indices) # G Delta_t
        g_delta_tree = _build_tree_from_parts(g_delta_parts, params, lin_indices) # G delta_t

        # Compute scalar terms for KFAC*
        # Note: grads is the original gradient g_t at theta_t
        Delta_g_Delta = tree_dot(current_preconditioned_delta_tree, g_Delta_tree)
        delta_g_delta = tree_dot(previous_update_tree, g_delta_tree)
        Delta_g_delta = tree_dot(current_preconditioned_delta_tree, g_delta_tree) # or tree_dot(previous_update_tree, g_Delta_tree)

        Delta_norm_sq = tree_dot(current_preconditioned_delta_tree, current_preconditioned_delta_tree)
        delta_norm_sq = tree_dot(previous_update_tree, previous_update_tree)
        Delta_delta_dot = tree_dot(current_preconditioned_delta_tree, previous_update_tree)
        
        # Dot products with original gradient g_t
        # Delta_t = (G + lambda_precond*I)^-1 g_t. So g_t^T Delta_t = Delta_t^T (G + lambda_precond*I) Delta_t
        # The KFAC* paper uses -g_t^T Delta_t as linear_coeff_alpha.
        # This is Delta_t^T (G_damped) Delta_t where G_damped includes preconditioning damping.
        # If we define Delta_t as the "pure" KFAC preconditioned grad G^-1 g_t (undamped G),
        # then g_t^T Delta_t = (G Delta_t)^T Delta_t = Delta_t^T G Delta_t.
        # The current `gw_kfac_layer` is (G_damped)^-1 g_layer. So `current_preconditioned_delta_tree` is (G_damped)^-1 g_t.
        # Let's call this Delta_damped_t.
        # Delta_grad_dot = tree_dot(current_preconditioned_delta_tree, grads) # = ((G+lambda_p*I)^-1 g_t)^T g_t
        # delta_grad_dot = tree_dot(previous_update_tree, grads)             # = (delta_t)^T g_t
        
        # According to Dangel et al. (Eq. 21), linear terms are -Delta_t^T g and -delta_t^T g
        # Here Delta_t is the preconditioned direction from KFAC (G+lambda I)^-1 g.
        # So this seems correct.
        Delta_grad_dot = -tree_dot(current_preconditioned_delta_tree, grads) 
        delta_grad_dot = -tree_dot(previous_update_tree, grads)
        
        # KFAC* quadratic model Hessian is G_quad = G_undamped + lambda_quad*I
        # quad_coeff_alpha = Delta_t^T G_quad Delta_t = Delta_t^T G_undamped Delta_t + lambda_quad * Delta_t^T Delta_t
        # Delta_g_Delta = Delta_t^T G_undamped Delta_t (computed with undamped factors in compute_gramian_vector_product)
        # lambda_quad is self.damping_kfac_star_model
        
        lambda_quad_damping = self.damping_kfac_star_model # Damping for the KFAC* quadratic model G + lambda_quad*I
        
        quad_coeff_alpha    = Delta_g_Delta + lambda_quad_damping * Delta_norm_sq
        quad_coeff_mu       = delta_g_delta + lambda_quad_damping * delta_norm_sq
        quad_coeff_alpha_mu = Delta_g_delta + lambda_quad_damping * Delta_delta_dot # This is the off-diagonal term

        linear_coeff_alpha = Delta_grad_dot # Already negated: -Delta_t^T g_t
        linear_coeff_mu    = delta_grad_dot # Already negated: -delta_t^T g_t

        # Solve the 2x2 system for alpha_star, mu_star
        # System: M @ [alpha, mu]^T = [linear_alpha, linear_mu]^T
        # M = [[quad_coeff_alpha, quad_coeff_alpha_mu], 
        #      [quad_coeff_alpha_mu, quad_coeff_mu]]
        
        # state.step was incremented at the start of this KFAC* block.
        # So, state.step == 1 corresponds to the first actual optimization update.
        if state.step == 1: # First optimization step, no meaningful previous_update_tree
            mu_star = 0.0
            # Check for quad_coeff_alpha being zero or very small to avoid division by zero
            if jnp.abs(quad_coeff_alpha) < 1e-8: # Heuristic threshold
                alpha_star = 0.0 # Avoid division by zero, effectively no update in this direction
            else:
                alpha_star = linear_coeff_alpha / quad_coeff_alpha
        else:
            # Matrix M for the 2x2 system
            M = jnp.array([[quad_coeff_alpha, quad_coeff_alpha_mu],
                           [quad_coeff_alpha_mu, quad_coeff_mu]])
            # Vector b for the 2x2 system
            b_vec = jnp.array([linear_coeff_alpha, linear_coeff_mu])
            
            # Solve M x = b for x = [alpha_star, mu_star]
            # Add solver damping for stability
            solver_damping = 1e-6 
            try:
                solution = jnp.linalg.solve(M + solver_damping * jnp.eye(2), b_vec)
                alpha_star, mu_star = solution[0], solution[1]
            except jnp.linalg.LinAlgError:
                # Fallback if solver fails (e.g. singular matrix)
                # Use only the current preconditioned gradient, similar to first step, but with potentially non-zero mu if previous update was large
                # A simple fallback: ignore coupling, solve independently if possible, or just use alpha.
                print("Warning: KFAC* 2x2 system solve failed. Using simplified update.")
                if jnp.abs(quad_coeff_alpha) < 1e-8:
                    alpha_star = 0.0
                else:
                    alpha_star = linear_coeff_alpha / quad_coeff_alpha
                mu_star = 0.0 # Or a more sophisticated fallback for mu_star if delta_g_delta is safe

        # Compute the final update direction: final_delta = alpha_star * Delta_t + mu_star * delta_t
        # Delta_t is current_preconditioned_delta_tree, delta_t is previous_update_tree
        final_update_tree = jax.tree_map(
            lambda delta, prev_delta: alpha_star * delta + mu_star * prev_delta,
            current_preconditioned_delta_tree,
            previous_update_tree
        )

        # Update parameters: params_new = params - final_delta_tree (not scaled by self.lr)
        final_params = jax.tree_map(lambda p, u: p - u, params, final_update_tree)

        # Store final_update_tree as the new momentum (mw, mb) in state.layers
        # Need to reconstruct the state.layers tuple with updated mw, mb
        new_kfac_state_layers_list = []
        final_update_tree_parts = [] # This needs to be deconstructed from final_update_tree

        # Deconstruct final_update_tree into parts list
        for kfac_idx, model_layer_idx in enumerate(lin_indices):
            layer_update_w = final_update_tree.layers[model_layer_idx].weight
            layer_update_b = final_update_tree.layers[model_layer_idx].bias
            final_update_tree_parts.append({'weight': layer_update_w, 'bias': layer_update_b})

        for kfac_idx, lf_old in enumerate(state.layers): # state.layers has A,B factors already updated
            new_mw = final_update_tree_parts[kfac_idx]['weight']
            new_mb = final_update_tree_parts[kfac_idx]['bias']
            new_kfac_state_layers_list.append(
                LayerFactors(
                    A_omega=lf_old.A_omega, B_omega=lf_old.B_omega,
                    A_boundary=lf_old.A_boundary, B_boundary=lf_old.B_boundary,
                    mw=new_mw, mb=new_mb # New momentum
                )
            )
        
        final_kfac_state_layers_tuple = tuple(new_kfac_state_layers_list)
        
        # Create the final PINNKFACState with updated momentum terms
        final_state = PINNKFACState(step=state.step, layers=final_kfac_state_layers_tuple) 
        new_model = eqx.combine(static, final_params)
        return new_model, final_state


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _get_activation_derivatives(
    activation_fn: Callable, 
    layer_description: str
) -> Tuple[Callable, Callable, Callable]:
    """
    Computes first and second derivatives of an activation function using jax.grad.

    Args:
        activation_fn: The activation function.
        layer_description: A string describing the layer, for error messages.

    Returns:
        A tuple (activation_fn, grad_fn, grad_grad_fn).

    Raises:
        TypeError: If jax.grad fails to compute derivatives, indicating the
                   function may not be JAX-differentiable.
    """
    try:
        # Try to evaluate the function with a tracer to catch issues early
        # This helps ensure the function is JAX-traceable with a scalar float.
        jax.eval_shape(activation_fn, jnp.array(0.0))
        
        grad_fn = jax.grad(activation_fn)
        # Test grad_fn
        jax.eval_shape(grad_fn, jnp.array(0.0))

        grad_grad_fn = jax.grad(grad_fn)
        # Test grad_grad_fn
        jax.eval_shape(grad_grad_fn, jnp.array(0.0))
        
        return activation_fn, grad_fn, grad_grad_fn
    except Exception as e:
        # Catch a broad range of JAX errors that might occur if fn is not diffable
        error_message = (
            f"Failed to compute derivatives for activation function in {layer_description}. "
            "Please ensure the activation function is JAX-differentiable and element-wise. "
            f"Underlying error: {e}"
        )
        raise TypeError(error_message) from e

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
    
    layers = list(model.layers)
    current_layer_index = 0
    processed_linear_layers = 0 # Counter for naming in errors

    while current_layer_index < len(layers):
        layer = layers[current_layer_index]
        if isinstance(layer, eqx.nn.Linear):
            acts.append(h) 
            s = layer(h)   
            pre.append(s)
            processed_linear_layers +=1
            
            # Check if the next layer is a Lambda (activation) layer
            if current_layer_index + 1 < len(layers) and isinstance(layers[current_layer_index+1], eqx.nn.Lambda):
                activation_layer_obj = layers[current_layer_index+1]
                layer_description = f"activation layer {type(activation_layer_obj.fn).__name__} after Linear layer {processed_linear_layers-1}"
                
                # Use the helper to get derivatives
                # _standard_forward_cache only needs the first derivative (grad_fn)
                _, grad_fn, _ = _get_activation_derivatives(activation_layer_obj.fn, layer_description)
                
                vmapped_grad_fn = jax.vmap(grad_fn)
                phi.append(vmapped_grad_fn(s))
                
                h = activation_layer_obj(s) # Apply activation
                current_layer_index += 1 # Consume the activation layer as well
            else: 
                # No activation layer follows this linear layer
                phi.append(jnp.ones_like(s)) 
                h = s # Output of linear layer is input to next
        else: 
            # Non-linear layer (could be an activation layer not immediately after a Linear, or other type)
            # If it's a Lambda layer, it's an activation function.
            # However, phi is specifically for derivatives of activations *after* linear layers.
            # So, we just apply the layer.
            h = layer(h) 
        current_layer_index += 1
        
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
    layers = list(model.layers)
    current_layer_index = 0
    processed_linear_layers = 0 # For naming in errors

    while current_layer_index < len(layers):
        layer = layers[current_layer_index]
        layer_description_prefix = f"layer {current_layer_index} ({type(layer).__name__})"

        if isinstance(layer, eqx.nn.Linear):
            aug_input_acts.append(current_aug_state)
            current_aug_state = _propagate_linear_augmented(current_aug_state, layer)
            aug_pre_acts.append(current_aug_state) 
            processed_linear_layers += 1

            # Check if next layer is an activation function
            if current_layer_index + 1 < len(layers) and isinstance(layers[current_layer_index+1], eqx.nn.Lambda):
                activation_layer_obj = layers[current_layer_index+1]
                act_fn_raw = activation_layer_obj.fn
                desc = f"activation layer {type(act_fn_raw).__name__} after Linear layer {processed_linear_layers-1}"
                
                act_fn, grad_fn, grad_grad_fn = _get_activation_derivatives(act_fn_raw, desc)
                
                vmap_grad_local = jax.vmap(grad_fn)
                vmap_grad_grad_local = jax.vmap(grad_grad_fn)
                
                current_aug_state = _propagate_activation_augmented(
                    current_aug_state, 
                    act_fn, # Use the (potentially wrapped) act_fn from helper
                    vmap_grad_local,
                    vmap_grad_grad_local
                )
                current_layer_index += 1 # Consume the activation layer
            # If no activation layer follows Linear, current_aug_state is already updated by _propagate_linear_augmented
        
        elif isinstance(layer, eqx.nn.Lambda): 
            act_fn_raw = layer.fn
            desc = f"standalone activation layer {type(act_fn_raw).__name__} at model index {current_layer_index}"
            
            act_fn, grad_fn, grad_grad_fn = _get_activation_derivatives(act_fn_raw, desc)

            vmap_grad_local = jax.vmap(grad_fn)
            vmap_grad_grad_local = jax.vmap(grad_grad_fn)
            
            current_aug_state = _propagate_activation_augmented(
                current_aug_state, 
                act_fn, 
                vmap_grad_local, 
                vmap_grad_grad_local
            )
        
        else: # For non-Linear, non-Lambda layers (e.g. custom layers, Dropout, Identity, etc.)
            # Default behavior: apply layer to value, zero out derivatives and laplacian
            # as their propagation rule through an arbitrary layer is unknown.
            # This assumes such layers might change feature dimensions, making `zeros_like` tricky.
            # A robust way is to pass the value, and if the output shape changes, create new zero derivatives.
            
            original_value_shape = current_aug_state.value.shape
            new_val = layer(current_aug_state.value)
            new_value_shape = new_val.shape

            if new_value_shape != original_value_shape and len(new_value_shape) == 2: # (batch, new_features)
                num_new_features = new_value_shape[1]
                # Create new zero derivatives matching the new feature dimension
                # Assuming spatial derivatives and laplacian should also have this new feature dimension.
                # This is a strong assumption; complex layers might have different rules.
                spatial_derivs_zeros = [jnp.zeros((new_value_shape[0], num_new_features)) for _ in current_aug_state.spatial_derivatives]
                laplacian_zero = jnp.zeros((new_value_shape[0], num_new_features))
            elif new_value_shape == original_value_shape:
                spatial_derivs_zeros = [jnp.zeros_like(s_deriv) for s_deriv in current_aug_state.spatial_derivatives]
                laplacian_zero = jnp.zeros_like(current_aug_state.laplacian)
            else: # Unexpected shape change (e.g. rank change)
                print(f"Warning: Layer {current_layer_index} ({layer}) changed value tensor shape from {original_value_shape} to {new_value_shape} in an unexpected way. Zeroing derivatives based on original feature count.")
                spatial_derivs_zeros = [jnp.zeros_like(s_deriv) for s_deriv in current_aug_state.spatial_derivatives]
                laplacian_zero = jnp.zeros_like(current_aug_state.laplacian)

            current_aug_state = AugmentedState(new_val, spatial_derivs_zeros, laplacian_zero)
        current_layer_index += 1 # Increment the loop counter for all layer types
        
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
        
        current_z_l = initial_input_to_network # This is Z_in_0
        
        linear_layer_indices_in_model = [i for i, layer in enumerate(model_for_grad_b.layers) if isinstance(layer, eqx.nn.Linear)]
        
        if len(s_out_aug_list_for_grad) != len(linear_layer_indices_in_model):
            raise ValueError("Mismatch: len(s_out_aug_list_for_grad) != num linear layers in model.")

        # Loop through the model layers, reconstructing the augmented state propagation
        # This loop needs to mirror the logic in _augmented_forward_cache for activations
        
        # Store pre-computed activation derivatives for layers in model_for_grad_b
        # This avoids re-computing them in the loop below.
        # This map will store: model_layer_index -> (act_fn, vmap_grad_fn, vmap_grad_grad_fn)
        activation_derivatives_map = {} 
        for k_model_idx, layer_obj_k in enumerate(model_for_grad_b.layers):
            if isinstance(layer_obj_k, eqx.nn.Lambda):
                act_fn_raw = layer_obj_k.fn
                desc = f"activation layer {type(act_fn_raw).__name__} at model index {k_model_idx} (in _compute_interior_loss_from_s_out_list)"
                
                # _get_activation_derivatives returns scalar derivative functions
                act_fn, grad_fn, grad_grad_fn = _get_activation_derivatives(act_fn_raw, desc)
                
                activation_derivatives_map[k_model_idx] = (
                    act_fn, # Original (or wrapped by helper) activation function
                    jax.vmap(grad_fn),
                    jax.vmap(grad_grad_fn)
                )
        
        # current_z_l starts as initial_input_to_network (Z_in_0, i.e. aug_state from coords)
        # We are processing s_out_aug_list_for_grad, where each element is an S_out_l
        # (output of a linear layer, pre-activation)
        
        # The loop should effectively be: Z_in_l = activate(S_out_{l-1}); S_out_l = linear_l(Z_in_l)
        # But here, s_out_aug_list_for_grad *are* the S_out_l values (vars for jax.grad).
        # We need to propagate from one S_out_l to the next S_out_l through an activation.
        # The variable current_z_l will represent Z_out_l (output of activation, input to next linear)
        # or S_out_l if no activation follows.

        for k_linear_idx in range(len(linear_layer_indices_in_model)):
            # s_out_k is S_out_l, the output of the k-th linear layer (from s_out_aug_list_for_grad)
            s_out_k = s_out_aug_list_for_grad[k_linear_idx] 
            
            model_idx_of_linear_k = linear_layer_indices_in_model[k_linear_idx]
            
            # current_z_l is now the state *after* the k-th linear layer, i.e. s_out_k
            current_z_l = s_out_k # This is S_out_l
            
            # If an activation follows this linear layer, propagate current_z_l through it
            if model_idx_of_linear_k + 1 < len(model_for_grad_b.layers) and \
               isinstance(model_for_grad_b.layers[model_idx_of_linear_k + 1], eqx.nn.Lambda):
                
                activation_model_idx = model_idx_of_linear_k + 1
                # act_derivatives will contain (act_fn, vmap_grad_fn, vmap_grad_grad_fn)
                # or the call to .get() will return None if no activation layer was there.
                act_derivatives = activation_derivatives_map.get(activation_model_idx)

                if act_derivatives: # Ensure there is an activation layer to process
                    act_fn_local, vmap_grad_local, vmap_grad_grad_local = act_derivatives
                    current_z_l = _propagate_activation_augmented(
                        current_z_l, # This is s_out_k, the pre-activation state
                        act_fn_local, # The actual activation function callable
                        vmap_grad_local, 
                        vmap_grad_grad_local
                    )
                # If act_derivatives is None, means no Lambda layer was at activation_model_idx,
                # so current_z_l (which is S_out_l) correctly passes through.
                # The _get_activation_derivatives call during map population would have raised
                # a TypeError if a Lambda layer was present but not differentiable.
            
            # If no activation layer follows, current_z_l remains S_out_l.
            # This current_z_l then becomes the input for the *next* linear layer's processing
            # in the conceptual forward pass being reconstructed here for differentiation.
            # The final current_z_l after the loop will be the Z_out_L (final activated output)
            # or S_out_L (final linear output if no activation at the end).
            
        # The .laplacian access should be fine with the new AugmentedState structure.
        # current_z_l is the state after the last processed layer (linear or activation)
        final_lap_val_batch = current_z_l.laplacian.squeeze(-1) if current_z_l.laplacian.ndim == 2 and current_z_l.laplacian.shape[-1] == 1 else current_z_l.laplacian
        
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
        # Corrected to use aug_input_acts_per_layer_values from the initial _augmented_forward_cache call
        z_in_l_minus_1_aug = aug_input_acts_per_layer_values[idx] 
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


def tree_dot(tree1: Any, tree2: Any) -> jnp.ndarray:
    """Computes the dot product of two PyTrees with the same structure."""
    leaves1, _ = jax.tree_util.tree_flatten(tree1)
    leaves2, _ = jax.tree_util.tree_flatten(tree2)
    if len(leaves1) != len(leaves2):
        raise ValueError("PyTrees must have the same number of leaves.")
    
    dot_product = jnp.array(0.0)
    for leaf1, leaf2 in zip(leaves1, leaves2):
        if not isinstance(leaf1, jnp.ndarray) or not isinstance(leaf2, jnp.ndarray):
            # This might happen with static components if not filtered out before calling
            # For safety, skip non-array leaves or ensure inputs are filtered.
            continue
        if leaf1.shape != leaf2.shape:
            raise ValueError(f"Leaves must have the same shape, got {leaf1.shape} and {leaf2.shape}")
        dot_product += jnp.sum(leaf1 * leaf2)
    return dot_product


def compute_gramian_vector_product(
    vector_tree: Any, # PyTree like params (weights & biases for each layer)
    model_for_structure: eqx.Module, # Used to get linear layer indices and structure
    kfac_factors_layers: Tuple[LayerFactors, ...], # From PINNKFACState.layers
) -> Any:
    """
    Computes the product G @ vector_tree, where G is the KFAC approximation.
    G = G_omega + G_boundary.
    For weights of a layer l: (G_l V)_w = (B_omega V_w A_omega^T) + (B_boundary V_w A_boundary^T)
    For biases of a layer l: (G_l V)_b = (B_omega V_b) + (B_boundary V_b)
    (Note: A_omega is (n,n), B_omega is (m,m). V_w is (m,n). B V A^T is (m,m)(m,n)(n,n) = (m,n))
    """
    
    vector_params, _ = eqx.partition(vector_tree, eqx.is_array) # Ensure we have a params-like structure
    
    # Prepare a list to store parts of the output tree
    output_parts = [] 
    
    all_model_layers = list(model_for_structure.layers)
    lin_indices = [k for k, layer_obj in enumerate(all_model_layers) if isinstance(layer_obj, eqx.nn.Linear)]
    
    if len(lin_indices) != len(kfac_factors_layers):
        raise ValueError("Mismatch between number of linear layers in model and KFAC factors.")

    current_kfac_factor_idx = 0
    for i, layer_obj in enumerate(all_model_layers):
        if isinstance(layer_obj, eqx.nn.Linear):
            lf = kfac_factors_layers[current_kfac_factor_idx]
            
            # Get corresponding weight and bias from the input vector_tree
            # Assuming vector_tree has a .layers attribute similar to an Equinox model
            vw = vector_tree.layers[i].weight # V_w, shape (out_features, in_features)
            vb = vector_tree.layers[i].bias   # V_b, shape (out_features,)

            # Omega part for weights: B_omega @ vw @ A_omega.T
            # lf.A_omega is (in_features, in_features), lf.B_omega is (out_features, out_features)
            g_vw_omega = lf.B_omega @ vw @ lf.A_omega.T 
            
            # Boundary part for weights: B_boundary @ vw @ A_boundary.T
            g_vw_boundary = lf.B_boundary @ vw @ lf.A_boundary.T
            
            final_g_vw = g_vw_omega + g_vw_boundary
            
            # Omega part for biases: B_omega @ vb
            g_vb_omega = lf.B_omega @ vb
            
            # Boundary part for biases: B_boundary @ vb
            g_vb_boundary = lf.B_boundary @ vb
            
            final_g_vb = g_vb_omega + g_vb_boundary
            
            # Store as a LayerFactors-like structure (or just the computed parts)
            # For now, let's assume we're rebuilding a full PyTree for the output.
            # This requires knowing the original PyTree structure of params.
            # A simpler way: create a list of (weight_prod, bias_prod) and then tree_unflatten.
            output_parts.append(eqx.nn.Linear(weight=final_g_vw, bias=final_g_vb, in_features=vw.shape[1], out_features=vw.shape[0], key=None)) # Key not needed for data
            current_kfac_factor_idx += 1
        else:
            # For non-Linear layers, Gv is conceptually zero or identity if v is zero there.
            # If vector_tree has components for these layers, they should be zeroed.
            # Or, ensure vector_tree only has components for linear layers.
            # For now, just pass through the original layer from model_for_structure if it's not Linear,
            # but this means the output PyTree won't match vector_tree if vector_tree had non-array parts here.
            # Safest: output tree should match structure of vector_tree.
            # If vector_tree.layers[i] is not Linear, what to do?
            # The input `vector_tree` should ideally only contain parameters for which KFAC factors exist (Linear layers).
            # The preconditioned updates and momentum terms (lf.mw, lf.mb) are stored per KFAC factor,
            # so they align with linear layers.
            # Let's build a new model structure containing only these updated linear layers.
            # This means the output PyTree will be a model with only Linear layers (or their Gv products).
             output_parts.append(all_model_layers[i]) # Pass through non-linear layers as-is from model_for_structure


    # Reconstruct the PyTree. This assumes the output structure is like a Sequential model.
    # This needs to be robust. If vector_tree is params, output should also be params-like.
    # A more robust way: get tree_def from vector_tree and unflatten.
    # However, vector_tree might be `grads` or `params` or `state.layers[k].mw/mb` (which are not full PyTrees).
    
    # Let's make compute_gramian_vector_product return a list of (weight_gv, bias_gv) tuples,
    # and the main `step` function will build the PyTrees it needs from this list.
    # This avoids making assumptions about the overall PyTree structure here.
    
    # Revised plan for output:
    # Return a PyTree that has the same structure as `vector_tree`, but with Gv values.
    # This requires vector_tree to be a PyTree of parameters (like model).
    
    # Create a dummy list of Gv products for linear layers, and non-array for others.
    # This is getting complicated. Let's simplify:
    # The `vector_tree` will always be a PyTree of the parameters (like `params` or `grads`).
    # The output should be a PyTree of the same structure.
    
    # Initialize an empty PyTree like vector_tree but with zeros.
    # Then fill in the Gv products for the linear layers.
    
    # Simpler approach: construct a list of the Gv products for linear layers only.
    # The caller (`step` method) will be responsible for creating full PyTrees if needed.
    # This seems more modular.
    
    gv_products_list = [] # List of (gv_weight, gv_bias)
    
    # We need to iterate based on kfac_factors_layers as that defines what we have G for.
    # And map these back to the right place in vector_tree.
    # lin_indices from model_for_structure gives the indices in model.layers that are Linear.
    
    for kfac_idx, lf_factors in enumerate(kfac_factors_layers):
        model_layer_idx = lin_indices[kfac_idx] # Get the actual model layer index
        
        # Get corresponding weight and bias from the input vector_tree
        # This assumes vector_tree is a PyTree with a .layers attribute like a model.
        try:
            current_layer_in_vector_tree = vector_tree.layers[model_layer_idx]
            vw = current_layer_in_vector_tree.weight
            vb = current_layer_in_vector_tree.bias
        except AttributeError:
             raise ValueError("vector_tree is expected to have a .layers attribute similar to an Equinox model.")

        g_vw_omega = lf_factors.B_omega @ vw @ lf_factors.A_omega.T 
        g_vw_boundary = lf_factors.B_boundary @ vw @ lf_factors.A_boundary.T
        final_g_vw = g_vw_omega + g_vw_boundary
            
        g_vb_omega = lf_factors.B_omega @ vb
        g_vb_boundary = lf_factors.B_boundary @ vb
        final_g_vb = g_vb_omega + g_vb_boundary
            
        gv_products_list.append({'weight': final_g_vw, 'bias': final_g_vb})

    # The caller will use this list to build a full PyTree matching `params` or `grads`.
    # For example, by creating a zero tree and then using tree_at to fill values.
    return gv_products_list
