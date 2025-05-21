"""PINN-specific KFAC implementation.

This module implements a Kronecker-Factored Approximate Curvature optimiser
geared towards Physics-Informed Neural Networks as described in
``Tex/2025_05_20_e8fffb9338419e358febg.tex``. The implementation
closely follows Algorithm~1 in that document and maintains separate
Kronecker factors for the PDE operator term and the boundary term.

The optimiser is intentionally written with clarity in mind. It currently
supports sequential MLP models composed of ``equinox.nn.Linear`` layers
and activation functions. 
The propagation of derivatives (for computing PDE residuals) uses a generalized
`AugmentedState` that stores Taylor coefficients. This allows for handling
various derivative orders required by different PDEs. For common second-order PDEs,
this involves propagating value (0th order), gradient (1st order), and components
for second-order derivatives (e.g., to compute the Laplacian).

The optimiser supports two main update variants:
1.  **"kfac_star"**: Implements the KFAC* algorithm (Martens & Grosse, 2015),
    which solves a 2x2 quadratic subproblem to find optimal step sizes for
    the current preconditioned gradient and the previous update direction.
2.  **"kfac_standard"**: Implements a more standard KFAC update with optional
    momentum and line search for determining the step size.

All four Kronecker factors (A_omega, B_omega, A_boundary, B_boundary) are tracked
and used to precondition gradients.
"""
from __future__ import annotations

from typing import Callable, List, NamedTuple, Tuple, Any 

import equinox as eqx
import jax
import jax.numpy as jnp


class AugmentedState(NamedTuple):
    """
    Represents the augmented state propagated through the network using Taylor coefficients.

    The primary field `taylor_coefficients` is a list of JAX arrays. The k-th
    element in this list, `taylor_coefficients[k]`, stores all k-th order Taylor
    coefficients of the network's activations with respect to the input spatial
    coordinates.

    Each JAX array `taylor_coefficients[k]` has the shape:
    `(batch_size, num_components_k, features)`, where:
    - `batch_size`: The number of input samples.
    - `num_components_k`: The number of distinct k-th order derivative components
      being tracked. For example:
        - k=0 (value): `num_components_k` is 1.
        - k=1 (gradients): `num_components_k` is `spatial_dim` (e.g., df/dx, df/dy).
        - k=2 (2nd derivatives): `num_components_k` could be `spatial_dim` (if
          tracking pure second derivatives like d^2f/dx^2, d^2f/dy^2) or
          `spatial_dim * spatial_dim` (if tracking all Hessian components).
          The exact structure depends on the PDE and how `from_coords` is configured.
    - `features`: The number of output features of the current layer.

    The `pde_order` and `spatial_dim` (passed to `from_coords` or propagation rules)
    implicitly define the structure and size of these coefficient tensors.
    """
    taylor_coefficients: List[jnp.ndarray]

    @classmethod
    def from_coords(cls, coords: jnp.ndarray, pde_order: int, spatial_dim: int) -> "AugmentedState":
        """
        Initializes AugmentedState from input spatial coordinates.

        This function sets up the initial Taylor coefficients for the input layer.
        The 0-th order coefficient is the coordinates themselves. Higher-order
        coefficients represent derivatives of the coordinates.

        Args:
            coords: A JAX array of shape `(batch_size, spatial_dim)` representing
                the input spatial coordinates.
            pde_order: The maximum derivative order required for the PDE. This
                determines how many Taylor coefficient tensors are initialized
                (pde_order + 1).
            spatial_dim: The number of spatial dimensions of the input coordinates.
                This influences `num_components_k` for k=1 and k=2.

        Returns:
            An `AugmentedState` instance initialized for the input layer.
            - `taylor_coefficients[0]` (value): `coords`, expanded to
              `(batch_size, 1, spatial_dim)`.
            - `taylor_coefficients[1]` (gradients): Identity matrices for each sample,
              representing `d(coords)/d(coords_i)`, shape `(batch_size, spatial_dim, spatial_dim)`.
            - `taylor_coefficients[2]` (2nd derivatives): Zero tensors, shape
              `(batch_size, spatial_dim, spatial_dim)`, assuming 2nd order components
              are $d^2(coords)/dx_i dx_j$, which are all zero.
            - Higher order coefficients (k > 2) are initialized to zero tensors with
              a placeholder number of components if `pde_order` requires them.
        """
        batch_size = coords.shape[0]
        num_features = spatial_dim # At the input layer, features are the spatial dimensions
        
        coeffs_list = []

        # Order 0: Value
        val = jnp.expand_dims(coords, axis=1) 
        coeffs_list.append(val)

        if pde_order >= 1:
            # Order 1: Gradients (e.g., dx_j/dx_i)
            grad_components = []
            for i in range(spatial_dim):
                deriv_i_output = jnp.zeros((batch_size, num_features))
                deriv_i_output = deriv_i_output.at[:, i].set(1.0)
                grad_components.append(jnp.expand_dims(deriv_i_output, axis=1)) 
            if grad_components:
                 coeffs_list.append(jnp.concatenate(grad_components, axis=1)) 
            else: 
                 coeffs_list.append(jnp.empty((batch_size, 0, num_features)))

        if pde_order >= 2:
            # Order 2: Second derivatives (d^2 x_k / (dx_i dx_j)) are all zero for input coords.
            # Current implementation stores `spatial_dim` components for 2nd order,
            # typically representing [d^2/dx_0^2, d^2/dx_1^2, ...].
            coeffs_list.append(jnp.zeros((batch_size, spatial_dim, num_features)))
            
            # Initialize higher-order coefficients (k > 2) if needed
            for order in range(3, pde_order + 1):
                # Placeholder: number of components for orders > 2 needs specific definition based on PDE.
                # Defaulting to 1 component of zeros.
                num_higher_order_components = 1 
                coeffs_list.append(jnp.zeros((batch_size, num_higher_order_components, num_features)))

        return cls(coeffs_list)

    def concatenate_components(self, per_sample_norm: bool = False) -> jnp.ndarray:
        """
        Concatenates all Taylor coefficient components for KFAC factor calculation.

        Each `self.taylor_coefficients[k]` (shape `(batch, num_k_comps, features)`)
        is reshaped and then concatenated along `axis=0`. The goal is to produce
        a 2D array of shape `(S_total * batch_size, features)`, where `S_total`
        is the sum of `num_k_comps` over all orders `k`.

        The reshaping `(num_k_comps, batch, features) -> (num_k_comps * batch, features)`
        ensures that all components of a certain type (e.g., all 0-th order value
        components from all batch samples) are grouped together, followed by all
        1st-order derivative components, etc. This matches the expectation of KFAC
        factor computation where activations/gradients are stacked.

        Args:
            per_sample_norm: (Currently unused) Intended for future use if normalization
                             per sample component is needed.

        Returns:
            A JAX array of shape `(S_total * batch_size, features)`.
        """
        reshaped_components = []
        for s_k_coeffs in self.taylor_coefficients:
            batch_size, num_k_comps, features = s_k_coeffs.shape
            if num_k_comps == 0:
                continue
            # (batch, num_k_comps, features) -> (num_k_comps, batch, features)
            s_k_transposed = jnp.transpose(s_k_coeffs, (1, 0, 2))
            # -> (num_k_comps * batch, features)
            reshaped_s_k = jnp.reshape(s_k_transposed, (num_k_comps * batch_size, features))
            reshaped_components.append(reshaped_s_k)
            
        if not reshaped_components: 
            if self.taylor_coefficients and self.taylor_coefficients[0].shape[2] > 0:
                 features = self.taylor_coefficients[0].shape[2]
                 batch_size = self.taylor_coefficients[0].shape[0] 
                 return jnp.empty((0, features)) 
            else: 
                 raise ValueError("Cannot concatenate components: no data or feature dimension is zero.")

        concatenated = jnp.concatenate(reshaped_components, axis=0) 
        return concatenated

    @property
    def num_s_components(self) -> int:
        """
        Calculates the total number of derivative components (S_total).

        This is the sum of `num_components_k` (the second dimension of each tensor
        in `taylor_coefficients`) across all derivative orders `k`.
        For example, if pde_order=2 and spatial_dim=d:
        - Order 0: 1 component (value)
        - Order 1: d components (gradients d/dx_i)
        - Order 2: d components (pure second derivatives d^2/dx_i^2)
        Total S = 1 + d + d.

        Returns:
            The total number of S components.
        """
        s_total = 0
        for s_k_coeffs in self.taylor_coefficients:
            # s_k_coeffs has shape (batch, num_k_comps, features)
            s_total += s_k_coeffs.shape[1] # Add num_k_comps
        return s_total

def _propagate_linear_augmented(aug_state: AugmentedState, layer: eqx.nn.Linear) -> AugmentedState:
    """
    Propagates an `AugmentedState` through an `eqx.nn.Linear` layer.

    The propagation rules are applied to each Taylor coefficient tensor:
    - For the 0-th order coefficient (value $S_0$): $Z_0 = W S_0 + b$.
      The input $S_0$ of shape `(batch, 1, features_in)` is squeezed,
      transformed, and then expanded back.
    - For higher-order coefficients ($S_k, k \ge 1$): $Z_k = W S_k$.
      Each component of $S_k$ (shape `(batch, features_in)`) is multiplied
      by the weight matrix $W$. This is done efficiently by reshaping.

    Args:
        aug_state: The input `AugmentedState` before the linear layer.
        layer: The `eqx.nn.Linear` layer.

    Returns:
        A new `AugmentedState` representing the state after the linear transformation.
    """
    new_taylor_coefficients = []
    
    # Order 0: Z_0 = W S_0 + b
    s0_in = aug_state.taylor_coefficients[0] 
    s0_in_squeezed = jnp.squeeze(s0_in, axis=1) 
    s0_out_squeezed = s0_in_squeezed @ layer.weight.T + layer.bias 
    s0_out = jnp.expand_dims(s0_out_squeezed, axis=1)
    new_taylor_coefficients.append(s0_out)

    # Higher orders (k >= 1): Z_k = W S_k
    for k in range(1, len(aug_state.taylor_coefficients)):
        sk_in = aug_state.taylor_coefficients[k] 
        if sk_in.shape[1] == 0: # Handle cases where an order might have 0 components
            features_out = s0_out.shape[2]
            batch_size = sk_in.shape[0]
            new_taylor_coefficients.append(jnp.empty((batch_size, 0, features_out)))
            continue

        # Reshape for efficient matrix multiplication:
        # (batch, num_k_comps, features_in) -> (batch * num_k_comps, features_in)
        batch_size, num_k_comps, features_in = sk_in.shape
        sk_in_reshaped = jnp.reshape(sk_in, (batch_size * num_k_comps, features_in))
        # (batch * num_k_comps, features_in) @ (features_in, features_out) -> (batch * num_k_comps, features_out)
        sk_out_reshaped = sk_in_reshaped @ layer.weight.T
        features_out = layer.weight.shape[0] 
        # (batch * num_k_comps, features_out) -> (batch, num_k_comps, features_out)
        sk_out = jnp.reshape(sk_out_reshaped, (batch_size, num_k_comps, features_out))
        new_taylor_coefficients.append(sk_out)
        
    return AugmentedState(new_taylor_coefficients)

def _propagate_activation_augmented(
    aug_state_pre_activation: AugmentedState, 
    activation_fn: Callable, # Expected to be vmapped for (batch, features) inputs
    vmap_activation_fn_grad: Callable, # Expected to be vmapped for (batch, features) inputs
    vmap_activation_fn_grad_grad: Callable # Expected to be vmapped for (batch, features) inputs
) -> AugmentedState:
    """
    Propagates an `AugmentedState` through an element-wise activation function.

    This function applies Taylor series propagation rules (Faà di Bruno's formula
    for composition) for the first three orders (0th, 1st, 2nd).
    Let $S = (S_0, S_1, S_2, \dots)$ be the Taylor coefficients of the input to the
    activation function $\sigma(\cdot)$. The output Taylor coefficients $Z = (Z_0, Z_1, Z_2, \dots)$
    are computed as:
    - $Z_0 = \sigma(S_0)$
    - $Z_1 = \sigma'(S_0) S_1$
    - $Z_2 = \sigma'(S_0) S_2 + \frac{1}{2} \sigma''(S_0) S_1^2$
    
    The products are element-wise for the feature dimension and broadcasted over
    the derivative components dimension if necessary. For the $Z_2$ rule, it's
    assumed that $S_1$ and $S_2$ have a compatible number of components (e.g.,
    both have `spatial_dim` components if $S_1$ represents directional derivatives
    $d/dx_i$ and $S_2$ represents pure second derivatives $d^2/dx_i^2$).

    Taylor coefficients of order higher than 2 are currently zeroed out, as their
    propagation rules are not implemented.

    Args:
        aug_state_pre_activation: The input `AugmentedState` (pre-activation).
        activation_fn: The element-wise activation function, vmapped for batched features.
        vmap_activation_fn_grad: The vmapped first derivative of `activation_fn`.
        vmap_activation_fn_grad_grad: The vmapped second derivative of `activation_fn`.

    Returns:
        A new `AugmentedState` representing the state after the activation.
    """
    s_coeffs = aug_state_pre_activation.taylor_coefficients
    new_taylor_coefficients = []

    # S_0 is (batch, 1, features). Squeeze for element-wise application of sigma.
    s0_in_squeezed = jnp.squeeze(s_coeffs[0], axis=1) # Shape: (batch, features)

    # Z_0 = sigma(S_0)
    z0_out_squeezed = activation_fn(s0_in_squeezed) # Shape: (batch, features)
    z0_out = jnp.expand_dims(z0_out_squeezed, axis=1) # Shape: (batch, 1, features)
    new_taylor_coefficients.append(z0_out)

    pde_order = len(s_coeffs) - 1 # Max order available in input state
    
    if pde_order >= 1:
        s1_in = s_coeffs[1] # Shape: (batch, num_s1_comps, features)
        
        # sigma'(S_0)
        sigma_prime_s0 = vmap_activation_fn_grad(s0_in_squeezed) # Shape: (batch, features)
        # Expand for broadcasting: (batch, 1, features)
        sigma_prime_s0_expanded = jnp.expand_dims(sigma_prime_s0, axis=1)
        
        # Z_1 = sigma'(S_0) * S_1 (element-wise product, broadcasts over num_s1_comps)
        z1_out = sigma_prime_s0_expanded * s1_in 
        new_taylor_coefficients.append(z1_out)

    if pde_order >= 2:
        s1_in = s_coeffs[1] # Shape: (batch, num_s1_comps, features)
        s2_in = s_coeffs[2] # Shape: (batch, num_s2_comps, features)

        # sigma'(S_0) and sigma''(S_0)
        sigma_prime_s0 = vmap_activation_fn_grad(s0_in_squeezed)     # Shape: (batch, features)
        sigma_prime_prime_s0 = vmap_activation_fn_grad_grad(s0_in_squeezed) # Shape: (batch, features)

        # Expand for broadcasting
        sigma_prime_s0_expanded = jnp.expand_dims(sigma_prime_s0, axis=1)     # Shape: (batch, 1, features)
        sigma_prime_prime_s0_expanded = jnp.expand_dims(sigma_prime_prime_s0, axis=1) # Shape: (batch, 1, features)

        # Term 1: sigma'(S_0) * S_2
        term1_z2 = sigma_prime_s0_expanded * s2_in 
        
        # Term 2: (1/2) * sigma''(S_0) * S_1^2
        s1_squared = jnp.square(s1_in) # Element-wise square, shape: (batch, num_s1_comps, features)
        
        # This specific Z2 rule requires num_s1_comps == num_s2_comps for element-wise operations.
        if s1_in.shape[1] != s2_in.shape[1]:
            raise ValueError(
                f"For Z2 rule s_1^2 term, num_components of s1 ({s1_in.shape[1]}) "
                f"must match num_components of s2 ({s2_in.shape[1]}). "
                "This implies s2 components should align with s1 components (e.g., pure derivatives)."
            )

        term2_z2_direct_formula = 0.5 * sigma_prime_prime_s0_expanded * s1_squared
        z2_out = term1_z2 + term2_z2_direct_formula 
        new_taylor_coefficients.append(z2_out)

    # For orders higher than 2, zero them out as propagation rules are not yet implemented.
    if pde_order > 2:
        num_output_features = z0_out.shape[2] # Feature dimension from Z0 output
        for k_higher in range(3, pde_order + 1):
            sk_higher_in = s_coeffs[k_higher]
            zeros_higher_k = jnp.zeros((sk_higher_in.shape[0], sk_higher_in.shape[1], num_output_features))
            new_taylor_coefficients.append(zeros_higher_k)

    return AugmentedState(new_taylor_coefficients)


class LayerFactors(NamedTuple):
    """
    Stores KFAC factors and previous updates for a single linear layer.

    Attributes:
        A_omega: Kronecker factor for input activations (interior points).
        B_omega: Kronecker factor for output gradients (interior points).
        A_boundary: Kronecker factor for input activations (boundary points).
        B_boundary: Kronecker factor for output gradients (boundary points).
        prev_update_w: The previous update direction $\delta_{t-1}$ for weights.
                       Used for momentum in both KFAC* and standard KFAC.
        prev_update_b: The previous update direction $\delta_{t-1}$ for biases.
    """
    A_omega: jnp.ndarray
    B_omega: jnp.ndarray
    A_boundary: jnp.ndarray
    B_boundary: jnp.ndarray
    prev_update_w: jnp.ndarray 
    prev_update_b: jnp.ndarray 


class PINNKFACState(NamedTuple):
    """Internal state of the PINNKFAC optimiser."""
    step: int
    layers: Tuple[LayerFactors, ...]


class PINNKFAC(eqx.Module):
    """
    KFAC optimiser for Physics-Informed Neural Networks.

    This optimiser uses Kronecker-Factored Approximate Curvature (KFAC) to
    precondition gradients. It supports:
    - Separate factors for interior (PDE residual) and boundary condition losses.
    - A generalized `AugmentedState` for propagating Taylor coefficients, enabling
      computation of various derivatives needed for PDE residuals.
    - Two update variants:
        1. "kfac_star": Uses the KFAC* algorithm (Martens & Grosse, 2015) to
           solve a 2x2 quadratic subproblem for step sizes.
        2. "kfac_standard": Applies a KFAC-preconditioned gradient with optional
           momentum and line search.

    Hyperparameters:
        lr: Learning rate, primarily used by the "kfac_standard" variant when
            `use_line_search` is False. KFAC* determines its own step sizes.
        damping_omega: Damping added to Kronecker factors from the interior loss
                       term before inversion (for preconditioning $\Delta_t$).
        damping_boundary: Damping added to Kronecker factors from the boundary loss
                          term before inversion (for preconditioning $\Delta_t$).
        damping_kfac_star_model: Additional damping applied to the Hessian of the
                                 KFAC* quadratic subproblem ($G_{undamped} + \lambda_{quad}I$).
                                 Only used if `update_variant` is "kfac_star".
        decay: Decay rate for the Exponential Moving Average (EMA) of KFAC factors.
        update_variant: Specifies the update rule. Options:
                        - "kfac_star" (default)
                        - "kfac_standard"
        momentum_coeff: Coefficient for momentum term in the "kfac_standard" variant.
                        $\hat{\delta}_t = \text{momentum_coeff} \cdot \delta_{t-1} + \Delta_t$.
        use_line_search: If True and `update_variant` is "kfac_standard", a line
                         search is performed to find the optimal step size.
        line_search_grid_coeffs: A grid of coefficients used for the line search if
                                 `use_line_search` is True.
    """
    lr: float = 1e-2 
    damping_omega: float = 1e-3
    damping_boundary: float = 1e-3
    damping_kfac_star_model: float = 1e-3 
    decay: float = 0.95 

    update_variant: str = eqx.static_field(default="kfac_star") 
    momentum_coeff: float = 0.9 
    use_line_search: bool = eqx.static_field(default=True) 
    line_search_grid_coeffs: jnp.ndarray = eqx.static_field(default_factory=lambda: 2.0**jnp.arange(-10, 1, dtype=jnp.float32))

    def init(self, model: eqx.Module) -> PINNKFACState:
        """Initializes the KFAC optimiser state."""
        factors: List[LayerFactors] = []
        for layer in _linear_layers(model):
            m, n = layer.out_features, layer.in_features
            factors.append(
                LayerFactors(
                    A_omega=jnp.eye(n), B_omega=jnp.eye(m),
                    A_boundary=jnp.eye(n), B_boundary=jnp.eye(m),
                    prev_update_w=jnp.zeros_like(layer.weight), 
                    prev_update_b=jnp.zeros_like(layer.bias),   
                )
            )
        return PINNKFACState(step=0, layers=tuple(factors))

    def _compute_kfac_star_update(
        self,
        params: Any, # Current model parameters (PyTree)
        grads: Any,  # True gradients of the total loss (PyTree)
        current_preconditioned_delta_tree: Any, # Preconditioned grad Delta_t (PyTree)
        previous_update_tree: Any, # Previous update delta_{t-1} (PyTree)
        model_structure: eqx.Module, # Full model structure (used by compute_gramian_vector_product)
        current_kfac_factors_state: PINNKFACState, # Contains updated KFAC factors and current step
        lin_indices: List[int], # Indices of linear layers in the model
        _build_tree_from_parts_fn: Callable # Helper to build PyTrees (unused if GvP returns PyTree directly)
    ) -> Tuple[Any, Tuple[LayerFactors, ...]]: 
        """
        Computes the model update using the KFAC* algorithm.

        This involves solving a 2x2 quadratic subproblem to find optimal step sizes
        (alpha_star, mu_star) for the update:
        final_update = alpha_star * Delta_t + mu_star * delta_{t-1}

        Args:
            params: Current model parameters.
            grads: True gradients of the total loss.
            current_preconditioned_delta_tree: $\Delta_t = (G_{damped})^{-1} g_t$.
            previous_update_tree: $\delta_{t-1}$, the update applied in the previous step.
            model_structure: The Equinox model, used for structure in Gv products.
            current_kfac_factors_state: KFAC state with updated factors and step count.
            lin_indices: List of indices for linear layers.
            _build_tree_from_parts_fn: (Currently unused as compute_gramian_vector_product returns PyTrees)

        Returns:
            A tuple (final_params_pytree, final_kfac_state_layers_tuple) containing
            the updated model parameters and the KFAC factors for the next state
            (with prev_update_w/b updated to `final_update_tree`).
        """
        
        # Gramian-vector products using G_undamped (EMA factors from current_kfac_factors_state.layers)
        g_Delta_tree = compute_gramian_vector_product(
            current_preconditioned_delta_tree, model_structure, current_kfac_factors_state.layers
        ) 
        g_delta_tree = compute_gramian_vector_product(
            previous_update_tree, model_structure, current_kfac_factors_state.layers
        ) 

        # KFAC* quadratic model coefficients
        Delta_g_Delta = tree_dot(current_preconditioned_delta_tree, g_Delta_tree)
        delta_g_delta = tree_dot(previous_update_tree, g_delta_tree)
        Delta_g_delta = tree_dot(current_preconditioned_delta_tree, g_delta_tree)

        Delta_norm_sq = tree_dot(current_preconditioned_delta_tree, current_preconditioned_delta_tree) 
        delta_norm_sq = tree_dot(previous_update_tree, previous_update_tree)                         
        Delta_delta_dot = tree_dot(current_preconditioned_delta_tree, previous_update_tree)          
        
        # Damping for the KFAC* quadratic model's Hessian: G_quad = G_undamped + lambda_quad*I
        lambda_quad_damping = self.damping_kfac_star_model 
        
        quad_coeff_alpha    = Delta_g_Delta + lambda_quad_damping * Delta_norm_sq
        quad_coeff_mu       = delta_g_delta + lambda_quad_damping * delta_norm_sq
        quad_coeff_alpha_mu = Delta_g_delta + lambda_quad_damping * Delta_delta_dot

        # Linear coefficients for KFAC* system: -Delta_t^T g_t and -delta_{t-1}^T g_t
        Delta_grad_dot = -tree_dot(current_preconditioned_delta_tree, grads) 
        delta_grad_dot = -tree_dot(previous_update_tree, grads)             
        linear_coeff_alpha = Delta_grad_dot 
        linear_coeff_mu    = delta_grad_dot 

        # Solve the 2x2 system for (alpha_star, mu_star)
        if current_kfac_factors_state.step == 1: # First optimization step
            mu_star = 0.0
            if jnp.abs(quad_coeff_alpha) < 1e-8: 
                alpha_star = 0.0 
            else:
                alpha_star = linear_coeff_alpha / quad_coeff_alpha
        else:
            M = jnp.array([[quad_coeff_alpha, quad_coeff_alpha_mu],
                           [quad_coeff_alpha_mu, quad_coeff_mu]])
            b_vec = jnp.array([linear_coeff_alpha, linear_coeff_mu])
            solver_damping = 1e-6 # Small damping for numerical stability of the 2x2 solve
            try:
                solution = jnp.linalg.solve(M + solver_damping * jnp.eye(2), b_vec)
                alpha_star, mu_star = solution[0], solution[1]
            except jnp.linalg.LinAlgError:
                print("Warning: KFAC* 2x2 system solve failed. Using simplified update.")
                if jnp.abs(quad_coeff_alpha) < 1e-8: alpha_star = 0.0
                else: alpha_star = linear_coeff_alpha / quad_coeff_alpha
                mu_star = 0.0 

        # Compute the final update direction for KFAC*
        kfac_star_update_tree = jax.tree_map(
            lambda delta, prev_delta: alpha_star * delta + mu_star * prev_delta,
            current_preconditioned_delta_tree,
            previous_update_tree
        )
        # Apply update to parameters
        final_params_pytree = jax.tree_map(lambda p, u: p - u, params, kfac_star_update_tree)

        # Store kfac_star_update_tree (which is delta_t for this step) as prev_update_w/b
        new_kfac_state_layers_list = []
        update_parts_to_store = [] 
        for kfac_idx, model_layer_idx in enumerate(lin_indices):
            layer_update_w = kfac_star_update_tree.layers[model_layer_idx].weight
            layer_update_b = kfac_star_update_tree.layers[model_layer_idx].bias
            update_parts_to_store.append({'weight': layer_update_w, 'bias': layer_update_b})

        for kfac_idx, lf_old in enumerate(current_kfac_factors_state.layers): 
            new_prev_w = update_parts_to_store[kfac_idx]['weight']
            new_prev_b = update_parts_to_store[kfac_idx]['bias']
            new_kfac_state_layers_list.append(
                LayerFactors(
                    A_omega=lf_old.A_omega, B_omega=lf_old.B_omega,
                    A_boundary=lf_old.A_boundary, B_boundary=lf_old.B_boundary,
                    prev_update_w=new_prev_w, prev_update_b=new_prev_b 
                )
            )
        return final_params_pytree, tuple(new_kfac_state_layers_list)

    def _compute_standard_kfac_update(
        self,
        params: Any, # Current model parameters (PyTree)
        current_preconditioned_delta_tree: Any, # Preconditioned grad Delta_t (PyTree)
        previous_update_tree: Any, # Previous update delta_{t-1} (PyTree)
        current_kfac_factors_state: PINNKFACState, # KFAC state with updated factors
        loss_fn_for_line_search: Callable, # Receives params PyTree, returns scalar loss
        lin_indices: List[int] # Indices of linear layers
    ) -> Tuple[Any, Tuple[LayerFactors, ...]]: # Returns final_params_pytree, final_kfac_state_layers_tuple
        """
        Computes the model update using standard KFAC with momentum and optional line search.

        The update sequence is:
        1. Momentum: $\hat{\delta}_t = \text{momentum_coeff} \cdot \delta_{t-1} + \Delta_t$.
        2. Scaling (line search or fixed LR): $\delta_t = \alpha_\star \cdot \hat{\delta}_t$ or $\delta_t = \text{lr} \cdot \hat{\delta}_t$.
        3. Parameter update: $\theta_{t+1} = \theta_t - \delta_t$.

        Args:
            params: Current model parameters.
            current_preconditioned_delta_tree: $\Delta_t = (G_{damped})^{-1} g_t$.
            previous_update_tree: $\delta_{t-1}$, the update applied in the previous step.
            current_kfac_factors_state: KFAC state with updated factors.
            loss_fn_for_line_search: Function to evaluate total loss, used by line search.
            lin_indices: List of indices for linear layers.

        Returns:
            A tuple (final_params_pytree, final_kfac_state_layers_tuple) containing
            the updated model parameters and the KFAC factors for the next state
            (with prev_update_w/b updated to $\delta_t$).
        """
        # Momentum step
        hat_delta_t_tree = jax.tree_map(
            lambda prev_delta, current_delta: self.momentum_coeff * prev_delta + current_delta,
            previous_update_tree,
            current_preconditioned_delta_tree 
        )

        actual_update_to_apply = None 
        if self.use_line_search:
            # Line search for alpha_star (effective learning rate)
            def compute_total_loss_at_perturbed_params(alpha_lr_val, base_params_tree, update_direction_tree_ls):
                perturbed_params_tree = jax.tree_map(
                    lambda p, d: p - alpha_lr_val * d, base_params_tree, update_direction_tree_ls
                )
                return loss_fn_for_line_search(perturbed_params_tree)

            vmapped_loss_eval = jax.vmap(
                compute_total_loss_at_perturbed_params, 
                in_axes=(0, None, None), 
            )
            all_losses = vmapped_loss_eval(
                self.line_search_grid_coeffs, params, hat_delta_t_tree
            )
            alpha_star_idx = jnp.argmin(all_losses)
            alpha_star = self.line_search_grid_coeffs[alpha_star_idx]
            actual_update_to_apply = jax.tree_map(lambda d: alpha_star * d, hat_delta_t_tree)
        else:
            # No line search, use fixed learning rate self.lr
            actual_update_to_apply = jax.tree_map(lambda d: self.lr * d, hat_delta_t_tree)

        # Parameter update
        final_params_pytree = jax.tree_map(
            lambda p, upd: p - upd, params, actual_update_to_apply
        )

        # Store actual_update_to_apply (this is delta_t for this step) as new prev_update_w/b
        new_kfac_state_layers_list = []
        update_parts_to_store = [] 
        for kfac_idx, model_layer_idx in enumerate(lin_indices):
            layer_update_w = actual_update_to_apply.layers[model_layer_idx].weight
            layer_update_b = actual_update_to_apply.layers[model_layer_idx].bias
            update_parts_to_store.append({'weight': layer_update_w, 'bias': layer_update_b})

        for kfac_idx, lf_old in enumerate(current_kfac_factors_state.layers): 
            new_prev_w = update_parts_to_store[kfac_idx]['weight']
            new_prev_b = update_parts_to_store[kfac_idx]['bias']
            new_kfac_state_layers_list.append(
                LayerFactors(
                    A_omega=lf_old.A_omega, B_omega=lf_old.B_omega,
                    A_boundary=lf_old.A_boundary, B_boundary=lf_old.B_boundary,
                    prev_update_w=new_prev_w, prev_update_b=new_prev_b 
                )
            )
        return final_params_pytree, tuple(new_kfac_state_layers_list)


    def step(
        self,
        model: eqx.Module,
        rhs_fn: Callable[[jnp.ndarray], jnp.ndarray],
        bc_fn: Callable[[jnp.ndarray], jnp.ndarray],
        interior: jnp.ndarray,
        boundary: jnp.ndarray,
        state: PINNKFACState,
    ) -> Tuple[eqx.Module, PINNKFACState]:
        """
        Performs one KFAC optimiser step.

        The process involves:
        1. Computing standard gradients of the total loss.
        2. Calculating KFAC Kronecker factors (A and B) for interior and boundary losses.
           These factors are updated using an exponential moving average.
        3. Computing the preconditioned gradient $\Delta_t = (G_{damped})^{-1} g_t$,
           where $G_{damped}$ is the KFAC approximation of the Fisher/Gramian,
           damped for stability.
        4. Retrieving the previous update direction $\delta_{t-1}$.
        5. Dispatching to the chosen update variant ("kfac_star" or "kfac_standard")
           to compute the final parameter update and the new $\delta_t$ to be stored.
        6. Updating model parameters and KFAC state.
        """
        params, static = eqx.partition(model, eqx.is_array)

        # Define loss functions for main gradient calculation and line search
        def interior_loss_for_grads(p_tree): 
            m = eqx.combine(static, p_tree)
            pde_order_for_aug = 2 # Assuming PDE requires up to 2nd order derivatives for Laplacian
            spatial_dim_for_aug = interior.shape[1]
            initial_aug_state_for_loss = AugmentedState.from_coords(interior, pde_order_for_aug, spatial_dim_for_aug)
            _, _, final_aug_output_for_loss = _augmented_forward_cache(m, initial_aug_state_for_loss)
            
            if pde_order_for_aug < 2 or len(final_aug_output_for_loss.taylor_coefficients) <= 2:
                raise ValueError("PDE order must be at least 2 to compute Laplacian from AugmentedState.")
            
            s2_coeffs = final_aug_output_for_loss.taylor_coefficients[2] # (batch, num_s2_comps, features)
            # Assuming s2_coeffs store [d^2u/dx_i^2] components, sum over components for Laplacian
            lap_from_aug = jnp.sum(s2_coeffs, axis=1) # (batch, features)
            
            if lap_from_aug.ndim == 2 and lap_from_aug.shape[-1] == 1: # If features=1, squeeze to (batch,)
                lap_from_aug = lap_from_aug.squeeze(-1) 
            
            res_standard = lap_from_aug - rhs_fn(interior) # rhs_fn output should match lap_from_aug
            return 0.5 * jnp.mean(res_standard**2)

        def boundary_loss_for_grads(p_tree):
            m = eqx.combine(static, p_tree)
            preds = jax.vmap(m)(boundary) # Standard model prediction for boundary
            res = preds.squeeze() - bc_fn(boundary) # bc_fn output should match preds.squeeze()
            return 0.5 * jnp.mean(res**2)

        loss_fn_for_grads = lambda p_tree: interior_loss_for_grads(p_tree) + boundary_loss_for_grads(p_tree)
        
        # 1. Compute true gradients of the total loss
        loss_val, grads = jax.value_and_grad(loss_fn_for_grads)(params)
        m_eval = eqx.combine(static, params) # Model with current parameters for factor calculation

        # 2. Calculate KFAC factors for boundary term (standard KFAC)
        y_b, acts_b_std, pre_b_std, phi_b_std = _standard_forward_cache(m_eval, boundary)
        res_b = y_b.squeeze() - bc_fn(boundary) 
        grad_out_b = res_b / boundary.shape[0]  
        deltas_b_std = _standard_backward_pass(m_eval, pre_b_std, phi_b_std, grad_out_b)
        
        # 3. Calculate KFAC factors for interior term (augmented KFAC)
        pde_order_for_factors = 2 # Consistent with loss calculation
        spatial_dim_for_factors = interior.shape[1]
        aug_factors_i = _augmented_factor_terms(
            m_eval, params, interior, rhs_fn, pde_order_for_factors, spatial_dim_for_factors
        )

        # 4. Update KFAC factor EMAs (A_omega, B_omega, A_boundary, B_boundary)
        new_layers_factors_list = []
        for i, lf in enumerate(state.layers): # lf contains delta_{t-1} from previous step
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
            
            # Carry over prev_update_w/b (delta_{t-1})
            new_layers_factors_list.append(
                LayerFactors(A_om, B_om, A_bd, B_bd, lf.prev_update_w, lf.prev_update_b)
            )
        
        # This state has updated A, B factors and carries delta_{t-1}. Step is incremented.
        current_kfac_factors_state = PINNKFACState(step=state.step + 1, layers=tuple(new_layers_factors_list)) 

        # 5. Compute Preconditioned Gradient (Delta_t) and retrieve Previous Update (delta_{t-1})
        lin_indices = [k for k, l_obj in enumerate(model.layers) if isinstance(l_obj, eqx.nn.Linear)]
        current_preconditioned_delta_parts = [] 
        previous_update_parts = []            
        
        for kfac_idx, model_layer_idx in enumerate(lin_indices):
            lf = current_kfac_factors_state.layers[kfac_idx] # lf has updated A,B and old delta_{t-1}
            
            # Add damping for inversion: EMA(Factor) + damping_type * I
            Aw_damped = lf.A_omega + self.damping_omega * jnp.eye(lf.A_omega.shape[0])
            Gw_damped = lf.B_omega + self.damping_omega * jnp.eye(lf.B_omega.shape[0])
            Aw_b_damped = lf.A_boundary + self.damping_boundary * jnp.eye(lf.A_boundary.shape[0])
            Gw_b_damped = lf.B_boundary + self.damping_boundary * jnp.eye(lf.B_boundary.shape[0])

            eig_A, UA = jnp.linalg.eigh(Aw_damped)
            eig_G, UG = jnp.linalg.eigh(Gw_damped)
            eig_Ab, UAb = jnp.linalg.eigh(Aw_b_damped)
            eig_Gb, UGb = jnp.linalg.eigh(Gw_b_damped)
            
            current_layer_grads = grads.layers[model_layer_idx] 
            gw = current_layer_grads.weight
            gb = current_layer_grads.bias
            
            # Compute (G_damped)^-1 g_t for weights
            gw_kfac_layer = UA.T @ gw @ UG
            precond_denominator_w = (eig_A[:, None] * eig_G[None, :]) + \
                                  (eig_Ab[:, None] * eig_Gb[None, :]) 
            gw_kfac_layer = gw_kfac_layer / (precond_denominator_w + 1e-12) # Add epsilon for stability
            gw_kfac_layer = UA @ gw_kfac_layer @ UG.T

            # Compute (G_damped)^-1 g_t for biases
            gb_kfac_layer = UG.T @ gb 
            precond_denominator_b = eig_G + eig_Gb 
            gb_kfac_layer = gb_kfac_layer / (precond_denominator_b + 1e-12) # Add epsilon
            gb_kfac_layer = UG @ gb_kfac_layer 
            
            current_preconditioned_delta_parts.append({'weight': gw_kfac_layer, 'bias': gb_kfac_layer})
            previous_update_parts.append({'weight': lf.prev_update_w, 'bias': lf.prev_update_b})

        # Helper to build PyTrees (Delta_t, delta_{t-1}) from lists of layer parts
        def _build_tree_from_parts(parts_list_local, template_params_local, linear_indices_local):
            zero_params = jax.tree_map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x, template_params_local)
            filled_tree = zero_params
            if len(parts_list_local) != len(linear_indices_local):
                raise ValueError(f"Parts list length {len(parts_list_local)} != linear indices length {len(linear_indices_local)}")
            for k_idx, p_dict in enumerate(parts_list_local):
                model_l_idx = linear_indices_local[k_idx]
                if not hasattr(template_params_local.layers[model_l_idx], 'weight') or \
                   not hasattr(template_params_local.layers[model_l_idx], 'bias'):
                    raise ValueError(f"Target layer {model_l_idx} in template is not a Linear layer.")
                weight_path = lambda tree: tree.layers[model_l_idx].weight
                bias_path = lambda tree: tree.layers[model_l_idx].bias
                filled_tree = eqx.tree_at(weight_path, filled_tree, p_dict['weight'])
                filled_tree = eqx.tree_at(bias_path, filled_tree, p_dict['bias'])
            return filled_tree

        current_preconditioned_delta_tree = _build_tree_from_parts(current_preconditioned_delta_parts, params, lin_indices) # Delta_t
        previous_update_tree = _build_tree_from_parts(previous_update_parts, params, lin_indices) # delta_{t-1}
        
        # 6. Dispatch to selected update variant
        final_params_pytree = None
        final_kfac_state_layers_tuple = None # Will contain updated prev_update_w/b

        if self.update_variant == "kfac_star":
            final_params_pytree, final_kfac_state_layers_tuple = self._compute_kfac_star_update(
                params, grads, current_preconditioned_delta_tree, previous_update_tree,
                model, current_kfac_factors_state, lin_indices, _build_tree_from_parts
            )
        
        elif self.update_variant == "kfac_standard":
            final_params_pytree, final_kfac_state_layers_tuple = self._compute_standard_kfac_update(
                params, current_preconditioned_delta_tree, previous_update_tree,
                current_kfac_factors_state, loss_fn_for_grads, lin_indices
            )
        
        else:
            raise ValueError(f"Unknown update_variant: {self.update_variant}")

        # 7. Finalize state and model
        # current_kfac_factors_state.step already has the incremented step count.
        # final_kfac_state_layers_tuple has updated prev_update_w/b (delta_t for this step).
        final_state = PINNKFACState(step=current_kfac_factors_state.step, layers=final_kfac_state_layers_tuple)
        new_model = eqx.combine(static, final_params_pytree)
        return new_model, final_state

def _get_activation_derivatives(
    activation_fn: Callable, 
    layer_description: str
) -> Tuple[Callable, Callable, Callable]:
    try:
        jax.eval_shape(activation_fn, jnp.array(0.0))
        grad_fn = jax.grad(activation_fn)
        jax.eval_shape(grad_fn, jnp.array(0.0))
        grad_grad_fn = jax.grad(grad_fn)
        jax.eval_shape(grad_grad_fn, jnp.array(0.0))
        return activation_fn, grad_fn, grad_grad_fn
    except Exception as e:
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

def _standard_forward_cache(model: eqx.Module, x: jnp.ndarray):
    acts = []  
    pre = []   
    phi = []   
    h = x 
    layers = list(model.layers)
    current_layer_index = 0
    processed_linear_layers = 0 
    while current_layer_index < len(layers):
        layer = layers[current_layer_index]
        if isinstance(layer, eqx.nn.Linear):
            acts.append(h) 
            s = layer(h)   
            pre.append(s)
            processed_linear_layers +=1
            if current_layer_index + 1 < len(layers) and isinstance(layers[current_layer_index+1], eqx.nn.Lambda):
                activation_layer_obj = layers[current_layer_index+1]
                layer_description = f"activation layer {type(activation_layer_obj.fn).__name__} after Linear layer {processed_linear_layers-1}"
                _, grad_fn, _ = _get_activation_derivatives(activation_layer_obj.fn, layer_description)
                vmapped_grad_fn = jax.vmap(grad_fn)
                phi.append(vmapped_grad_fn(s))
                h = activation_layer_obj(s) 
                current_layer_index += 1 
            else: 
                phi.append(jnp.ones_like(s)) 
                h = s 
        else: 
            h = layer(h) 
        current_layer_index += 1
    return h, acts, pre, phi

def _standard_backward_pass(model: eqx.Module, pre_activations: List[jnp.ndarray], act_derivatives: List[jnp.ndarray], grad_output: jnp.ndarray) -> List[jnp.ndarray]:
    deltas = []
    if grad_output.ndim == 1:
        g = grad_output[:, None] 
    else:
        g = grad_output
    linear_layer_indices = [i for i, l_obj in enumerate(model.layers) if isinstance(l_obj, eqx.nn.Linear)]
    for idx_in_model_lists in reversed(range(len(linear_layer_indices)))):
        layer_object = _linear_layers(model)[idx_in_model_lists] 
        phi_l = act_derivatives[idx_in_model_lists] 
        g_s_l = g * phi_l 
        deltas.insert(0, g_s_l) 
        g = g_s_l @ layer_object.weight 
    return deltas

def _augmented_forward_cache(model: eqx.Module, initial_aug_state: AugmentedState) -> Tuple[List[AugmentedState], List[AugmentedState], AugmentedState]:
    """
    Performs a forward pass propagating the `AugmentedState`.

    Caches and returns:
    - `aug_input_acts`: List of `AugmentedState` objects, representing the inputs
      ($Z_{in}^{(l)}$) to each `eqx.nn.Linear` layer.
    - `aug_pre_acts`: List of `AugmentedState` objects, representing the outputs
      of each `eqx.nn.Linear` layer before activation ($S_{out}^{(l)}$).
    - `final_aug_output`: The `AugmentedState` after the final layer of the model.
    """
    aug_input_acts: List[AugmentedState] = []
    aug_pre_acts: List[AugmentedState] = []
    current_aug_state = initial_aug_state
    layers = list(model.layers)
    current_layer_index = 0
    processed_linear_layers = 0 
    while current_layer_index < len(layers):
        layer = layers[current_layer_index]
        if isinstance(layer, eqx.nn.Linear):
            aug_input_acts.append(current_aug_state)
            current_aug_state = _propagate_linear_augmented(current_aug_state, layer)
            aug_pre_acts.append(current_aug_state) 
            processed_linear_layers += 1
            if current_layer_index + 1 < len(layers) and isinstance(layers[current_layer_index+1], eqx.nn.Lambda):
                activation_layer_obj = layers[current_layer_index+1]
                act_fn_raw = activation_layer_obj.fn
                desc = f"activation layer {type(act_fn_raw).__name__} after Linear layer {processed_linear_layers-1}"
                act_fn, grad_fn, grad_grad_fn = _get_activation_derivatives(act_fn_raw, desc)
                vmap_grad_local = jax.vmap(jax.vmap(grad_fn)) # Ensure vmap over batch and features
                vmap_grad_grad_local = jax.vmap(jax.vmap(grad_grad_fn))
                current_aug_state = _propagate_activation_augmented(
                    current_aug_state, jax.vmap(jax.vmap(act_fn)), vmap_grad_local,vmap_grad_grad_local
                )
                current_layer_index += 1 
        elif isinstance(layer, eqx.nn.Lambda): 
            act_fn_raw = layer.fn
            desc = f"standalone activation layer {type(act_fn_raw).__name__} at model index {current_layer_index}"
            act_fn, grad_fn, grad_grad_fn = _get_activation_derivatives(act_fn_raw, desc)
            vmap_grad_local = jax.vmap(jax.vmap(grad_fn))
            vmap_grad_grad_local = jax.vmap(jax.vmap(grad_grad_fn))
            current_aug_state = _propagate_activation_augmented(
                current_aug_state, jax.vmap(jax.vmap(act_fn)), vmap_grad_local, vmap_grad_grad_local
            )
        else: 
            # Generic layer handling: apply to 0-th order, zero out higher orders.
            val_coeff_squeezed = jnp.squeeze(current_aug_state.taylor_coefficients[0], axis=1)
            new_val_squeezed = layer(val_coeff_squeezed) 
            new_val_coeff = jnp.expand_dims(new_val_squeezed, axis=1) 
            
            new_coeffs_list_unknown_layer = [new_val_coeff]
            num_new_features = new_val_coeff.shape[2]

            for k_order in range(1, len(current_aug_state.taylor_coefficients)):
                s_k_old = current_aug_state.taylor_coefficients[k_order]
                batch_size_k, num_k_comps, _ = s_k_old.shape
                new_coeffs_list_unknown_layer.append(jnp.zeros((batch_size_k, num_k_comps, num_new_features)))
            current_aug_state = AugmentedState(taylor_coefficients=new_coeffs_list_unknown_layer)
        current_layer_index += 1 
    return aug_input_acts, aug_pre_acts, current_aug_state

def _augmented_factor_terms(
    model_eval: eqx.Module, 
    params: Any, # Unused, model_eval is already combined. Kept for API consistency if ever needed.
    interior_pts: jnp.ndarray, 
    rhs_fn: Callable[[jnp.ndarray], jnp.ndarray],
    pde_order: int, 
    spatial_dim: int 
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Calculates KFAC factor contributions ($A_\Omega, B_\Omega$) from interior points.

    $A_\Omega$ factors are derived from the augmented input activations ($Z_{in}^{(l)}$)
    to each linear layer.
    $B_\Omega$ factors are derived from the gradients of the interior loss $L_\Omega$
    with respect to the augmented pre-activations ($S_{out}^{(l)}$) of each linear layer.

    Args:
        model_eval: The Equinox model with current parameters.
        params: (Unused) Model parameters.
        interior_pts: Sampled interior coordinates for PDE loss.
        rhs_fn: Function defining the PDE right-hand side.
        pde_order: Maximum derivative order for `AugmentedState`.
        spatial_dim: Number of spatial dimensions.

    Returns:
        A list of tuples `(a_contrib, b_contrib)` for each linear layer,
        where `a_contrib` is for $A_\Omega$ and `b_contrib` is for $B_\Omega$.
    """
    initial_aug_state = AugmentedState.from_coords(interior_pts, pde_order, spatial_dim)
    # aug_input_acts_per_layer_values are Z_in_l
    # aug_pre_acts_per_layer_values are S_out_l
    aug_input_acts_per_layer_values, aug_pre_acts_per_layer_values, _ = \
        _augmented_forward_cache(model_eval, initial_aug_state)

    def compute_interior_loss_from_s_out_list(
        s_out_aug_list_for_grad: List[AugmentedState], # List of S_out_l (vars for grad)
        model_static_parts: eqx.Module, 
        interior_pts_for_loss_calc: jnp.ndarray, 
        rhs_fn_for_loss_calc: Callable, 
        pde_order_for_loss: int, 
        spatial_dim_for_loss: int # Unused, but part of the context
    ):
        """
        Computes $L_\Omega$ by reconstructing the forward pass from a list of $S_{out}^{(l)}$.
        This function is differentiated w.r.t. `s_out_aug_list_for_grad`.
        """
        current_propagated_state: AugmentedState = None 
        linear_layer_indices_in_model = [i for i, layer in enumerate(model_static_parts.layers) if isinstance(layer, eqx.nn.Linear)]
        
        if len(s_out_aug_list_for_grad) != len(linear_layer_indices_in_model):
            raise ValueError("Mismatch: len(s_out_aug_list_for_grad) != num linear layers in model.")

        activation_derivatives_map = {} 
        for k_model_idx, layer_obj_k in enumerate(model_static_parts.layers):
            if isinstance(layer_obj_k, eqx.nn.Lambda):
                act_fn_raw = layer_obj_k.fn
                desc = f"activation layer {type(act_fn_raw).__name__} at model index {k_model_idx} (in compute_interior_loss_from_s_out_list)"
                act_fn, grad_fn, grad_grad_fn = _get_activation_derivatives(act_fn_raw, desc)
                # Ensure derivatives are vmapped for (batch, features) inputs
                activation_derivatives_map[k_model_idx] = (
                    jax.vmap(jax.vmap(act_fn)), 
                    jax.vmap(jax.vmap(grad_fn)), 
                    jax.vmap(jax.vmap(grad_grad_fn))
                )
        
        for k_linear_idx in range(len(linear_layer_indices_in_model)):
            s_out_l = s_out_aug_list_for_grad[k_linear_idx] # Current S_out_l (variable for grad)
            model_idx_of_linear_k = linear_layer_indices_in_model[k_linear_idx]
            current_propagated_state = s_out_l # This is S_out_l
            
            activation_model_idx = model_idx_of_linear_k + 1
            if activation_model_idx < len(model_static_parts.layers) and \
               isinstance(model_static_parts.layers[activation_model_idx], eqx.nn.Lambda):
                act_derivatives = activation_derivatives_map.get(activation_model_idx)
                if act_derivatives: 
                    act_fn_local_vmap, vmap_grad_local, vmap_grad_grad_local = act_derivatives
                    # current_propagated_state becomes Z_out_l
                    current_propagated_state = _propagate_activation_augmented(
                        s_out_l, act_fn_local_vmap, vmap_grad_local, vmap_grad_grad_local
                    )
            # If no activation, current_propagated_state remains S_out_l.
            # This is the final state of the network if this was the last layer.
            
        if current_propagated_state is None: 
            raise ValueError("No linear layers found or processed in compute_interior_loss_from_s_out_list.")
        
        # Compute loss using the final propagated state (Z_out_L or S_out_L)
        if pde_order_for_loss < 2 or len(current_propagated_state.taylor_coefficients) <= 2:
            raise ValueError("PDE order must be at least 2 to compute Laplacian for loss.")
        s2_coeffs_final = current_propagated_state.taylor_coefficients[2] 
        lap_val_final = jnp.sum(s2_coeffs_final, axis=1) # Sum over num_s2_comps
        if lap_val_final.ndim == 2 and lap_val_final.shape[-1] == 1:
            lap_val_final = lap_val_final.squeeze(-1) 
        res_interior = lap_val_final - rhs_fn_for_loss_calc(interior_pts_for_loss_calc)
        loss_val = 0.5 * jnp.mean(jnp.square(res_interior)) 
        return loss_val
    
    _, model_static_parts = eqx.partition(model_eval, eqx.is_array, is_leaf=lambda x: x is None)
    curried_loss_fn_for_grad = lambda s_list: compute_interior_loss_from_s_out_list(
        s_list, model_static_parts, interior_pts, rhs_fn, pde_order, spatial_dim 
    )
    # grads_s_out_l_list contains dL/dS_out_l for each layer, as AugmentedState objects
    grads_s_out_l_list = jax.grad(curried_loss_fn_for_grad, argnums=0)(aug_pre_acts_per_layer_values)
    
    factor_contributions = []
    processed_linear_layers = _linear_layers(model_eval)
    if len(processed_linear_layers) != len(aug_input_acts_per_layer_values) or \
       len(processed_linear_layers) != len(grads_s_out_l_list):
        raise ValueError("Mismatch in the number of linear layers and cached/gradient states.")

    for idx, _ in enumerate(processed_linear_layers): 
        # a_contrib from Z_in_l (input to linear layer l)
        z_in_l_aug = aug_input_acts_per_layer_values[idx] 
        a_contrib = z_in_l_aug.concatenate_components() 
        
        # b_contrib from dL/dS_out_l (gradient w.r.t. output of linear layer l)
        grad_S_out_l_aug = grads_s_out_l_list[idx] 
        b_contrib = grad_S_out_l_aug.concatenate_components() 
        
        factor_contributions.append((a_contrib, b_contrib))
    return factor_contributions

def _standard_factor_terms(model_eval_std: eqx.Module, params_std: Any, pts_std: jnp.ndarray, fn_boundary_val_std: Callable):
    # This function is now only for boundary terms.
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
            continue # Skip non-array leaves (e.g. static parts of a model)
        if leaf1.shape != leaf2.shape:
            raise ValueError(f"Leaves must have the same shape, got {leaf1.shape} and {leaf2.shape}")
        dot_product += jnp.sum(leaf1 * leaf2)
    return dot_product

def compute_gramian_vector_product(
    vector_tree: Any, 
    model_for_structure: eqx.Module, 
    kfac_factors_layers: Tuple[LayerFactors, ...], 
) -> Any: 
    """
    Computes the product G @ vector_tree, where G is the KFAC approximation 
    (G = G_omega + G_boundary), using undamped KFAC factors.
    The returned PyTree has the same structure as `vector_tree`.
    """
    all_model_layers = list(model_for_structure.layers)
    lin_indices = [k for k, layer_obj in enumerate(all_model_layers) if isinstance(layer_obj, eqx.nn.Linear)]
    
    if len(lin_indices) != len(kfac_factors_layers):
        raise ValueError("Mismatch between number of linear layers in model and KFAC factors.")

    gv_products_list = [] 
    for kfac_idx_loop, lf_factors_loop in enumerate(kfac_factors_layers):
        model_layer_idx_loop = lin_indices[kfac_idx_loop] 
        try:
            current_layer_in_vector_tree_loop = vector_tree.layers[model_layer_idx_loop]
            vw_loop = current_layer_in_vector_tree_loop.weight
            vb_loop = current_layer_in_vector_tree_loop.bias
        except AttributeError:
             raise ValueError("vector_tree is expected to have a .layers attribute similar to an Equinox model for Gv products.")

        # G_omega V_w = B_omega V_w A_omega^T
        g_vw_omega_loop = lf_factors_loop.B_omega @ vw_loop @ lf_factors_loop.A_omega.T 
        # G_boundary V_w = B_boundary V_w A_boundary^T
        g_vw_boundary_loop = lf_factors_loop.B_boundary @ vw_loop @ lf_factors_loop.A_boundary.T
        final_g_vw_loop = g_vw_omega_loop + g_vw_boundary_loop
            
        # G_omega V_b = B_omega V_b
        g_vb_omega_loop = lf_factors_loop.B_omega @ vb_loop
        # G_boundary V_b = B_boundary V_b
        g_vb_boundary_loop = lf_factors_loop.B_boundary @ vb_loop
        final_g_vb_loop = g_vb_omega_loop + g_vb_boundary_loop
            
        gv_products_list.append({'weight': final_g_vw_loop, 'bias': final_g_vb_loop})

    # Build the output PyTree using the structure of vector_tree as a template
    output_tree = jax.tree_map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x, vector_tree)
    
    if not (hasattr(vector_tree, "layers") and isinstance(vector_tree.layers, (list, tuple))):
         # This warning or error might be too strict if vector_tree is not a full model PyTree.
         # However, the path access below relies on this structure.
         print("Warning: vector_tree in compute_gramian_vector_product might not match expected model structure.")

    for kfac_idx_fill, gv_parts_dict in enumerate(gv_products_list):
        model_layer_idx_fill = lin_indices[kfac_idx_fill]
        try:
            _ = getattr(vector_tree.layers[model_layer_idx_fill], 'weight')
            _ = getattr(vector_tree.layers[model_layer_idx_fill], 'bias')
        except (AttributeError, IndexError) as e:
            raise ValueError(
                f"Layer at index {model_layer_idx_fill} in vector_tree does not have "
                ".weight and .bias attributes or is out of bounds. Ensure vector_tree matches model structure for linear layers."
            ) from e

        weight_path = lambda tree: tree.layers[model_layer_idx_fill].weight
        bias_path = lambda tree: tree.layers[model_layer_idx_fill].bias
        
        output_tree = eqx.tree_at(weight_path, output_tree, gv_parts_dict['weight'])
        output_tree = eqx.tree_at(bias_path, output_tree, gv_parts_dict['bias'])
        
    return output_tree
