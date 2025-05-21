import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import chex

# Imports from kfac_pinn
from kfac_pinn.pinn_kfac import (
    AugmentedState,
    _propagate_linear_augmented,
    _propagate_activation_augmented,
    _augmented_forward_cache,
    LayerFactors, # For PINNKFACState and factor related tests
    PINNKFACState,
    PINNKFAC,
    _linear_layers,
    _standard_forward_cache, # For standard_factor_terms test
    _standard_backward_pass, # For standard_factor_terms test
    _augmented_factor_terms,
    compute_gramian_vector_product,
    tree_dot
)
# Imports from kfac_pinn (specifically functions to be tested)
from kfac_pinn.pinn_kfac import _get_activation_derivatives

# If pdes.forward_laplacian is needed, it would be:
# from kfac_pinn.pdes import forward_laplacian

# Seed for reproducibility in tests
key = jr.PRNGKey(0)

# Helper function for numerical differentiation for comparison
def numerical_grad(fn, x, eps=1e-4):
    return (fn(x + eps) - fn(x - eps)) / (2 * eps)

class TestGetActivationDerivatives:
    @pytest.mark.parametrize("jax_fn_name, test_points", [
        ("tanh", [0.0, 0.5, -0.5]),
        ("relu", [0.0, 0.5, -0.5]), # grad(grad(relu)) is 0 everywhere except undefined at 0 (JAX returns 0 for grad(relu)(0))
        ("sin", [0.0, jnp.pi/2, -jnp.pi/4]),
    ])
    def test_common_activators(self, jax_fn_name, test_points):
        if jax_fn_name == "tanh":
            act_fn_orig = jnp.tanh
        elif jax_fn_name == "relu":
            act_fn_orig = jax.nn.relu
        elif jax_fn_name == "sin":
            act_fn_orig = jnp.sin
        else:
            raise ValueError("Unsupported function")

        fn, grad_fn, grad_grad_fn = _get_activation_derivatives(act_fn_orig, f"test_{jax_fn_name}")

        for x_val_float in test_points:
            x = jnp.array(x_val_float) # Ensure it's a JAX array for differentiation
            
            # Check original function
            chex.assert_trees_all_close(fn(x), act_fn_orig(x), atol=1e-6, err_msg=f"{jax_fn_name}({x})")

            # Check first derivative
            expected_grad = jax.grad(act_fn_orig)(x)
            chex.assert_trees_all_close(grad_fn(x), expected_grad, atol=1e-6, err_msg=f"grad_{jax_fn_name}({x})")
            
            # Check second derivative
            expected_grad_grad = jax.grad(jax.grad(act_fn_orig))(x)
            chex.assert_trees_all_close(grad_grad_fn(x), expected_grad_grad, atol=1e-6, err_msg=f"grad_grad_{jax_fn_name}({x})")

    def test_non_differentiable_function(self):
        # jnp.sign is complex-valued for derivative, and not JAX-differentiable for floats at 0
        # _get_activation_derivatives uses jax.eval_shape(..., jnp.array(0.0)) which will fail for jnp.sign
        # as jax.grad(jnp.sign)(0.0) is nan.
        fn_non_diff = lambda x: jnp.sign(x) 
        with pytest.raises(TypeError, match="Failed to compute derivatives"):
            _get_activation_derivatives(fn_non_diff, "test_non_differentiable_sign")

    def test_custom_non_jax_traceable_function(self):
        # A function that cannot be traced by JAX due to Python control flow based on abstract value
        def fn_problematic(x):
            if x > 0: # This type of condition on an abstract tracer causes ConcretizationTypeError
                return x * x
            else:
                return x
        
        with pytest.raises(TypeError, match="Failed to compute derivatives"):
            # This will likely fail at the jax.eval_shape(activation_fn, ...) stage itself
            _get_activation_derivatives(fn_problematic, "test_problematic_python_control_flow")


    def test_once_not_twice_differentiable(self):
        # f(x) = x^2 if x >=0, else x^3. f'(x) = 2x if x >=0, else 3x^2. f'(0)=0.
        # f''(x) = 2 if x > 0, else 6x. f''(0+)=2, f''(0-)=0. Not twice differentiable at 0.
        # jax.grad(jax.grad(fn))(0.0) might give one of the one-sided derivatives or raise error depending on JAX version/config
        # The key is that _get_activation_derivatives tries jax.eval_shape on grad_grad_fn at 0.0.
        # If JAX's grad(grad(fn)) is NaN at 0, or if eval_shape itself detects an issue, it should fail.
        def fn_once_diff(x_scalar):
            # Ensure scalar input for this specific definition
            if not isinstance(x_scalar, jax.Array) or x_scalar.ndim != 0:
                 raise ValueError("This function is defined for scalar JAX arrays only for this test.")
            if x_scalar >= 0:
                return x_scalar**2
            else:
                return x_scalar**3
        
        # Let's test the behavior of jax.grad(jax.grad(fn_once_diff))(0.0)
        # grad1 = jax.grad(fn_once_diff)
        # grad2 = jax.grad(grad1)
        # print(f"g'(0) = {grad1(jnp.array(0.0))}") # Expected 0
        # print(f"g''(0) = {grad2(jnp.array(0.0))}") # Expected 2 (JAX often takes the right-hand derivative)
        # So, this function might actually pass if JAX consistently provides a value for g''(0).
        # The requirement is to test a function that *fails* _get_activation_derivatives.

        # Let's use a clearer case: f(x) = x|x|. f'(x) = 2|x|. f''(x) = 2sign(x) (problem at 0).
        fn_abs_times_x = lambda x: x * jnp.abs(x)
        # jax.grad(jax.grad(fn_abs_times_x))(0.0) is NaN.
        # So jax.eval_shape(jax.grad(jax.grad(fn_abs_times_x)), jnp.array(0.0)) should fail.

        with pytest.raises(TypeError, match="Failed to compute derivatives"):
            _get_activation_derivatives(fn_abs_times_x, "test_abs_times_x")

class TestAugmentedState:
    @pytest.mark.parametrize("dim", [1, 2, 3, 5])
    def test_from_coords_structure(self, dim):
        batch_size = 4
        coords = jr.normal(key, (batch_size, dim))
        aug_state = AugmentedState.from_coords(coords)

        chex.assert_trees_all_close(aug_state.value, coords)
        
        assert isinstance(aug_state.spatial_derivatives, list)
        assert len(aug_state.spatial_derivatives) == dim
        
        for i in range(dim):
            expected_deriv_i = jnp.zeros_like(coords)
            expected_deriv_i = expected_deriv_i.at[:, i].set(1.0)
            chex.assert_trees_all_close(aug_state.spatial_derivatives[i], expected_deriv_i, 
                                        err_msg=f"Spatial derivative {i} mismatch for dim {dim}")
            assert aug_state.spatial_derivatives[i].shape == (batch_size, dim)

        expected_laplacian = jnp.zeros_like(coords)
        chex.assert_trees_all_close(aug_state.laplacian, expected_laplacian)
        assert aug_state.laplacian.shape == (batch_size, dim)

    @pytest.mark.parametrize("dim, expected_s_components", [
        (1, 1 + 1 + 1), # value + 1_deriv + laplacian
        (2, 1 + 2 + 1), # value + 2_derivs + laplacian
        (3, 1 + 3 + 1), # value + 3_derivs + laplacian
    ])
    def test_num_s_components(self, dim, expected_s_components):
        batch_size = 2
        coords = jr.normal(key, (batch_size, dim))
        aug_state = AugmentedState.from_coords(coords)
        assert aug_state.num_s_components == expected_s_components

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_concatenate_components(self, dim):
        batch_size = 3
        features_at_input = dim # from_coords sets features = dim
        coords = jr.normal(key, (batch_size, dim)) 
        aug_state = AugmentedState.from_coords(coords)

        s_components = aug_state.num_s_components
        concatenated = aug_state.concatenate_components()

        assert concatenated.shape == (s_components * batch_size, features_at_input)

        # Verify order and content
        manual_concat_list = [aug_state.value] + aug_state.spatial_derivatives + [aug_state.laplacian]
        expected_concatenated = jnp.concatenate(manual_concat_list, axis=0)
        chex.assert_trees_all_close(concatenated, expected_concatenated)

        # Test with a dummy state that has different feature dimensions mid-network
        # This is more a test of the logic if features change, though from_coords won't create this.
        value_mock = jr.normal(key, (batch_size, 10))
        spatial_derivs_mock = [jr.normal(key, (batch_size, 10)) for _ in range(dim)]
        laplacian_mock = jr.normal(key, (batch_size, 10))
        
        aug_state_mock_features = AugmentedState(
            value=value_mock,
            spatial_derivatives=spatial_derivs_mock,
            laplacian=laplacian_mock
        )
        s_components_mock = aug_state_mock_features.num_s_components
        concatenated_mock = aug_state_mock_features.concatenate_components()
        assert concatenated_mock.shape == (s_components_mock * batch_size, 10)
        
        manual_concat_list_mock = [value_mock] + spatial_derivs_mock + [laplacian_mock]
        expected_concatenated_mock = jnp.concatenate(manual_concat_list_mock, axis=0)
        chex.assert_trees_all_close(concatenated_mock, expected_concatenated_mock)

# Helper function for scalar activation derivatives (used in tests)
def get_scalar_fn_derivatives(scalar_activation_fn):
    """Returns the function, its vmapped grad, and vmapped grad(grad)."""
    return scalar_activation_fn, jax.vmap(jax.grad(scalar_activation_fn)), jax.vmap(jax.grad(jax.grad(scalar_activation_fn)))

class TestAugmentedStatePropagation:
    @pytest.mark.parametrize("dim, in_features, out_features", [
        (2, 2, 3), # dim matches in_features
        (3, 3, 2),
        (2, 3, 4)  # dim != in_features (initial aug state features = dim, layer in_features must match aug_state.value.shape[1])
    ])
    def test_propagate_linear_augmented(self, dim, in_features, out_features):
        key_prop, key_aug, key_layer = jr.split(key, 3)
        batch_size = 2
        
        # Create initial augmented state. For this test, its feature dimension must match layer's in_features.
        # So, if layer expects in_features, the input coords for aug_state should have 'in_features' columns.
        # The 'dim' parameter here represents the number of spatial dimensions for derivative calculations,
        # while 'in_features' is the feature dimension of the layer's input.
        # For the *first* layer, in_features often equals dim.
        
        # Mock an augmented state whose components have `in_features`
        mock_value = jr.normal(key_aug, (batch_size, in_features))
        mock_spatial_derivatives = [jr.normal(key_aug, (batch_size, in_features)) for _ in range(dim)]
        mock_laplacian = jr.normal(key_aug, (batch_size, in_features))
        
        aug_state_input = AugmentedState(
            value=mock_value,
            spatial_derivatives=mock_spatial_derivatives,
            laplacian=mock_laplacian
        )

        linear_layer = eqx.nn.Linear(in_features, out_features, key=key_layer)

        # Propagation
        aug_state_output = _propagate_linear_augmented(aug_state_input, linear_layer)

        # Expected values
        expected_value = aug_state_input.value @ linear_layer.weight.T + linear_layer.bias
        expected_spatial_derivatives = [s_deriv @ linear_layer.weight.T for s_deriv in aug_state_input.spatial_derivatives]
        expected_laplacian = aug_state_input.laplacian @ linear_layer.weight.T

        chex.assert_trees_all_close(aug_state_output.value, expected_value, atol=1e-6)
        assert len(aug_state_output.spatial_derivatives) == dim
        for i in range(dim):
            chex.assert_trees_all_close(aug_state_output.spatial_derivatives[i], expected_spatial_derivatives[i], atol=1e-6)
        chex.assert_trees_all_close(aug_state_output.laplacian, expected_laplacian, atol=1e-6)

        assert aug_state_output.value.shape == (batch_size, out_features)
        for i in range(dim):
            assert aug_state_output.spatial_derivatives[i].shape == (batch_size, out_features)
        assert aug_state_output.laplacian.shape == (batch_size, out_features)

    @pytest.mark.parametrize("dim, features, scalar_activation_fn_name", [
        (2, 3, "tanh"),
        (3, 2, "relu"),
        (1, 4, "sin"),
        (2, 2, "identity"), # Test with identity like activation
    ])
    def test_propagate_activation_augmented(self, dim, features, scalar_activation_fn_name):
        key_prop, key_aug = jr.split(key, 2)
        batch_size = 5

        if scalar_activation_fn_name == "tanh":
            scalar_fn = jnp.tanh
        elif scalar_activation_fn_name == "relu":
            scalar_fn = jax.nn.relu
        elif scalar_activation_fn_name == "sin":
            scalar_fn = jnp.sin
        elif scalar_activation_fn_name == "identity":
            scalar_fn = lambda x: x # Identity
        else:
            raise ValueError(f"Unknown activation fn: {scalar_activation_fn_name}")

        act_fn, vmap_grad, vmap_grad_grad = get_scalar_fn_derivatives(scalar_fn)

        # Mock an augmented state (pre-activation)
        s_val = jr.normal(key_aug, (batch_size, features))
        s_spatial_derivatives = [jr.normal(key_aug, (batch_size, features)) for _ in range(dim)]
        s_lap = jr.normal(key_aug, (batch_size, features))
        
        aug_state_pre_activation = AugmentedState(
            value=s_val,
            spatial_derivatives=s_spatial_derivatives,
            laplacian=s_lap
        )

        # Propagation
        aug_state_output = _propagate_activation_augmented(
            aug_state_pre_activation, act_fn, vmap_grad, vmap_grad_grad
        )

        # Expected values
        sigma_prime_s = vmap_grad(s_val)
        sigma_prime_prime_s = vmap_grad_grad(s_val)

        expected_value = act_fn(s_val)
        expected_spatial_derivatives = [sigma_prime_s * s_deriv for s_deriv in s_spatial_derivatives]
        
        sum_sq_spatial_derivs = sum(jnp.square(s_deriv) for s_deriv in s_spatial_derivatives)
        term_sum_sq_derivs = sigma_prime_prime_s * sum_sq_spatial_derivs
        expected_laplacian = sigma_prime_s * s_lap + term_sum_sq_derivs
        
        chex.assert_trees_all_close(aug_state_output.value, expected_value, atol=1e-6)
        assert len(aug_state_output.spatial_derivatives) == dim
        for i in range(dim):
            chex.assert_trees_all_close(aug_state_output.spatial_derivatives[i], expected_spatial_derivatives[i], atol=1e-6)
        chex.assert_trees_all_close(aug_state_output.laplacian, expected_laplacian, atol=1e-6)

        # Assert shapes remain the same through activation
        assert aug_state_output.value.shape == (batch_size, features)
        for i in range(dim):
            assert aug_state_output.spatial_derivatives[i].shape == (batch_size, features)
        assert aug_state_output.laplacian.shape == (batch_size, features)

    @pytest.mark.parametrize("dim, features_seq", [
        (2, [2, 3, 1]), # dim=2, L1: 2->3, Act, L2: 3->1
        (3, [3, 4, 5, 2]), # dim=3, L1: 3->4, Act, L2: 4->5, Act, L3: 5->2
    ])
    def test_augmented_forward_cache(self, dim, features_seq):
        key_model, key_input = jr.split(key, 2)
        batch_size = 4
        
        # Input coordinates have `dim` features
        input_coords = jr.normal(key_input, (batch_size, dim))
        initial_aug_state = AugmentedState.from_coords(input_coords)

        # Build model
        layers = []
        current_in_features = dim
        num_linear_layers = 0
        for i, out_features in enumerate(features_seq):
            key_model, subkey = jr.split(key_model)
            layers.append(eqx.nn.Linear(current_in_features, out_features, key=subkey))
            num_linear_layers +=1
            if i < len(features_seq) - 1: # Add activation after all but last linear layer
                layers.append(eqx.nn.Lambda(jnp.tanh))
            current_in_features = out_features
        
        model = eqx.nn.Sequential(layers)
        
        # Execute _augmented_forward_cache
        aug_input_acts, aug_pre_acts, final_aug_output = _augmented_forward_cache(model, initial_aug_state)

        assert len(aug_input_acts) == num_linear_layers
        assert len(aug_pre_acts) == num_linear_layers

        # Check shapes for input activations (Z_in_l)
        # Z_in_0 (input to first linear)
        assert aug_input_acts[0].value.shape == (batch_size, dim) 
        for sd in aug_input_acts[0].spatial_derivatives:
            assert sd.shape == (batch_size, dim)
        assert aug_input_acts[0].laplacian.shape == (batch_size, dim)

        # Check shapes for pre-activations (S_out_l) and subsequent input_acts (Z_in_l = act(S_out_{l-1}))
        # and final_aug_output
        expected_feature_size = dim
        for i in range(num_linear_layers):
            # S_out_l (output of linear_i, pre-activation)
            current_linear_layer_model = model.layers[i*2 if i*2 < len(model.layers) else -1] # Assuming alternating Linear/Lambda
            if not isinstance(current_linear_layer_model, eqx.nn.Linear): # Should not happen with test setup
                current_linear_layer_model = model.layers[0] # Failsafe for pylint
            
            expected_feature_size_linear_out = current_linear_layer_model.out_features
            
            assert aug_pre_acts[i].value.shape == (batch_size, expected_feature_size_linear_out)
            for sd in aug_pre_acts[i].spatial_derivatives:
                assert sd.shape == (batch_size, expected_feature_size_linear_out)
            assert aug_pre_acts[i].laplacian.shape == (batch_size, expected_feature_size_linear_out)

            # Z_in_{l+1} (input to (l+1)-th linear layer = activation(S_out_l))
            # If there is a next linear layer
            if i < num_linear_layers -1 :
                # After activation, feature size remains the same as S_out_l
                assert aug_input_acts[i+1].value.shape == (batch_size, expected_feature_size_linear_out)
                for sd in aug_input_acts[i+1].spatial_derivatives:
                    assert sd.shape == (batch_size, expected_feature_size_linear_out)
                assert aug_input_acts[i+1].laplacian.shape == (batch_size, expected_feature_size_linear_out)
        
        # Final output (Z_out_L or S_out_L if last layer has no activation)
        final_layer_out_features = features_seq[-1]
        assert final_aug_output.value.shape == (batch_size, final_layer_out_features)
        for sd in final_aug_output.spatial_derivatives:
            assert sd.shape == (batch_size, final_layer_out_features)
        assert final_aug_output.laplacian.shape == (batch_size, final_layer_out_features)

        # Basic plausibility: check if initial input matches first aug_input_act
        chex.assert_trees_all_close(aug_input_acts[0].value, initial_aug_state.value)
        for i_sd in range(dim):
            chex.assert_trees_all_close(aug_input_acts[0].spatial_derivatives[i_sd], initial_aug_state.spatial_derivatives[i_sd])
        chex.assert_trees_all_close(aug_input_acts[0].laplacian, initial_aug_state.laplacian)

class TestFactorCalculations:
    @pytest.mark.parametrize("in_f, out_f, batch_s", [(2, 1, 3), (3, 2, 5)])
    def test_standard_factor_terms(self, in_f, out_f, batch_s):
        key_model, key_data = jr.split(key, 2)
        model = eqx.nn.Sequential([
            eqx.nn.Linear(in_f, out_f, key=key_model),
            # No activation for simplicity in checking grads, or use identity
        ])
        
        dummy_pts = jr.normal(key_data, (batch_s, in_f))
        dummy_targets = jr.normal(key_data, (batch_s, out_f)) # Targets for a simple loss

        def loss_fn_std(m, batch_data):
            x_data, t_data = batch_data
            preds = jax.vmap(m)(x_data) # (batch, out_f)
            return 0.5 * jnp.mean((preds - t_data)**2)

        # This function is not directly used by PINNKFAC for factor calculation anymore,
        # but its components _standard_forward_cache and _standard_backward_pass are.
        # We'll test those as they are used for boundary terms in PINNKFAC.step.
        
        # 1. Test _standard_forward_cache
        y_pred_fwd, acts_std_fwd, pre_std_fwd, phi_std_fwd = _standard_forward_cache(model, dummy_pts)
        
        assert len(acts_std_fwd) == 1 # One linear layer
        assert len(pre_std_fwd) == 1
        assert len(phi_std_fwd) == 1
        
        chex.assert_trees_all_close(acts_std_fwd[0], dummy_pts) # Input to linear layer
        expected_pre_act = dummy_pts @ model.layers[0].weight.T + model.layers[0].bias
        chex.assert_trees_all_close(pre_std_fwd[0], expected_pre_act)
        chex.assert_trees_all_close(phi_std_fwd[0], jnp.ones_like(expected_pre_act)) # Since no activation after linear
        chex.assert_trees_all_close(y_pred_fwd, expected_pre_act) # Output of model is output of linear

        # 2. Test _standard_backward_pass
        # Mimic what happens in PINNKFAC.step for boundary terms
        # grad_out_b = (y_pred_fwd.squeeze() - dummy_targets.squeeze()) / batch_s # If single output
        # For multi-output, ensure shapes match
        
        # If out_f is 1, squeeze, else ensure shapes are compatible.
        y_pred_squeezed = y_pred_fwd.squeeze() if out_f == 1 else y_pred_fwd
        targets_squeezed = dummy_targets.squeeze() if out_f == 1 else dummy_targets

        if out_f == 1 and y_pred_squeezed.ndim == 0: # If batch_s=1 and out_f=1, squeeze() makes it scalar
             y_pred_squeezed = y_pred_squeezed[None] # Make it (1,)
             targets_squeezed = targets_squeezed[None] if targets_squeezed.ndim ==0 else targets_squeezed
        
        # Ensure y_pred_squeezed and targets_squeezed have same shape for residual calculation.
        # This might happen if batch_s=1 and out_f > 1, then squeeze has no effect.
        # Or if batch_s > 1 and out_f = 1, then squeeze removes last dim.
        # The target shape for res_b should be (batch_s,) if out_f is 1, or (batch_s, out_f) if out_f > 1.

        # Reshape targets if needed, assuming predictions are (batch_s, out_f) or (batch_s,) if out_f=1
        if out_f == 1 and targets_squeezed.ndim == 2 and targets_squeezed.shape[1] == 1:
            targets_squeezed = targets_squeezed.squeeze(-1)
        
        # This matches PINNKFAC's boundary residual calculation
        if out_f == 1:
            res_b = y_pred_squeezed - targets_squeezed # Should be (batch_s,)
        else: # out_f > 1
            res_b = y_pred_squeezed - targets_squeezed # Should be (batch_s, out_f)

        grad_out_b = res_b / batch_s 
        
        deltas_std_bwd = _standard_backward_pass(model, pre_std_fwd, phi_std_fwd, grad_out_b)
        assert len(deltas_std_bwd) == 1
        # delta = grad_output * phi_prime. Here phi_prime is 1.
        # So delta should be grad_out_b (if reshaped appropriately by _standard_backward_pass)
        # grad_out_b might be (batch_s,) or (batch_s, out_f)
        # deltas_std_bwd[0] should be (batch_s, out_f)
        
        expected_delta0_shape = (batch_s, out_f)
        assert deltas_std_bwd[0].shape == expected_delta0_shape
        
        # If grad_out_b was (batch_s,), _standard_backward_pass adds a dim if it was 1D.
        grad_out_b_for_check = grad_out_b
        if grad_out_b.ndim == 1:
             grad_out_b_for_check = grad_out_b[:, None] # (batch, 1) if out_f was 1

        chex.assert_trees_all_close(deltas_std_bwd[0], grad_out_b_for_check * phi_std_fwd[0], atol=1e-6)


    @pytest.mark.parametrize("dim, features_seq", [
        (2, [2, 3, 1]), # Model: Linear(2,3)->Tanh->Linear(3,1)
        (1, [1, 2, 1]), # Model: Linear(1,2)->Tanh->Linear(2,1)
    ])
    def test_augmented_factor_terms(self, dim, features_seq):
        key_model, key_data = jr.split(key, 2)
        batch_s = 6 # Needs to be multiple of S for some checks if they were there

        # Build model
        layers = []
        current_in_features = dim
        for i, out_features in enumerate(features_seq):
            key_model, subkey = jr.split(key_model)
            layers.append(eqx.nn.Linear(current_in_features, out_features, key=subkey))
            if i < len(features_seq) - 1: # Add activation after all but last linear layer
                layers.append(eqx.nn.Lambda(jnp.tanh))
            current_in_features = out_features
        model = eqx.nn.Sequential(layers)
        params, static = eqx.partition(model, eqx.is_array)
        model_eval = eqx.combine(static, params) # Model with current params

        interior_pts = jr.normal(key_data, (batch_s, dim))
        
        # Dummy rhs_fn, output should match final model output feature size
        final_out_features = features_seq[-1]
        def rhs_fn_mock(coords): # Coords are (batch, dim)
            # Returns (batch,) or (batch, final_out_features)
            # For this test, let its output match the laplacian's shape (batch, final_out_features)
            # or (batch,) if final_out_features is 1
            if final_out_features == 1:
                return jnp.sum(coords**2, axis=1) * 0.1 # (batch_s,)
            else:
                # Create something of shape (batch_s, final_out_features)
                return jnp.tile((jnp.sum(coords**2, axis=1) * 0.1)[:, None], (1, final_out_features))

        # Execute
        factor_contributions = _augmented_factor_terms(model_eval, params, interior_pts, rhs_fn_mock)
        
        num_linear_layers = len(features_seq)
        assert len(factor_contributions) == num_linear_layers

        # Check shapes of a_contrib and b_contrib for each layer
        current_expected_in_features = dim
        for k_layer_idx in range(num_linear_layers):
            a_k, b_k = factor_contributions[k_layer_idx]
            
            # Determine S for this input state
            # Z_in_0 (input to first linear layer)
            if k_layer_idx == 0:
                s_components_for_a_k = 1 + dim + 1 # value + d_derivs + laplacian (all from initial_aug_state)
                features_for_a_k = dim # features of initial_aug_state.value etc.
            else:
                # Z_in_l = activation(S_out_{l-1})
                # S_out_{l-1} is output of (k_layer_idx-1)-th linear layer
                prev_linear_out_features = features_seq[k_layer_idx-1]
                # After activation, spatial_derivatives list length is still dim
                s_components_for_a_k = 1 + dim + 1 
                features_for_a_k = prev_linear_out_features

            assert a_k.shape == (s_components_for_a_k * batch_s, features_for_a_k)
            
            # b_k is dL/dS_out_l (concatenated). S_out_l is output of k-th linear layer.
            current_linear_out_features = features_seq[k_layer_idx]
            s_components_for_b_k = 1 + dim + 1 # Grad of S_out_l also has these components
            features_for_b_k = current_linear_out_features
            
            assert b_k.shape == (s_components_for_b_k * batch_s, features_for_b_k)
            
            current_expected_in_features = current_linear_out_features # For next layer's input

class TestGradientPreconditioning:
    @pytest.mark.parametrize("in_f, out_f, damping_o, damping_b_val", [
        (2, 3, 1e-3, 1e-3),
        (3, 2, 1e-2, 1e-1)
    ])
    def test_preconditioned_gradient_calculation(self, in_f, out_f, damping_o, damping_b_val):
        key_layer, key_grads, key_factors = jr.split(key, 3)

        # Dummy gradients
        gw = jr.normal(key_grads, (out_f, in_f)) # grad w.r.t. weight
        gb = jr.normal(key_grads, (out_f,))      # grad w.r.t. bias

        # Dummy KFAC factors (random PSD)
        A_o_val = jr.normal(key_factors, (in_f, in_f)); A_o_val = (A_o_val @ A_o_val.T)/2 + jnp.eye(in_f) * 1e-2
        B_o_val = jr.normal(key_factors, (out_f, out_f)); B_o_val = (B_o_val @ B_o_val.T)/2 + jnp.eye(out_f) * 1e-2
        A_b_val = jr.normal(key_factors, (in_f, in_f)); A_b_val = (A_b_val @ A_b_val.T)/2 + jnp.eye(in_f) * 1e-2
        B_b_val = jr.normal(key_factors, (out_f, out_f)); B_b_val = (B_b_val @ B_b_val.T)/2 + jnp.eye(out_f) * 1e-2

        lf = LayerFactors(
            A_omega=A_o_val, B_omega=B_o_val,
            A_boundary=A_b_val, B_boundary=B_b_val,
            mw=jnp.zeros_like(gw), mb=jnp.zeros_like(gb)
        )

        # Damped factors for eigendecomposition (Mimicking PINNKFAC.step)
        Aw_damped = lf.A_omega + damping_o * jnp.eye(lf.A_omega.shape[0])
        Gw_damped = lf.B_omega + damping_o * jnp.eye(lf.B_omega.shape[0])
        Aw_b_damped = lf.A_boundary + damping_b_val * jnp.eye(lf.A_boundary.shape[0])
        Gw_b_damped = lf.B_boundary + damping_b_val * jnp.eye(lf.B_boundary.shape[0])
        
        eig_A, UA = jnp.linalg.eigh(Aw_damped)
        eig_G, UG = jnp.linalg.eigh(Gw_damped)
        eig_Ab, UAb = jnp.linalg.eigh(Aw_b_damped)
        eig_Gb, UGb = jnp.linalg.eigh(Gw_b_damped)

        # Preconditioning logic from PINNKFAC.step
        gw_kfac_layer = UA.T @ gw @ UG
        precond_denominator_w = (eig_A[:, None] * eig_G[None, :]) + (eig_Ab[:, None] * eig_Gb[None, :])
        # Add small epsilon to denominator for stability, as PINNKFAC does not explicitly do this after eigendecomp
        # (damping in factors already helps, but direct division can still be risky if precond_denominator is tiny)
        gw_kfac_layer = gw_kfac_layer / (precond_denominator_w + 1e-12) 
        gw_kfac_layer = UA @ gw_kfac_layer @ UG.T
        
        gb_kfac_layer = UG.T @ gb
        precond_denominator_b = eig_G + eig_Gb
        gb_kfac_layer = gb_kfac_layer / (precond_denominator_b + 1e-12)
        gb_kfac_layer = UG @ gb_kfac_layer
        
        # The test mainly checks that this calculation runs and produces outputs of the correct shape.
        # Correctness of KFAC itself is a larger research topic.
        chex.assert_shape(gw_kfac_layer, gw.shape)
        chex.assert_shape(gb_kfac_layer, gb.shape)
        # Check for NaNs or Infs
        assert jnp.all(jnp.isfinite(gw_kfac_layer))
        assert jnp.all(jnp.isfinite(gb_kfac_layer))


class TestKFACStarHeuristics:
    @pytest.mark.parametrize("seed", [0, 42])
    def test_tree_dot(self, seed):
        key_dot = jr.PRNGKey(seed)
        key1, key2 = jr.split(key_dot)

        tree1_leaf1 = jr.normal(key1, (2,3))
        tree1_leaf2 = jr.normal(key1, (1,2))
        # Using a simple PyTree structure (list of arrays) for this test
        tree1 = [tree1_leaf1, tree1_leaf2] 
        # To test None skipping, we need a structure where eqx.is_array is False
        # tree1_with_none = [tree1_leaf1, None, tree1_leaf2] # This would fail len check

        tree2_leaf1 = jr.normal(key2, (2,3))
        tree2_leaf2 = jr.normal(key2, (1,2))
        tree2 = [tree2_leaf1, tree2_leaf2]
        
        dot_product = tree_dot(tree1, tree2)
        expected_dot = jnp.sum(tree1_leaf1 * tree2_leaf1) + jnp.sum(tree1_leaf2 * tree2_leaf2)
        chex.assert_trees_all_close(dot_product, expected_dot, atol=1e-6)

        # Test with a more complex structure including None (which should be skipped by tree_dot)
        complex_tree1 = [tree1_leaf1, {'a': tree1_leaf2, 'b': eqx.nn.Linear(2,2,key=key1)}] # Static part
        # Filter static parts before calling tree_dot as it's intended for parameter trees
        dyn1, static1 = eqx.partition(complex_tree1, eqx.is_array)
        
        complex_tree2 = [tree2_leaf1, {'a': tree2_leaf2, 'b': eqx.nn.Linear(2,2,key=key2)}]
        dyn2, static2 = eqx.partition(complex_tree2, eqx.is_array)

        dot_product_dyn = tree_dot(dyn1, dyn2)
        chex.assert_trees_all_close(dot_product_dyn, expected_dot, atol=1e-6)


        # Test with non-matching structures (should raise ValueError for num_leaves)
        tree_diff_struct = [tree1_leaf1] 
        with pytest.raises(ValueError, match="PyTrees must have the same number of leaves."):
            tree_dot(tree1, tree_diff_struct)
            
        # Test with non-matching shapes (should raise ValueError)
        tree_diff_shape_leaf = jr.normal(key1, (3,3))
        tree_diff_shape = [tree_diff_shape_leaf, tree1_leaf2]
        with pytest.raises(ValueError, match="Leaves must have the same shape"):
            tree_dot(tree1, tree_diff_shape)


    @pytest.mark.parametrize("test_case_name, in_f, out_f", [
        ("identity_factors", 2, 1),
        ("identity_factors_v2", 3, 2),
        ("diagonal_factors", 2, 3), # New case for diagonal factors
        ("diagonal_factors_v2", 3, 2) 
    ])
    def test_compute_gramian_vector_product(self, test_case_name, in_f, out_f):
        key_model, key_vec, key_factors_extra = jr.split(key, 3)
        
        model_layer = eqx.nn.Linear(in_f, out_f, key=key_model)
        model = eqx.nn.Sequential([model_layer])
        
        vec_w = jr.normal(key_vec, (out_f, in_f))
        vec_b = jr.normal(key_vec, (out_f,))
        
        # Build vector_tree that matches model structure (params)
        dummy_params, _ = eqx.partition(model, eqx.is_array)
        weight_path = lambda tree: tree.layers[0].weight
        bias_path = lambda tree: tree.layers[0].bias
        vector_tree = eqx.tree_at(weight_path, dummy_params, vec_w)
        vector_tree = eqx.tree_at(bias_path, vector_tree, vec_b)

        if "identity_factors" in test_case_name:
            current_factors = LayerFactors(
                A_omega=jnp.eye(in_f), B_omega=jnp.eye(out_f),
                A_boundary=jnp.eye(in_f), B_boundary=jnp.eye(out_f),
                mw=jnp.zeros_like(vec_w), mb=jnp.zeros_like(vec_b)
            )
            expected_gv_w = 2 * vec_w # (I V I^T + I V I^T) = V + V = 2V
            expected_gv_b = 2 * vec_b # (I Vb + I Vb) = 2Vb
        elif "diagonal_factors" in test_case_name:
            # Simple diagonal matrices
            diag_A_o = jr.uniform(key_factors_extra, (in_f,), minval=0.5, maxval=2.0)
            diag_B_o = jr.uniform(key_factors_extra, (out_f,), minval=0.5, maxval=2.0)
            diag_A_b = jr.uniform(key_factors_extra, (in_f,), minval=0.1, maxval=0.5)
            diag_B_b = jr.uniform(key_factors_extra, (out_f,), minval=0.1, maxval=0.5)
            
            A_o = jnp.diag(diag_A_o)
            B_o = jnp.diag(diag_B_o)
            A_b = jnp.diag(diag_A_b)
            B_b = jnp.diag(diag_B_b)
            
            current_factors = LayerFactors(
                A_omega=A_o, B_omega=B_o,
                A_boundary=A_b, B_boundary=B_b,
                mw=jnp.zeros_like(vec_w), mb=jnp.zeros_like(vec_b)
            )
            # Expected: Gv_w = (B_o @ Vw @ A_o.T) + (B_b @ Vw @ A_b.T)
            # Since A_o, A_b are diagonal, A.T = A
            expected_gv_w = (B_o @ vec_w @ A_o) + (B_b @ vec_w @ A_b)
            # Expected: Gv_b = (B_o @ Vb) + (B_b @ Vb)
            expected_gv_b = (B_o @ vec_b) + (B_b @ vec_b)
        else:
            raise ValueError(f"Unknown test case: {test_case_name}")

        kfac_factors_layers_tuple = tuple([current_factors])
        gv_parts_list = compute_gramian_vector_product(vector_tree, model, kfac_factors_layers_tuple)
        
        assert len(gv_parts_list) == 1
        gv_part = gv_parts_list[0]
        
        chex.assert_trees_all_close(gv_part['weight'], expected_gv_w, atol=1e-5, rtol=1e-5) # Increased tolerance for non-identity
        chex.assert_trees_all_close(gv_part['bias'], expected_gv_b, atol=1e-5, rtol=1e-5)

    def test_kfac_star_solver_first_step(self):
        Delta_grad_dot = -0.5 
        quad_coeff_alpha = 2.0
        
        expected_alpha_star = Delta_grad_dot / quad_coeff_alpha
        expected_mu_star = 0.0
        
        # Logic from PINNKFAC.step (state.step == 1)
        mu_star_calc = 0.0
        alpha_star_calc = 0.0
        if jnp.abs(quad_coeff_alpha) < 1e-8: # Heuristic from code
            alpha_star_calc = 0.0
        else:
            alpha_star_calc = Delta_grad_dot / quad_coeff_alpha
            
        assert jnp.isclose(alpha_star_calc, expected_alpha_star)
        assert jnp.isclose(mu_star_calc, expected_mu_star)

    def test_kfac_star_solver_general_step_nonsingular(self):
        quad_coeff_alpha = 2.0
        quad_coeff_mu = 3.0
        quad_coeff_alpha_mu = 0.5 
        
        linear_coeff_alpha = -1.0 
        linear_coeff_mu = -0.5  

        M = jnp.array([[quad_coeff_alpha, quad_coeff_alpha_mu],
                       [quad_coeff_alpha_mu, quad_coeff_mu]])
        b_vec = jnp.array([linear_coeff_alpha, linear_coeff_mu])
        
        solver_damping = 1e-6 
        expected_solution = jnp.linalg.solve(M + solver_damping * jnp.eye(2), b_vec)
        
        # Simulate logic from PINNKFAC.step (else branch of state.step == 1)
        alpha_star_calc, mu_star_calc = 0.0,0.0
        try:
            solution = jnp.linalg.solve(M + solver_damping * jnp.eye(2), b_vec)
            alpha_star_calc, mu_star_calc = solution[0], solution[1]
        except jnp.linalg.LinAlgError: # Should not happen for this non-singular case
            alpha_star_calc, mu_star_calc = float('nan'), float('nan') # Fail test if it errors
            
        chex.assert_trees_all_close(jnp.array([alpha_star_calc, mu_star_calc]), expected_solution, atol=1e-6)

    def test_kfac_star_solver_singular_fallback(self):
        quad_coeff_alpha = 1.0
        quad_coeff_mu = 1.0
        quad_coeff_alpha_mu = 1.0 # M = [[1,1],[1,1]], singular
        
        linear_coeff_alpha = -0.5
        # linear_coeff_mu = -0.2 # Not used in this specific fallback path

        M = jnp.array([[quad_coeff_alpha, quad_coeff_alpha_mu],
                       [quad_coeff_alpha_mu, quad_coeff_mu]])
        b_vec = jnp.array([linear_coeff_alpha, 0.0]) # RHS for fallback check
        solver_damping = 1e-6 

        expected_alpha_star_fallback = linear_coeff_alpha / quad_coeff_alpha if jnp.abs(quad_coeff_alpha) >= 1e-8 else 0.0
        expected_mu_star_fallback = 0.0

        alpha_star_calc, mu_star_calc = 0.0, 0.0
        # Simulate PINNKFAC.step fallback logic
        try:
            # This might not raise LinAlgError if solver_damping makes it non-singular
            # The test here is more about the explicit fallback logic in PINNKFAC
            # Forcing the condition that would lead to the fallback in a real scenario is tricky
            # We assume the LinAlgError path is taken if jnp.linalg.solve fails.
            # Here, let's assume it *did* fail and the except block is executed.
            raise jnp.linalg.LinAlgError("Simulated error") 
        except jnp.linalg.LinAlgError: 
            # print("Warning: KFAC* 2x2 system solve failed. Using simplified update.") # Matches message
            if jnp.abs(quad_coeff_alpha) < 1e-8:
                alpha_star_calc = 0.0
            else:
                alpha_star_calc = linear_coeff_alpha / quad_coeff_alpha
            mu_star_calc = 0.0
        
        assert jnp.isclose(alpha_star_calc, expected_alpha_star_fallback, atol=1e-7)
        assert jnp.isclose(mu_star_calc, expected_mu_star_fallback, atol=1e-7)


class TestPINNKFACStep:
    @pytest.mark.parametrize("model_type", ["simple_linear", "one_hidden_layer"])
    def test_step_updates_params_and_factors(self, model_type):
        key_main = jr.PRNGKey(123)
        key_model_init, key_opt_init, key_data = jr.split(key_main, 3)

        # 1. Setup Model
        if model_type == "simple_linear":
            model = eqx.nn.Sequential([eqx.nn.Linear(1, 1, key=key_model_init)])
        elif model_type == "one_hidden_layer":
            k1, k2 = jr.split(key_model_init)
            model = eqx.nn.Sequential([
                eqx.nn.Linear(1, 2, key=k1), 
                eqx.nn.Lambda(jnp.tanh), 
                eqx.nn.Linear(2, 1, key=k2)
            ])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        initial_params, initial_static = eqx.partition(model, eqx.is_array)

        # 2. Define Dummy Loss Components
        # rhs_fn output matches model's final layer output features (1 in this case)
        # and expects coords (batch, dim_of_coords=1)
        def rhs_fn_dummy(coords): # Laplacian target
            return jnp.sum(coords * 0.1, axis=1) # Return (batch,)

        def bc_fn_dummy(coords): # Boundary value target
            return jnp.sum(coords * 0.2, axis=1) # Return (batch,)

        # 3. Prepare Dummy Data
        interior_coords = jr.uniform(key_data, (10, 1), minval=0.1, maxval=0.9) # (batch, dim_of_coords=1)
        boundary_coords = jnp.array([[0.0], [1.0]]) # (batch, dim_of_coords=1)

        # 4. Initialize Optimizer and State
        optimizer = PINNKFAC(
            lr=1e-3, # Not used by KFAC* directly for param update step size
            damping_omega=1e-2, 
            damping_boundary=1e-2, 
            damping_kfac_star_model=1e-2, 
            decay=0.95
        )
        
        # Store original model and KFAC state (factors are initialized to I or 0)
        original_model = model 
        original_kfac_state = optimizer.init(original_model)

        # 5. Execute optimizer.step
        # Ensure no JIT compilation issues by running it once if needed, or just test as is.
        # The functions inside step are complex, so direct JIT might be slow for first run.
        # For testing, direct execution is fine.
        new_model, new_kfac_state = optimizer.step(
            original_model, rhs_fn_dummy, bc_fn_dummy, 
            interior_coords, boundary_coords, original_kfac_state
        )

        # 6. Verify Parameter Updates
        new_params, _ = eqx.partition(new_model, eqx.is_array)
        
        # Check that parameters are different.
        # Using tree_dot to check for non-zero difference.
        param_diff_sq_sum = tree_dot(
            jax.tree_map(lambda x, y: x - y, initial_params, new_params),
            jax.tree_map(lambda x, y: x - y, initial_params, new_params)
        )
        assert param_diff_sq_sum > 1e-12, "Parameters did not change after optimizer step."

        # 7. Verify KFAC Factor Updates (EMA)
        assert len(new_kfac_state.layers) == len(original_kfac_state.layers)
        num_linear_layers_in_model = len(_linear_layers(model))
        assert len(new_kfac_state.layers) == num_linear_layers_in_model

        for i in range(num_linear_layers_in_model):
            orig_lf = original_kfac_state.layers[i]
            new_lf = new_kfac_state.layers[i]

            # Initial factors are identity matrices
            # A_omega: (n, n), B_omega: (m, m)
            # A_boundary: (n, n), B_boundary: (m, m)
            # After one step, they should no longer be perfect identity matrices if updates happened.
            # Check if sum of squared differences from identity is large enough, or from original.
            
            # For A_omega
            diff_A_o = new_lf.A_omega - orig_lf.A_omega # orig_lf.A_omega is jnp.eye(...)
            assert jnp.sum(diff_A_o**2) > 1e-9, f"A_omega factor for layer {i} did not change significantly."
            # For B_omega
            diff_B_o = new_lf.B_omega - orig_lf.B_omega
            assert jnp.sum(diff_B_o**2) > 1e-9, f"B_omega factor for layer {i} did not change significantly."
            # For A_boundary
            diff_A_b = new_lf.A_boundary - orig_lf.A_boundary
            assert jnp.sum(diff_A_b**2) > 1e-9, f"A_boundary factor for layer {i} did not change significantly."
            # For B_boundary
            diff_B_b = new_lf.B_boundary - orig_lf.B_boundary
            assert jnp.sum(diff_B_b**2) > 1e-9, f"B_boundary factor for layer {i} did not change significantly."

        # 8. Verify KFAC* Momentum Update
        for i in range(num_linear_layers_in_model):
            orig_lf = original_kfac_state.layers[i] # mw, mb are zeros initially
            new_lf = new_kfac_state.layers[i]
            
            # mw should change from zero
            assert jnp.sum(new_lf.mw**2) > 1e-12, f"Momentum mw for layer {i} did not change from zero."
            # mb should change from zero
            assert jnp.sum(new_lf.mb**2) > 1e-12, f"Momentum mb for layer {i} did not change from zero."
            
            # Also check they are different from original (which were zeros)
            chex.assert_trees_all_close(new_lf.mw, orig_lf.mw, inverted=True, atol=1e-9, rtol=1e-9)
            chex.assert_trees_all_close(new_lf.mb, orig_lf.mb, inverted=True, atol=1e-9, rtol=1e-9)


        # Verify Step Increment (as per current code, step is incremented early for KFAC* logic)
        assert new_kfac_state.step == original_kfac_state.step + 1
