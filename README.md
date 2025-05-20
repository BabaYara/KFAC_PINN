# KFAC_PINN

A small Python package implementing the Kronecker-Factored Approximate Curvature
(KFAC) optimizer for Physics-Informed Neural Networks (PINNs). It is built using
[JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

## Installation

Clone the repository and install the requirements:

```bash
pip install -r requirements.txt
```

## Usage

The package exposes a minimal `KFAC` optimizer, utilities for constructing
PINNs and a small training loop helper.  See the notebooks in the
`examples/` directory for demonstrations.

```python
import jax
import jax.numpy as jnp
import equinox as eqx

from kfac_pinn import KFAC, pinn, training

model = pinn.make_mlp()
opt = KFAC(lr=1e-2)

def loss_fn(m, x):
    res = pinn.pinn_residual(m, x, lambda x: jnp.zeros_like(x))
    return jnp.mean(res ** 2)

data = [jnp.linspace(0, 1, 32).reshape(-1, 1)] * 100
model, state = training.train(model, opt, loss_fn, data, steps=100)
```

## Examples

Several example notebooks are provided:

- `examples/basic_pinn.ipynb` – Train a simple PINN solving a Poisson equation.
- `examples/custom_network.ipynb` – Demonstrates creating a custom network.
- `examples/train_poisson.ipynb` – Full PINN training loop using the helper utilities.

Run them with Jupyter to see the optimizer in action.
