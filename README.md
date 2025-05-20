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

The package exposes a minimal `KFAC` optimizer and helper utilities for building
PINNs. See the notebooks in the `examples/` directory for demonstrations.

```python
import jax
import jax.numpy as jnp
import equinox as eqx

from kfac_pinn import KFAC, pinn

model = pinn.make_mlp()
opt = KFAC(lr=1e-2)
state = opt.init(model, jnp.ones((10, 1)))

# define your physics loss here
```

## Examples

Two example notebooks are provided:

- `examples/basic_pinn.ipynb` – Training a simple PINN using KFAC.
- `examples/custom_network.ipynb` – Demonstrates creating a custom network.

Run them with Jupyter to see the optimizer in action.
