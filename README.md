# KFAC_PINN

A small Python package implementing the Kronecker-Factored Approximate Curvature
(KFAC) optimizer for Physics-Informed Neural Networks (PINNs). It is built using
[JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).
The optimizer included here is a **minimal diagonal variant** that approximates
the Fisher information with a running average of squared gradients. It is not a
full KFAC implementation but is sufficient for the simple PINN demonstrations
in this repository.

## Installation

Clone the repository and install the requirements:

```bash
pip install -r requirements.txt
```

Alternatively install the package in editable mode using `pyproject.toml`:

```bash
pip install -e .
```

## Usage

The package exposes a minimal `KFAC` optimizer, utilities for constructing
PINNs, PDE helper functions and a small training loop helper. See the
notebooks in the `examples/` directory for demonstrations.

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

- `examples/basic_pinn.ipynb` – Train a 1D Poisson PINN.
- `examples/custom_network.ipynb` – Demonstrates creating a custom network.
- `examples/train_poisson.ipynb` – Full 1D training loop.
- `examples/poisson_2d.ipynb` – New example solving a 2D Poisson problem.
- `examples/heat_equation.ipynb` – Basic 1D heat equation demo.

Run them with Jupyter to see the optimizer in action.

When opening the notebooks directly inside the `examples/` folder, make sure
the package can be imported by installing it in editable mode:

```bash
pip install -e .
```

The notebooks also include a small snippet that automatically adjusts the
Python path so they work out-of-the-box when run from the `examples/`
directory.
