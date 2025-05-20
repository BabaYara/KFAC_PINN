# KFAC_PINN

A small Python package implementing the Kronecker-Factored Approximate Curvature
(KFAC) optimizer for Physics-Informed Neural Networks (PINNs). It is built using
[JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

This repository now provides a *full* KFAC implementation for fully connected
networks built from ``equinox.nn.Linear`` layers. The optimizer maintains
Kronecker-factored estimates of the curvature matrices for each layer and uses
them to precondition parameter updates. It is suitable for the PINN examples
included here as well as more general applications.

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

The package exposes two optimizers: a lightweight `KFAC` implementation and
`PINNKFAC`, which follows Algorithm&nbsp;1 in the accompanying paper and keeps
separate Kronecker factors for PDE and boundary contributions. In addition the
package provides utilities for constructing PINNs, PDE helper functions and a
small training loop helper. See the notebooks in the `examples/` directory for
demonstrations.

```python
import jax
import jax.numpy as jnp
import equinox as eqx

from kfac_pinn import PINNKFAC, pinn, training

model = pinn.make_mlp()
opt = PINNKFAC(lr=1e-2)

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
- `examples/poisson_2d.ipynb` – 2D Poisson problem.
- `examples/heat_equation.ipynb` – Basic 1D heat equation demo.
- `examples/full_kfac_demo.ipynb` – Demonstrates the full KFAC optimiser.

Run them with Jupyter to see the optimizer in action.

When opening the notebooks directly inside the `examples/` folder, make sure
the package can be imported by installing it in editable mode:

```bash
pip install -e .
```

The notebooks also include a small snippet that automatically adjusts the
Python path so they work out-of-the-box when run from the `examples/`
directory.
