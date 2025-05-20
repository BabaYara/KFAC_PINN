"""KFAC_PINN Package

A simple implementation of the Kronecker-Factored Approximate Curvature (KFAC)
optimizer for Physics-Informed Neural Networks (PINNs). This package exposes a
basic optimizer and helper routines built on top of ``jax`` and ``equinox``.
"""

__version__ = "0.1.0"

from .kfac import KFAC
from . import pinn
from . import pdes
from . import training

__all__ = ["KFAC", "pinn", "pdes", "training", "__version__"]
