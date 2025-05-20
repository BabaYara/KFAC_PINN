"""KFAC_PINN Package

Implementation of the Kronecker-Factored Approximate Curvature (KFAC)
optimizer for Physics-Informed Neural Networks (PINNs). The package exposes a
full optimiser and helper routines built on top of ``jax`` and ``equinox``.
"""

__version__ = "0.3.0"

from .kfac import KFAC
from .pinn_kfac import PINNKFAC
from . import pinn
from . import pdes
from . import training

__all__ = ["KFAC", "PINNKFAC", "pinn", "pdes", "training", "__version__"]
