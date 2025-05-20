"""KFAC_PINN Package

A simple implementation of the Kronecker-Factored Approximate Curvature (KFAC)
optimizer for Physics-Informed Neural Networks (PINNs). This package exposes a
basic optimizer and helper routines built on top of ``jax`` and ``equinox``.
"""

from .kfac import KFAC
from . import pinn

__all__ = ["KFAC", "pinn"]
