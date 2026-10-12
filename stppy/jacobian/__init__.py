"""Phase-gradient (propagation) analysis of spatiotemporal patterns."""

from .phase_gradient import (
    build_gradient_operators,
    compute_phase_gradient,
    propagation_field,
)
from .pole_arrows import PoleArrow, classify_propagation, pole_arrows

__all__ = [
    'build_gradient_operators',
    'compute_phase_gradient',
    'propagation_field',
    'PoleArrow',
    'classify_propagation',
    'pole_arrows',
]
