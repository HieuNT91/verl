from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple

from sklearn.gaussian_process.kernels import Kernel, RBF, WhiteKernel, ConstantKernel as C


@dataclass
class GPRConfig:
    """Configuration for Gaussian Process Regression variance predictor.

    Attributes:
        length_scale: Initial length scale for RBF kernel.
        length_scale_bounds: Bounds for length scale hyperparameter.
        const_value: Initial constant kernel amplitude.
        const_value_bounds: Bounds for constant kernel amplitude.
        noise_level: Initial WhiteKernel noise level.
        noise_level_bounds: Bounds for WhiteKernel noise level.
        n_restarts_optimizer: Number of restarts for optimizer.
        alpha: Added to the diagonal of the kernel matrix for numerical stability.
        normalize_y: Whether to normalize target values.
        kernel: Optional explicit kernel to override the composed default.
        random_state: Random seed for model initialization.
    """

    length_scale: float = 1.0
    length_scale_bounds: Tuple[float, float] = (1e-2, 1e3)
    const_value: float = 1.0
    const_value_bounds: Tuple[float, float] = (1e-3, 1e3)
    noise_level: float = 1e-1
    noise_level_bounds: Tuple[float, float] = (1e-6, 1e1)

    n_restarts_optimizer: int = 3
    alpha: float = 1e-10
    normalize_y: bool = True

    kernel: Optional[Kernel] = None
    random_state: Optional[int] = 42

    # Target transform options
    use_logit: bool = True
    logit_eps: float = 1e-6
    transform_std_to_prob: bool = True  # delta-method approx for std back to prob space

    def build_kernel(self) -> Kernel:
        if self.kernel is not None:
            return self.kernel
        return RBF(
            length_scale=self.length_scale,
            length_scale_bounds=self.length_scale_bounds,
        ) 
