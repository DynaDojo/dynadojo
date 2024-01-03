"""
Dynamic Mode Decomposition
============================
"""

import numpy as np
import pydmd

from ..abstractions import AbstractAlgorithm


class DMD(AbstractAlgorithm):
    """
    Dynamic mode decomposition. Implementation uses the ``pydmd`` library [1]_.

    References
    ------------
    .. [1] https://pydmd.github.io/PyDMD/
    """
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)
        self._model = pydmd.OptDMD(svd_rank=embed_dim, factorization="svd")

    def fit(self, x: np.ndarray, **kwargs) -> None:
        self._model = self._model.fit(x[0].T)

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        result = [x0.T]
        for _ in range(timesteps - 1):
            result.append(self._model.predict(result[-1]))
        result = np.array(result)
        result = np.transpose(result, axes=(2, 0, 1))
        return result
