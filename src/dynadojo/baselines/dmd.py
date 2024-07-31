"""
Dynamic Mode Decomposition.
"""

import numpy as np
import pydmd

from ..abstractions import AbstractAlgorithm


class DMD(AbstractAlgorithm):
    """
    Dynamic mode decomposition. Implementation uses the ``pydmd`` library [1]_.

    Note
    -----
    For an example of ``DMD`` with a challenge, see ``LDSystem``.

    References
    ------------
    .. [1] https://pydmd.github.io/PyDMD/

    Example
    ---------
    .. include:: ../dmd_example.rst
    """
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float = 0, **kwargs):
        """
        Initialize the class.

        Parameters
        -------------
        embed_dim : int
            Recommended embedding dimension should be <5.
        timesteps : int
            Timesteps in the training trajectories.
        max_control_cost : float
            Ignores control, defaults to 0.
        **kwargs :
            Additional keyword arguments.
        """
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)
        self._model = pydmd.OptDMD(svd_rank=embed_dim, factorization="svd")

    def fit(self, x: np.ndarray, **kwargs) -> None:
        self._model = self._model.fit(x[0].T)
        pred = self._model.predict(x[0].T)
        loss = self.mse(x[0].T, pred)
        return {
            "train_loss": loss
        }

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        result = [x0.T]
        for _ in range(timesteps - 1):
            result.append(self._model.predict(result[-1]))
        result = np.array(result)
        result = np.transpose(result, axes=(2, 0, 1))
        return result
    
    def mse(self, actual, pred): 
        actual, pred = np.array(actual), np.array(pred)
        return np.square(np.subtract(actual,pred)).mean() 
