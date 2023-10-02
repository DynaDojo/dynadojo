import numpy as np
import pysindy as ps

from ..abstractions import AbstractModel


class SINDy(AbstractModel):
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, differentiation_method=None, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)
        optimizer = ps.STLSQ(threshold=0.2)
        self._t = np.linspace(0, 1, timesteps)
        self._model = ps.SINDy(
            differentiation_method=differentiation_method,
            optimizer=optimizer,
        )

    def fit(self, x: np.ndarray, **kwargs) -> None:
        for example in x:
            self._model.fit(example, t=self._t)

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        results = []
        for point in x0:
            results.append(self._model.simulate(point, self._t))
        results = np.array(results)
        return results
