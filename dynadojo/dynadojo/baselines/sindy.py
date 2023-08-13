import numpy as np
import pysindy as ps

from ..abstractions import AbstractModel


class SINDy(AbstractModel):
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, dt=None, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)
        self._dt = dt or 1 / timesteps
        self._model = ps.SINDy()

    def fit(self, x: np.ndarray, **kwargs) -> None:
        for example in x:
            self._model.fit(example, t=self._dt)

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        results = []
        t_test = np.arange(0, 1, self._dt)
        for point in x0:
            results.append(self._model.simulate(point, t_test))
        results = np.array(results)
        return results
