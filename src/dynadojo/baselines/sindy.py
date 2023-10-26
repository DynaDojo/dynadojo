import numpy as np
import pysindy as ps

from ..abstractions import AbstractModel


class SINDy(AbstractModel):
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, differentiation_method=None, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)
        np.random.seed(self._seed)
        optimizer = ps.STLSQ(threshold=0.1)
        self._timesteps = timesteps
        self._model = ps.SINDy(
            differentiation_method=differentiation_method,
            optimizer=optimizer,
            discrete_time=True,

        )

    #TODO: add control!

    def fit(self, x: np.ndarray, **kwargs) -> None:
        X = [*x]
        self._model.fit(X, multiple_trajectories=True, library_ensemble=True, quiet=True)

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        # really no way to do this without a for loop?
        results = [ self._model.simulate(point, timesteps) for point in x0 ]
        results = np.array(results)
        return results
