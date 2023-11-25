import numpy as np
from ..abstractions import AbstractAlgorithm

from sklearn.linear_model import LinearRegression as SKLinearRegression


class LinearRegression(AbstractAlgorithm):
    def __init__(self,
                 embed_dim,
                 timesteps,
                 max_control_cost,
                 seed=None):
        super().__init__(embed_dim, timesteps, max_control_cost, seed)
        self.A_hat = []
        self.U = None
        self.model = None
        self._rng = np.random.default_rng(seed=self._seed)

    def _fit(self, x: np.ndarray, **kwargs):
        N, T, D = x.shape
        X_train = x[:, :-1, :].reshape(N * (T - 1), D)
        y_train = x[:, 1:, :].reshape(N * (T - 1), D)

        self.model = SKLinearRegression()

        if self.U is not None:
            U = np.reshape(U, (U.shape[0] * U.shape[1], U.shape[2])).T
            X_train = np.vstack((X_train, U))

        self.model.fit(X_train, y_train)
        self.A_hat = self.model.coef_

    def _act(self, x, **kwargs):
        self.U = self._rng.uniform(-1, 1, [len(x[0]), self._timesteps, self._embed_dim])
        self.U = np.array(self.U)
        self.U = (self.U / np.linalg.norm(self.U, axis=-1)[:, :, np.newaxis]) * self._max_control_cost
        return self.U

    def _predict(self, x0, timesteps, **kwargs):
        preds = []
        traj = [x0.T]
        for _ in range(timesteps - 1):
            traj.append(self.A_hat @ traj[-1])
        preds.append(traj)
        preds = np.squeeze(np.array(preds), 0)
        preds = np.transpose(preds, (2, 0, 1))
        return preds
