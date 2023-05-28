import numpy as np
from sklearn.linear_model import LinearRegression

from ..abstractions import Model


class MyLinearRegression(Model):
    def __init__(self, embed_dim, timesteps, control_constraint):
        super().__init__(embed_dim, timesteps, control_constraint)
        self.A_hat = []
        self.U = None
        self.model = None
    
    def fit(self, x: np.ndarray, **kwargs):

        N, T, D = x.shape
        X_train = x[:, :-1, :].reshape(N * (T - 1), D)
        y_train = x[:, 1:, :].reshape(N * (T - 1), D)

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self.A_hat = self.model.coef_

    # def _act(self, x, **kwargs):
    #     self.U = np.random.uniform(-1,1, [len(x[0]), self._embed_dim, self._timesteps])
    #     self.U = np.array(self.U)
    #
    #     # We normalize U to mimic the limited current we can simulate in a neuron; a larger U would represent unrealistic neural control
    #     self.U = self.U / np.linalg.norm(self.U, axis=-1)[:, :, np.newaxis]
    #
    #     return self.U
    
    def predict(self, x0, timesteps, **kwargs):
        preds = []
        traj = [x0.T]
        for _ in range(timesteps-1):
            traj.append(self.A_hat @ traj[-1])
        preds.append(traj)
        preds = np.squeeze(np.array(preds), 0)
        preds = np.transpose(preds, (2, 0, 1))
        return preds


