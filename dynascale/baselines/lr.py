import numpy as np
from ..abstractions import Model

class NaiveLinearRegression(Model):
    def __init__(self, latent_dim, embed_dim, timesteps):
        super().__init__(latent_dim, embed_dim, timesteps)
        self.exp_At = np.zeros((embed_dim, embed_dim))


    def fit(self, x: np.ndarray, **kwargs):
        head = x[:, :-1].reshape(self.embed_dim, -1)
        tail = x[:, 1:].reshape(self.embed_dim, -1)
        self.exp_At = tail @ np.linalg.pinv(head)

    def _predict(self, x0: np.ndarray, timesteps: int, **kwargs):
        preds = [x0.T]
        for _ in range(timesteps - 1):
            xi = self.exp_At @ preds[-1]
            preds.append(xi)
        preds = np.array(preds)
        preds = np.transpose(preds, axes=(2, 0, 1))
        return preds
