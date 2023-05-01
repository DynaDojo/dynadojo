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

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs):
        preds = [x0]
        for _ in range(timesteps - 1):  # TODO: add general extensions for timesteps that are longer or shorter
            xi = preds[-1] @ self.exp_At  # TODO: check if multiplication should be other way
            preds.append(xi)
        preds = np.array(preds)
        preds = np.transpose(preds, axes=(1, 0, 2))
        return preds