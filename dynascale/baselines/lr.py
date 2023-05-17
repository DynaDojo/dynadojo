import numpy as np
from ..abstractions import Model

class LinearRegression(Model):
    def __init__(self, latent_dim, embed_dim, timesteps):
        super().__init__(latent_dim, embed_dim, timesteps)
        self.A_hat = []
        self.nextU = []
        self.U = None
    
    def fit(self, x: np.ndarray, **kwargs):
        X = x[:,:-1,:]
        Y = x[:,1:,:]
        U = self.U
        
        X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2])).T
        Y = np.reshape(Y, (Y.shape[0] * Y.shape[1], Y.shape[2])).T

        if U is not None:
            U = np.reshape(U, (U.shape[0] * U.shape[1], U.shape[2])).T
        
            d = X.shape[-1]
            matrix = Y@(np.linalg.pinv(np.vstack((X, U))))
            self.A_hat = matrix[:, :d]

        else:
            self.A_hat = Y@np.linalg.pinv(X)

    def act(self, x, **kwargs):
        self.nextU = np.random.uniform(-1, 1, [len(x[0]), self._embed_dim, self._timesteps])
        self.nextU = np.array(self.nextU)

        # We normalize U to mimic the limited current we can simulate in a neuron; a larger U would represent unrealistic neural control 
        self.nextU =  self.nextU / np.linalg.norm(self.nextU, axis=-1)[:, :, np.newaxis]

        return self.nextU
    
    def _predict(self, x0, timesteps, **kwargs):
        preds = []
        traj = [x0.T]
        for _ in range(timesteps-1):
            traj.append(self.A_hat @ traj[-1])
        preds.append(traj)
        preds = np.squeeze(np.array(preds), 0)
        preds = np.transpose(preds, (2, 0, 1))
        return preds


