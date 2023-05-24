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

        # X = x[:,:-1,:]
        # Y = x[:,1:,:]
        # X = np.reshape(X, (X.shape[0] * (X.shape[1]), X.shape[2])).T
        # Y = np.reshape(Y, (Y.shape[0] * (Y.shape[1]), Y.shape[2])).T

      

        # pinv = np.linalg.pinv(X)
        # print(np.shape(X))
        # print(np.shape(pinv))
        # print(np.shape(Y))
        # self.A_hat = np.dot(Y, pinv)
        # print(np.shape(self.A_hat))

        # print(f'X is: {X}')
        # print(f'Y is: {Y}')
        # print(f'self.A_hat is: {self.A_hat}')

        # if self.U is not None:
        #     print("using U")
        #     U = np.reshape(U, (U.shape[0] * U.shape[1], U.shape[2])).T
        
        #     d = X.shape[-1]
        #     matrix = Y@(np.linalg.pinv(np.vstack((X, U))))
        #     self.A_hat = matrix[:, :d]
            

        # else:
        #     print("NOT using U")
        #     pinv = np.linalg.pinv(X)
        #     self.A_hat = Y@pinv
        
        

    def act(self, x, **kwargs):
        self.U = np.random.uniform(-1,1, [len(x[0]), self._embed_dim, self._timesteps])
        self.U = np.array(self.U)

        # We normalize U to mimic the limited current we can simulate in a neuron; a larger U would represent unrealistic neural control 
        self.U =  self.U / np.linalg.norm(self.U, axis=-1)[:, :, np.newaxis]

        return self.U
    
    def _predict(self, x0, timesteps, **kwargs):
        preds = []
        traj = [x0.T]
        for _ in range(timesteps-1):
            traj.append(self.A_hat @ traj[-1])
        preds.append(traj)
        preds = np.squeeze(np.array(preds), 0)
        preds = np.transpose(preds, (2, 0, 1))
        return preds


