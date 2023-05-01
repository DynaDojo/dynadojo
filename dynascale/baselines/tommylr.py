import numpy as np
from ..abstractions import Model as MyModel

class TommyLinearRegression(MyModel):
    def __init__(self):
        self.A_hat = []
        self.nextU = []

    def fit(self, x: np.ndarray, **kwargs):
        X = x[:,:-1,:]
        Y = x[:,1:,:]
        B = kwargs.B
        U = kwargs.U
        
        X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2])).T
        Y = np.reshape(Y, (Y.shape[0] * Y.shape[1], Y.shape[2])).T
        U = np.reshape(U, (U.shape[0] * U.shape[1], U.shape[2])).T

        # If B is known (i.e., if B is the identity matrix)
        if len(B):
            A_hat = (Y - (B @ U))@np.linalg.pinv(X)
            self.A_hat = A_hat

        else: 
            d = X.shape[-1]
            matrix = Y@(np.linalg.pinv(np.vstack((X, U))))
            self.A_hat = matrix[:, :d]
            self.B_hat = matrix[:, d:]
        
    def act(self, x, **kwargs):
        B =  kwargs.B
        U =  kwargs.U
        shouldPredictB = kwargs.shouldPredictB

        if shouldPredictB:
            self.train(x, U)
        else:
            self.train(x, U, B)
        
        self.nextU = np.random.uniform(-1,1, [U.shape[0], U.shape[1], U.shape[2]])
        self.nextU = np.array(self.nextU)
        # We normalize U to mimic the limited current we can simulate in a neuron; a larger U would represent unrealistic neural control 
        self.nextU =  self.nextU / np.linalg.norm(self.nextU, axis=-1)[:, :, np.newaxis]

        if shouldPredictB:
            return (self.nextU, self.B_hat)
        else:
            return (self.nextU)
    
    def predict(self, x0, timesteps, **kwargs):
        allPredicted = []
        for sample in x0:
            trajectory = [sample]
            for x in range(timesteps):
                trajectory.append(self.A_hat @ trajectory[-1])
            allPredicted.append(trajectory)
        return np.array(allPredicted)

