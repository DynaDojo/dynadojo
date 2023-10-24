import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from ..abstractions import AbstractModel


class ODE(AbstractModel):
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, lr=3e-2, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)

        self.model = nn.Sequential(nn.Linear(embed_dim, 32), nn.Softplus(), nn.Linear(32, embed_dim))
        self.lr = lr
        self.mse_loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), self.lr)
    
    def forward(self, t, state):
        dx = self.model(state)
        return dx

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        pass

    def fit(self, x: np.ndarray, epochs=100, **kwargs):
        x = torch.tensor(x, dtype=torch.float32)
        state = x[:, 0, :]
        t = torch.linspace(0.0, self._timesteps, self._timesteps)
        for _ in range(epochs):
            self.opt.zero_grad()
            pred = odeint(self.forward, state, t, method='rk4')
            pred = pred.transpose(0, 1)
            loss = self.mse_loss(pred, x).float()
            loss.backward()
            self.opt.step()
    
    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        x0 = torch.tensor(x0, dtype=torch.float32)
        t = torch.linspace(0.0, timesteps, timesteps)
        return odeint(self, x0, t).detach().numpy()
