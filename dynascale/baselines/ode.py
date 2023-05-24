import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint

from ..abstractions import Model


class ODE(Model):
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, lr=3e-2, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)

        self.model = nn.Sequential(nn.Linear(embed_dim, 32), nn.Softplus(), nn.Linear(32, embed_dim))
        self.lr = lr
        self.mse_loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), self.lr)

    def _predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        pass

    def fit(self, x: np.ndarray, epochs=100, **kwargs):
        x = torch.tensor(x, dtype=torch.float32)
        state = x[:, 0, :]
        t = torch.linspace(0.0, self._timesteps, self._timesteps)
        for _ in range(epochs):
            self.opt.zero_grad()
            pred = odeint(self, state, t, method='midpoint')
            pred = pred.transpose(0, 1)
            loss = self.mse_loss(pred, x).float()
            loss.backward()
            self.opt.step()
#
# class _ODE(nn.Module):  # change inherit
#     def __init__(self, embed_dim, timesteps, lr = 3e-2, epochs = 100):
#         super().__init__()
#         self._embed_dim = embed_dim
#         self._timesteps = timesteps
#         self.lr = lr
#         self.epochs = epochs
#         self.f = nn.Sequential(nn.Linear(embed_dim, 32), nn.Softplus(), nn.Linear(32, embed_dim))
#
#     def forward(self, t, state):
#
#         dx = self.f(state)
#         return dx
#
#     def fit(self, x):
#         x = torch.tensor(x, dtype=torch.float32)
#         state = x[:, 0, :]
#         t = torch.linspace(0.0, self._timesteps, self._timesteps)
#
#         opt = torch.optim.Adam(self.f.parameters(), self.lr)
#         loss_MSE = nn.MSELoss()
#
#         for i in range(self.epochs):
#             opt.zero_grad()
#
#             pred = odeint(self, state, t, method='midpoint')
#             pred = pred.transpose(0, 1)
#
#             loss = loss_MSE(pred, x).float()
#             print(loss.item())
#             loss.backward()
#             opt.step()
#
#     def predict(self, x0):
#         x0 = torch.tensor(x0, dtype=torch.float32)
#         t = torch.linspace(0.0, self._timesteps, self._timesteps)
#         return odeint(self, x0, t)
