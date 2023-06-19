import numpy as np
import torch
import torch.nn as nn

from ..abstractions import AbstractModel


class CNN(AbstractModel):
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, lr=6e-2, k=3, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)

        self.lr = lr
        self.k = k
        self.c1 = nn.Sequential(
            nn.Conv1d(1, 32, self.k, padding='same'),
            nn.MaxPool1d(self.k),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 1, self.k, padding='same'),
            nn.BatchNorm1d(1)
        )
        n_channels = self.c1(torch.empty(1, 1, self._embed_dim)).size(-1)
        self.lin = nn.Sequential(nn.Linear(n_channels, embed_dim), nn.Sigmoid())

    def fit(self, x: np.ndarray, epochs=100, **kwargs):
        state = torch.tensor(x, dtype=torch.float32)
        batch_size = len(state)

        opt = torch.optim.Adam(self.c1.parameters(), self.lr)
        loss_BCEL = nn.BCELoss()
        lossMSE = nn.MSELoss()

        for i in range(epochs):
            opt.zero_grad()
            pred_states = state[:, 0, :].unsqueeze(1)
            loss = 0.0

            for t in range(self._timesteps - 1):
                state_t = state[:, t, :]
                state_t = torch.reshape(state_t, (batch_size, self._embed_dim)).unsqueeze(1)

                next_state = self.lin(self.c1(state_t))  # doesn't even call forward

                loss += lossMSE((next_state * (state[:, t, :].unsqueeze(1))), state[:, t+1, :].unsqueeze(1)) #loss_BCEL(next_state, state[:, t + 1, :].unsqueeze(1))

            # print(loss.item())
            loss.backward()
            opt.step()

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        state = torch.tensor(x0, dtype=torch.float32)
        pred = state.unsqueeze(1)

        for t in range(timesteps - 1):
            state_t = pred[:, -1, :].unsqueeze(1)

            next_prob = self.lin(self.c1(state_t))  # doesn't even call forward

            next_state = torch.distributions.bernoulli.Bernoulli(next_prob).sample()
            pred = torch.cat((pred, next_state), dim=1)

        return pred.detach().numpy()


class _CNN(nn.Module): #change to inherit from model instead
    def __init__(self, latent_dim, embed_dim, timesteps, lr = 6e-2, epochs = 100):
        super().__init__()
        self._latent_dim = latent_dim
        self._embed_dim = embed_dim
        self._timesteps = timesteps
        self.lr = lr
        self.epochs = epochs
        self.k = 5
        
        self.c1 = nn.Sequential(nn.Conv1d(1, 32, self.k, padding='same'), nn.MaxPool1d(self.k), nn.Conv1d(32, 1, self.k, padding='same'))
        
        n_channels = self.c1(torch.empty(1, 1, self._embed_dim)).size(-1)
        self.lin = nn.Sequential(nn.Linear(n_channels, embed_dim), nn.Sigmoid())

    def forward(self, x):
        return self.lin(self.c1(x))
    
    def fit(self, x):
        
        state = torch.tensor(x, dtype=torch.float32) 
        batch_size = len(state)
        
        opt = torch.optim.Adam(self.c1.parameters(), self.lr)
        loss_BCEL = nn.BCELoss()

        for i in range(self.epochs):
            opt.zero_grad()
            pred_states = state[:, 0, :].unsqueeze(1)
            loss = 0.0
           
            for t in range(self._timesteps-1):
                state_t = state[:, t, :]
                state_t = torch.reshape(state_t, (batch_size, self._embed_dim)).unsqueeze(1)
                
                next_state = self.lin(self.c1(state_t)) #doesnt even call forward
              
                loss += loss_BCEL(next_state, state[:, t+1, :].unsqueeze(1))
            
            print(loss.item())
            loss.backward()
            opt.step()

    def predict(self, x0, timesteps):
        
        state = torch.tensor(x0, dtype=torch.float32) 
        pred = state.unsqueeze(1)
           
        for t in range(timesteps-1):
            state_t = pred[:, -1, :].unsqueeze(1)
            
            next_prob = self.lin(self.c1(state_t)) #doesnt even call forward

            next_state = torch.distributions.bernoulli.Bernoulli(next_prob).sample()
            pred = torch.cat((pred, next_state), dim=1)

        return pred.detach().numpy()
