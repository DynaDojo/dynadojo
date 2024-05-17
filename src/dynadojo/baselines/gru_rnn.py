import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split

from ..abstractions import AbstractAlgorithm


class GRU_RNN(AbstractAlgorithm):
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, lr=1e-3, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)
        if kwargs.get('seed',None):
            torch.manual_seed(kwargs['seed'])
        self.hidden_size = kwargs.get('hidden_size', 32) #128
        self.num_layers = kwargs.get('num_layers', 5)
        self.lr = lr
        self.gru = nn.GRU(embed_dim, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, embed_dim)
        self.model = nn.ModuleList([self.gru, self.fc])

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return out

    def fit(self, x: np.ndarray, epochs=1000, **kwargs):
        state = torch.tensor(x, dtype=torch.float32)
        early_stopping = kwargs.get('early_stopping', False)
        if early_stopping:
            early_stopper = EarlyStopper(patience=kwargs.get('patience', 100), min_delta=kwargs.get('min_delta', 10))
            state, val_state = map(lambda x: x.dataset, random_split(state, [0.8,0.2]))
        batch_size = len(state)

        opt = torch.optim.Adam(self.model.parameters(), self.lr)
        lossMSE = nn.MSELoss()

        for epoch in range(epochs):
            opt.zero_grad()
            loss = 0.0
            losses = []
            if early_stopping:
                val_loss = 0.0

            for t in range(self._timesteps - 1):
                state_t = state[:, t, :]
                state_t = torch.reshape(state_t, (batch_size, self.embed_dim)).unsqueeze(1)
                next_state = self.forward(state_t)
                loss += lossMSE(next_state,(state[:, t+1, :]))
                if early_stopping:
                    val_state_t = val_state[:, t, :]
                    val_state_t = torch.reshape(val_state_t, (len(val_state), self.embed_dim)).unsqueeze(1)
                    val_next_state = self.forward(val_state_t)
                    val_loss += lossMSE(val_next_state, (val_state[:, t+1, :]))

            loss.backward()
            opt.step()

            losses.append(loss.item())
            log_freq = kwargs.get('log_freq', 100)
            if epoch % log_freq == 0 and kwargs.get('verbose', False):
                if early_stopping:
                    print(f"Epoch {epoch}, Loss {loss.item()}, Validation Loss {val_loss.item()}")
                else:
                    print(f"Epoch {epoch}, Loss {loss.item()}")
            if early_stopping and early_stopper.early_stop(val_loss.item()):
                break

        return losses

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        state = torch.tensor(x0, dtype=torch.float32)
        pred = state.unsqueeze(1)

        for t in range(timesteps - 1):
            state_t = pred[:, -1, :].unsqueeze(1)
            next_state = self.forward(state_t).unsqueeze(1)
            pred = torch.cat((pred, next_state), dim=1)
        
        return pred.detach().numpy()


class EarlyStopper:
    def __init__(self, patience=10, min_delta=1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class _GRU(nn.Module):
    def __init__(self, embed_dim: int, hidden_size=32, num_layers=5):
        super(_GRU, self).__init__()
        self.hidden_size  = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(embed_dim , hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, embed_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out,_ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
    

class _LSTM(nn.Module):
    def __init__(self, embed_dim: int, hidden_size=32, num_layers=5):
        super(_LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, embed_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x,(h0, c0))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out