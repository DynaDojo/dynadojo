import numpy as np
import torch
import torch.nn as nn

from ..abstractions import AbstractAlgorithm


class GRU_RNN(AbstractAlgorithm):
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, lr=1e-3, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)
        self.hidden_dim = 32 #128
        self.num_layers = 5
        self.lr = lr
        self.gru = nn.GRU(embed_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, embed_dim)
        self.model = nn.ModuleList([self.gru, self.fc])
        #self.model = nn.Sequential(nn.Linear(embed_dim, 32), nn.Softplus(), nn.Linear(32, embed_dim))
        #self.model = GRUModel(embed_dim,hidden_size=self.hidden_dim,num_layers=self.num_layers)
        #self.model = SimpleLSTM(input_size=embed_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers)

    def forward(self, x):
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        #out, _ = self.gru(x, h0)
        #out = out.reshape(out.shape[0], -1)
        #out = self.fc(out)
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        
        return out #self.model(x)

    def fit(self, x: np.ndarray, epochs=5000, **kwargs):
        state = torch.tensor(x, dtype=torch.float32)
        batch_size = len(state)

        opt = torch.optim.Adam(self.model.parameters(), self.lr)
        lossMSE = nn.MSELoss()

        for epoch in range(epochs):
            opt.zero_grad()
            pred_states = state[:, 0, :].unsqueeze(1)
            #print(state.shape)
            loss = 0.0
            losses = []

            #pred_states = self.model(state)
            #print(pred_states.shape,state.shape)
            #loss = lossMSE(pred_states, state)

            for t in range(self._timesteps - 1):
                state_t = state[:, t, :]
                state_t = torch.reshape(state_t, (batch_size, self.embed_dim)).unsqueeze(1)
                #print('state_t: ', state_t.shape)
                #self.x = self.x.reshape(self.x.size(0), 1, 28, 28).squeeze(1)
                next_state = self.forward(state_t)
                #next_state = self.model(state_t)
                #print('next_state: ', next_state.shape)
                #print('state[:, t+1, :]: ', state[:, t+1, :].shape)
                #loss += lossMSE((next_state * (state[:, t, :].unsqueeze(1))), state[:, t+1, :].unsqueeze(1))
                loss += lossMSE(next_state,(state[:, t+1, :]))
                #loss += lossMSE(next_state,torch.reshape(state[:, t+1, :], (batch_size, self.embed_dim)).unsqueeze(1))

            losses.append(loss.item())
            if epoch % 10 == 0 and kwargs.get('verbose', False):
                print(f"Epoch {epoch}, Loss {loss.item()}")

            loss.backward()
            opt.step()

        return losses

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        state = torch.tensor(x0, dtype=torch.float32)
        print(state.shape)
        pred = state.unsqueeze(1)
        #pred = self.model(state.unsqueeze(1)) #.unsqueeze(1)
        #print(pred.shape)

        for t in range(timesteps - 1):
            state_t = pred[:, -1, :].unsqueeze(1)
            #print(state_t.shape)
            #next_state = self.forward(state_t)
            next_state = self.forward(state_t).unsqueeze(1)
            #next_state = self.model(state_t).unsqueeze(1)
            #next_state = torch.reshape(next_state, (len(state), self.embed_dim)).unsqueeze(1)
            #print(next_state.shape)
            pred = torch.cat((pred, next_state), dim=1)
        #print(pred.shape)
        return pred.detach().numpy()



class GRUModel(nn.Module):
    def __init__(self, embed_dim: int, hidden_size=32, num_layers=5):
        super(GRUModel, self).__init__()
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
    

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=32, num_layers=5):
        super(SimpleLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x,(h0, c0))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out