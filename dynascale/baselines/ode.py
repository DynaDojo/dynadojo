class ODE(nn.Module):
    def __init__(self, latent_dim, embed_dim, timesteps, lr = 3e-2, epochs = 100):
        super().__init__()
        self._latent_dim = latent_dim
        self._embed_dim = embed_dim
        self._timesteps = timesteps
        self.lr = lr
        self.epochs = epochs
        self.f = nn.Sequential(nn.Linear(embed_dim, 32), nn.Softplus(), nn.Linear(32, embed_dim))

    def latent_dim(self):
        return self._latent_dim
    
    def embed_dim(self):
        return self._embed_dim
    
    def timesteps(self):
        return self._timesteps
    
    def forward(self, t, state):
        
        dx = self.f(state)
        return dx
    
    def fit(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        state = x[:, 0, :]
        t = torch.linspace(0.0, self._timesteps, self._timesteps)

        opt = torch.optim.Adam(self.f.parameters(), self.lr)
        loss_MSE = nn.MSELoss()
        
        for i in range(self.epochs):
            opt.zero_grad()
           
            pred = odeint(self, state, t, method='midpoint')
            pred = pred.transpose(0, 1)
            
            loss = loss_MSE(pred, x).float()
            print(loss.item())
            loss.backward()
            opt.step()

    def predict(self, x0):
        x0 = torch.tensor(x0, dtype=torch.float32)
        t = torch.linspace(0.0, self._timesteps, self._timesteps)
        return odeint(self, x0, t)
