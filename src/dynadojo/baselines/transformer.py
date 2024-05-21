"""
Decoder Transformer model for dynamical systems time series forecasting.
==============================================================
"""

import copy
import math
from ..abstractions import AbstractAlgorithm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class DecoderTransformer(AbstractAlgorithm, nn.Module):
    """ 
    Decoder-only transformer model for dynamical systems time series forecasting. Trained using Adam optimizer and Mean Squared Error loss function. 

    Resources:
    - https://nlp.seas.harvard.edu/2018/04/03/attention.html#training-loop
    - https://github.com/karpathy/minGPT/tree/master 
    - https://github.com/urchade/transformer-tutorial/blob/master/3%20GPT.ipynb

    Parameters
    ----------
    embed_dim : int
        The embedded dimension of the system and the input dimension of the transformer model. 
    timesteps : int
        The timesteps of the training trajectories. Must be greater than 2.
    max_control_cost : float, optional
        Ignores control, so defaults to 0.
    seed : int, optional
        Seed for reproducibility. Defaults to None.
    ============================== Transformer hyperparameters
    model_dim : int, optional
        The dimension of the transformer model. Defaults to 512.
    num_heads : int, optional
        The number of heads in the multi-head attention mechanism. Defaults to 8.
    num_layers : int, optional
        The number of layers in the transformer model. Defaults to 6.
    dropout : float, optional
        The dropout rate. Defaults to 0.1.
    ============================== Optimizer hyperparameters
    learning_rate : float, optional
        The learning rate of the Adam optimizer. Defaults to 0.001.
    **kwargs : dict, optional
        Additional keyword arguments
    """
    def __init__(self, embed_dim, timesteps, max_control_cost=0, seed=None, 
                 # Decoder layer hyperparameters
                 model_dim=512, num_heads=16, dim_feedforward=2048, dropout=0.1, activation='gelu', 
                 # Decoder Transformer hyperparameters
                 num_layers=6, norm=None, 
                 **kwargs):
        AbstractAlgorithm.__init__(self, embed_dim, timesteps, max_control_cost, seed=seed, **kwargs)
        nn.Module.__init__(self)

        self.model_dim = model_dim

        if seed:
            torch.manual_seed(seed)
            

        # Create decoder only transformer model from scratch
        self.embedding = nn.Linear(embed_dim, model_dim)
        self.max_timesteps = 5000
        self.positional_encoder = PositionalEncoding(model_dim, max_len=self.max_timesteps)      
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, 
                                                   activation=activation, norm_first = False, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, embed_dim)

        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self.dummy_param.device
    
    # nn.Module forward method
    def forward(self, x: torch.Tensor):
        """Forward pass of the transformer model.

        Parameters
        ----------
        x : torch.Tensor
            (batch_size, seq_len, input_dim) Input tensor.

        Returns
        -------
        torch.Tensor
            (batch_size, seq_len, input_dim) Output tensor.
        """
        assert x.shape[1] <= self.max_timesteps, f"Sequence length {x.shape[1]} exceeds maximum sequence length {self.max_timesteps}"
        
        batch_size, seq_len, input_dim = x.size()
        # Embed input to model_dim
        x = self.embedding(x)
        # Scale the embeddings
        x *= torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        # Add positional encoding
        x = self.positional_encoder(x)
        
        # Generate mask for the decoder to prevent attending to future positions
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        tgt_mask.to(x.device)
        output = self.decoder(x, x, tgt_mask=tgt_mask) 
        output = self.fc_out(output)
        return output


    
    def fit(self, 
            x: np.ndarray, 
            epochs=5000, 
            batch_size=32, 
            lr=0.001,
            patience=10, 
            min_delta=0.001,
            validation_split=0.2,
            verbose=0, 
            **kwargs) -> np.ndarray:
        """
        Fit the transformer model to the data. 

        Early stopping based on https://medium.com/@vrunda.bhattbhatt/a-step-by-step-guide-to-early-stopping-in-tensorflow-and-pytorch-59c1e3d0e376

        Parameters
        ----------
        x : np.ndarray
            (n, timesteps, embed_dim) Trajectories tensor.
        epochs : int, optional
            number of epochs to train the model, by default 2000
        batch_size : int, optional
            batch size for training, by default 32. After partitioning validation set, if batch_size > x.shape[0], batch_size = x.shape[0].
        patience : int, optional
            number of epochs to wait for improvement before stopping training, by default 10. Recommended to be between 10-100, typically 10 or 20.
        min_delta : float, optional
            minimum change in loss to be considered as improvement, by default 0.01
        validation_split : float, optional
            ratio of validation data to split from the training data, by default 0.2
        verbose : int, optional
            verbosity level, by default 0 
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        validation_size = int(x.shape[0] * validation_split)
        train_size = len(x) - validation_size
        train, val = random_split(x, [len(x)-validation_size, validation_size])
        # the inputs to the transformer will be the offset sequence
        # X should be the same as the val but truncated by 1 timestep
        x_val = torch.tensor(np.array(val[:][:-1, :]), dtype=torch.float32).to(self.device) 
        # the outputs will be the original sequence offset by 1 timestep
        y_val = torch.tensor(np.array(val[:][1:, :]), dtype=torch.float32).to(self.device) 

        
       
        if batch_size > train_size:
            batch_size = train_size
        x_train = torch.tensor(np.array(train[:][:-1, :]), dtype=torch.float32)

        y_train = torch.tensor(np.array(train[:][1:, :]), dtype=torch.float32)

        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        best_model_weights = None
        early_stop_p = patience

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            for input_seq, target_seq in dataloader:
                input_seq, target_seq = input_seq.to(self.device), target_seq.to(self.device)

                optimizer.zero_grad()
                output = self.forward(input_seq)
                
                loss = self.criterion(output, target_seq)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

                del input_seq, target_seq, output, loss


            self.eval()  # Set model to evaluation mode
            with torch.no_grad():  # Disable gradient calculation for validation
                # Make predictions on validation set 
                val_outputs = self(x_val).detach()
                val_loss = self.criterion(val_outputs, y_val).item()
            
            if verbose > 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}')

            # Early stopping
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                best_model_weights = copy.deepcopy(self.state_dict())  # Deep copy here      
                early_stop_p = patience  # Reset patience counter
            else:
                early_stop_p -= 1
                if early_stop_p == 0:
                    if verbose > 0:
                        print(f'Early stopping at epoch {epoch+1}')
                    break
            
            if self.device == 'mps':
                torch.mps.empty_cache()
            elif self.device == 'cuda':
                torch.cuda.empty_cache()
    
        # Load the best model weights
        self.load_state_dict(best_model_weights)
        


    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        """Predict the future trajectory of the system given the initial state.

        Parameters
        ----------
        x0 : np.ndarray
            (n, embed_dim) Initial conditions of the system.
        timesteps : int
            Number of timesteps, minus one, to predict into the future.

        Returns
        -------
        np.ndarray
            (n, timesteps, embed_dim) trajectories tensor, including the initial conditions.
        """
        self.eval()
        with torch.no_grad():
            initial_conditions = torch.tensor(x0, dtype=torch.float32).to(self.device) 
            next_input = initial_conditions.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)
            trajectory = next_input  
            # Iteratively predict future values
            for _ in range(timesteps - 1):

                with torch.no_grad():
                    output = self.forward(next_input)
                print(output.shape)

                next_input = output[:, -1:, :]  # Shape: (batch_size, 1, input_dim)
                trajectory = torch.cat([trajectory, next_input], dim=1)

        return trajectory.cpu().numpy()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, start_from_epoch=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_weights = None
        self.start_from_epoch = start_from_epoch

    def early_stop(self, epoch, validation_loss, weights):
        if epoch < self.start_from_epoch:
            return False
        
        if weights is None:
            self.best_weights = weights
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.best_weights = weights
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False